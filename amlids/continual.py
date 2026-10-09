"""Drift-aware adaptation: Page-Hinkley / ADWIN detectors, EWC, reservoir replay."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .train import freeze_norm_and_dropout, iterate


class PageHinkley:
    """m_T = sum_t (e_t - mean_T - delta); M_T = min m_t; drift if m_T - M_T > lam.
    e_t is the per-batch error rate in percent (Section 3.4)."""

    def __init__(self, delta=0.5, lam=50.0, min_batches=5):
        self.delta, self.lam, self.min_batches = delta, lam, min_batches
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.M = 0, 0.0, 0.0, 0.0

    def update(self, e):
        self.n += 1
        self.mean += (e - self.mean) / self.n
        self.m += e - self.mean - self.delta
        self.M = min(self.M, self.m)
        if self.n >= self.min_batches and self.m - self.M > self.lam:
            self.reset()
            return True
        return False


class ADWINDetector:
    def __init__(self, delta=0.002):
        from river.drift import ADWIN
        self.d = ADWIN(delta=delta)

    def update(self, e):
        self.d.update(e / 100.0)
        return self.d.drift_detected


class ReplayBuffer:
    """Reservoir sampling (Vitter's algorithm R); optional per-class balanced quotas."""

    def __init__(self, size, class_balanced=False, seed=0):
        self.size, self.cb = size, class_balanced
        self.rng = np.random.default_rng(seed)
        self.items, self.seen = [], 0                   # global reservoir
        self.per, self.pseen = {}, {}                   # class-balanced reservoirs

    def add(self, ids, labels):
        if not self.cb:
            for i in ids:
                self.seen += 1
                if len(self.items) < self.size:
                    self.items.append(i)
                else:
                    j = self.rng.integers(0, self.seen)
                    if j < self.size:
                        self.items[j] = i
            return
        for c in np.unique(labels):
            self.per.setdefault(c, []); self.pseen.setdefault(c, 0)
        quota = max(1, self.size // len(self.per))
        for c in self.per:                              # shrink when a new class appears
            if len(self.per[c]) > quota:
                keep = self.rng.choice(len(self.per[c]), quota, replace=False)
                self.per[c] = [self.per[c][k] for k in keep]
        for i, c in zip(ids, labels):
            self.pseen[c] += 1
            if len(self.per[c]) < quota:
                self.per[c].append(i)
            else:
                j = self.rng.integers(0, self.pseen[c])
                if j < quota:
                    self.per[c][j] = i

    def ids(self):
        if self.cb:
            return np.array([i for v in self.per.values() for i in v], dtype=np.int64)
        return np.array(self.items, dtype=np.int64)

    def sample(self, k):
        a = self.ids()
        if len(a) == 0:
            return a
        return a[self.rng.integers(0, len(a), k)]


class EWC:
    """Diagonal empirical Fisher on the adapted parameters; online accumulation over phases."""

    def __init__(self, model, gamma=1.0):
        self.model, self.gamma = model, gamma
        self.params = model.adapt_parameters()
        self.fisher = None
        self.star = None

    def consolidate(self, store, ids, device, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        ids = ids[rng.permutation(len(ids))[:n]]
        freeze_norm_and_dropout(self.model)
        fis = [torch.zeros_like(p) for p in self.params]
        for i in ids:                                    # per-sample gradients
            xn, xc, y = store.batch(np.array([i]), device)
            self.model.zero_grad(set_to_none=True)
            F.nll_loss(F.log_softmax(self.model(xn, xc), 1), y).backward()
            for f, p in zip(fis, self.params):
                if p.grad is not None:
                    f += p.grad.detach() ** 2
        fis = [f / len(ids) for f in fis]
        self.fisher = fis if self.fisher is None else [self.gamma * a + b for a, b in
                                                       zip(self.fisher, fis)]
        self.star = [p.detach().clone() for p in self.params]

    def penalty(self):
        if self.fisher is None:
            return 0.0
        return sum((f * (p - s) ** 2).sum() for f, p, s in zip(self.fisher, self.params, self.star))


def adapt(model, store, new_ids, device, buffer=None, ewc=None, lam=5000.0, epochs=5, lr=1e-4,
          bs=256, replay_ratio=0.3, seed=0):
    """Fine-tune the final LSTM layer + head on new labelled data, mixing replayed samples
    and adding the EWC penalty (lam=0 or ewc=None -> naive fine-tuning). Returns seconds."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    params = model.adapt_parameters()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    n_new = int(round(bs * (1 - replay_ratio))) if buffer is not None else bs
    for _ in range(epochs):
        freeze_norm_and_dropout(model)
        for b in iterate(new_ids, n_new, True, rng):
            if buffer is not None and len(buffer.ids()):
                b = np.concatenate([b, buffer.sample(bs - len(b))])
            xn, xc, y = store.batch(b, device)
            loss = F.cross_entropy(model(xn, xc), y)
            if ewc is not None and lam > 0:
                loss = loss + lam / 2 * ewc.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time() - t0
