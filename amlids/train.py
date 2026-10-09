"""Batching, training with early stopping, evaluation and metrics."""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN kernels: identical results when a run is repeated
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Store:
    """Holds the scaled arrays of one dataset plus optional synthetic (SMOTE) sequences.
    Ids >= N refer to synthetic samples."""

    def __init__(self, xn, xc, seq, y, synth=None):
        self.N = len(y)
        self.xn = np.vstack([xn, np.zeros((1, xn.shape[1]), np.float32)])
        self.xc = np.vstack([xc, np.zeros((1, xc.shape[1]), np.int64)])
        self.seq, self.y = seq, y
        self.synth = synth

    def synthetic_ids(self):
        return np.arange(self.N, self.N + len(self.synth[2])) if self.synth else np.array([], int)

    def labels(self, ids):
        out = np.empty(len(ids), dtype=np.int64)
        r = ids < self.N
        out[r] = self.y[ids[r]]
        if (~r).any():
            out[~r] = self.synth[2][ids[~r] - self.N]
        return out

    def batch(self, ids, device):
        r = ids < self.N
        s = self.seq[ids[r]]
        xn, xc = self.xn[s], self.xc[s]
        if (~r).any():
            k = ids[~r] - self.N
            xn = np.concatenate([xn, self.synth[0][k]])
            xc = np.concatenate([xc, self.synth[1][k]])
            order = np.concatenate([np.where(r)[0], np.where(~r)[0]])
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            xn, xc = xn[inv], xc[inv]
        return (torch.from_numpy(xn).to(device, non_blocking=True),
                torch.from_numpy(xc).to(device, non_blocking=True),
                torch.from_numpy(self.labels(ids)).to(device, non_blocking=True))


def iterate(ids, bs, shuffle, rng=None):
    ids = np.asarray(ids)
    if shuffle:
        ids = ids[rng.permutation(len(ids))]
    for i in range(0, len(ids), bs):
        yield ids[i:i + bs]


def class_weights(y, n_classes):
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = w / w[cnt > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, store, ids, device, bs=2048):
    model.eval()
    probs = []
    for b in iterate(ids, bs, False):
        xn, xc, _ = store.batch(b, device)
        probs.append(F.softmax(model(xn, xc), dim=1).float().cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 1))


def metrics(y, prob, classes):
    pred = prob.argmax(1)
    n = len(classes)
    present = sorted(set(np.unique(y).tolist()))          # macro over classes present in y
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=present, average='macro',
                                                 zero_division=0)
    _, rc, _, sup = precision_recall_fscore_support(y, pred, labels=range(n), zero_division=0)
    benign = [i for i, c in enumerate(classes) if c in ('Benign', 'Normal')]
    out = dict(accuracy=accuracy_score(y, pred) * 100, precision=p * 100, recall=r * 100,
               f1=f * 100, per_class_recall={c: rc[i] * 100 for i, c in enumerate(classes)
                                             if sup[i] > 0})
    if benign:
        b = benign[0]
        is_att, pred_att = y != b, pred != b
        tn = np.sum(~is_att & ~pred_att); fp = np.sum(~is_att & pred_att)
        fn = np.sum(is_att & ~pred_att); tp = np.sum(is_att & pred_att)
        out['fpr'] = fp / max(fp + tn, 1) * 100
        out['fnr'] = fn / max(fn + tp, 1) * 100
    present = [i for i in range(n) if (y == i).any()]
    try:
        if len(present) == 2 and n == 2:
            out['auc_roc'] = roc_auc_score(y, prob[:, 1]) * 100
            out['auc_pr'] = average_precision_score(y, prob[:, 1]) * 100
        elif len(present) > 2:
            Y = np.eye(n)[y][:, present]
            P = prob[:, present]
            out['auc_roc'] = roc_auc_score(Y, P, average='macro') * 100
            out['auc_pr'] = average_precision_score(Y, P, average='macro') * 100
    except ValueError:
        pass
    out['confusion'] = confusion_matrix(y, pred, labels=range(n)).tolist()
    return out


def freeze_norm_and_dropout(model):
    """cuDNN LSTM backward requires train mode; keep BN statistics and dropout fixed."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
            m.eval()


def train_model(model, store, tr_ids, va_ids, n_classes, device, lr=1e-3, wd=1e-5, bs=256,
                max_epochs=200, patience=20, steps_per_epoch=None, seed=0, log=print):
    """Adam + weighted CE + ReduceLROnPlateau; early stopping on validation macro-F1.
    Returns the best model and per-epoch curves (for Figure 5)."""
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    w = class_weights(store.labels(tr_ids), n_classes).to(device)
    best, best_f1, bad = None, -1, 0
    curves = []
    va_y = store.labels(va_ids)
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for step, b in enumerate(iterate(tr_ids, bs, True, rng)):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            xn, xc, y = store.batch(b, device)
            out = model(xn, xc)
            loss = F.cross_entropy(out, y, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() * len(b); tc += (out.argmax(1) == y).sum().item(); tn += len(b)
        prob = predict(model, store, va_ids, device)
        vloss = F.nll_loss(torch.log(torch.tensor(prob).clamp_min(1e-9)),
                           torch.tensor(va_y), weight=w.cpu()).item()
        m = metrics(va_y, prob, [str(i) for i in range(n_classes)])
        curves.append(dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn * 100,
                           val_loss=vloss, val_acc=m['accuracy'], val_f1=m['f1'],
                           lr=opt.param_groups[0]['lr'], sec=time.time() - t0))
        log(f'  ep {ep:3d} loss {tl / tn:.4f} val_loss {vloss:.4f} val_acc {m["accuracy"]:.2f} '
            f'val_f1 {m["f1"]:.2f} ({time.time() - t0:.0f}s)')
        sched.step(m['f1'])
        if m['f1'] > best_f1 + 1e-4:
            best_f1, best, bad = m['f1'], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return model, curves


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
