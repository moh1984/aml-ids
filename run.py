"""AML-IDS experiments.  See README.md for the command that produces each table.

  python run.py indomain  --dataset cicids2017 --model aml_ids
  python run.py classical --dataset cicids2017 --model rf
  python run.py cross     --source cicids2017 --target cicids2018
  python run.py phases    --dataset cicids2017 --phases "Monday,Tuesday,Wednesday|Thursday,Friday"
  python run.py stream    --dataset cicids2017 --phases "Monday,Tuesday,Wednesday|Thursday,Friday"
  python run.py latency   --dataset cicids2017
  python run.py branches  --dataset cicids2017
  python run.py stats / plots / tables
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from amlids import data as D
from amlids.continual import ADWINDetector, EWC, PageHinkley, ReplayBuffer, adapt
from amlids.models import build_model, count_params
from amlids.data import val_split
from amlids.train import (Store, iterate, metrics, predict, save_json, set_seed, train_model)


def done(path, args):
    """Skip work whose result file already exists (resume after a disconnected session)."""
    if os.path.exists(path) and not args.overwrite:
        log(f'skip (exists): {path}')
        return True
    return False


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# --------------------------------------------------------------------------
def split_val(t, seq, tr, seed, args):
    """10% validation split for early stopping. Default (used in the paper): random stratified sequences.
    --block-val: contiguous blocks; with --purge-boundaries, training sequences whose context contains
    validation records are removed too."""
    if getattr(args, 'block_val', False):
        tr2, va = D.val_split_blocks(t, tr, 0.1, seed)
        if getattr(args, 'purge_boundaries', False):
            tr2, _ = D.purge_boundaries(tr2, va, seq)
        return tr2, va
    return val_split(t.y, tr, 0.1, seed)


def prepare_fold(t, seq, tr, args, seed, smote):
    """Fit preprocessing on training ids, build the Store (with SMOTE if requested)."""
    pp = D.Preprocessor().fit(t, tr)
    xn, xc = pp.transform(t)
    synth = None
    if smote:
        synth = D.smote_sequences(xn, xc, seq, t.y, tr, args.smote_min, args.smote_target,
                                  seed=seed)
    store = Store(xn, xc, seq, t.y, synth)
    return pp, store


def official_or_folds(t, cfg_ds, n, seed_base):
    if cfg_ds['format'] == 'nsl':               # official KDDTrain+/KDDTest+, repeated seeds
        tr = np.where(t.files == 'train')[0]
        te = np.where(t.files == 'test')[0]
        return [(tr, te)] * n
    return D.make_folds(t, n_splits=n, seed=seed_base, chunk=5000)


def device_of(args):
    return torch.device(args.device if args.device else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))


def train_kwargs(args):
    return dict(lr=args.lr, wd=args.wd, bs=args.bs, max_epochs=args.max_epochs,
                patience=args.patience, steps_per_epoch=args.steps_per_epoch)


# --------------------------------------------------------------------------
def cmd_indomain(args):
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, binary=args.binary, max_rows=args.max_rows)
    log(f'{args.dataset}: {len(t)} rows, {t.num.shape[1]} numeric + {t.cat.shape[1]} categorical '
        f'features, classes={t.classes}')
    seq = D.build_sequences(t, args.T)
    dev = device_of(args)
    folds = official_or_folds(t, cfg[args.dataset], args.folds, args.seed)
    out = f'{args.out}/indomain/{args.dataset}/{args.tag or args.model}'
    for k, (tr, te) in enumerate(folds):
        seed = args.seed + k
        if done(f'{out}/fold{k}.json', args):
            continue
        set_seed(seed)
        if args.purge_boundaries:
            tr, n_purged = D.purge_boundaries(tr, te, seq)
            log(f'fold {k}: purged {n_purged} training sequences whose context overlaps the test fold')
        tr2, va = split_val(t, seq, tr, seed, args)
        pp, store = prepare_fold(t, seq, tr2, args, seed, args.smote)
        tr_ids = np.concatenate([tr2, store.synthetic_ids()])
        model = build_model(args.model, t.num.shape[1], pp.cardinalities, len(t.classes), args.T)
        log(f'fold {k}: train {len(tr_ids)} (synthetic {len(store.synthetic_ids())}) '
            f'val {len(va)} test {len(te)} params {count_params(model) / 1e6:.2f}M')
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, curves = train_model(model, store, tr_ids, va, len(t.classes), dev, seed=seed,
                                    log=log, **train_kwargs(args))
        train_min = (time.time() - t0) / 60
        prob = predict(model, store, te, dev)
        m = metrics(t.y[te], prob, t.classes)
        m.update(fold=k, train_min=train_min, epochs=len(curves), classes=t.classes,
                 params_m=count_params(model) / 1e6,
                 gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else None))
        log(f'fold {k}: acc {m["accuracy"]:.2f} f1 {m["f1"]:.2f} fpr {m.get("fpr", float("nan")):.2f}')
        save_json(m, f'{out}/fold{k}.json')
        save_json(curves, f'{out}/curves{k}.json')
        np.savez_compressed(f'{out}/preds{k}.npz', y=t.y[te], prob=prob.astype(np.float32), idx=te)
        torch.save(model.state_dict(), f'{out}/model{k}.pt')


def cmd_classical(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, binary=args.binary, max_rows=args.max_rows)
    folds = official_or_folds(t, cfg[args.dataset], args.folds, args.seed)
    out = f'{args.out}/indomain/{args.dataset}/{args.model}'
    for k, (tr, te) in enumerate(folds):
        seed = args.seed + k
        if done(f'{out}/fold{k}.json', args):
            continue
        pp = D.Preprocessor().fit(t, tr)
        xn, xc = pp.transform(t)
        X = np.hstack([xn, xc]).astype(np.float32)
        t0 = time.time()
        if args.model == 'svm':            # RBF SVM is O(n^2): stratified subsample
            sub = D.subsample(tr, t.y, min(1.0, args.svm_n / len(tr)), seed)
            clf = SVC(kernel='rbf', probability=True, class_weight='balanced',
                      random_state=seed).fit(X[sub], t.y[sub])
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=2, class_weight='balanced',
                                         random_state=seed).fit(X[tr], t.y[tr])
        prob = np.zeros((len(te), len(t.classes)), np.float32)
        prob[:, clf.classes_] = clf.predict_proba(X[te])
        m = metrics(t.y[te], prob, t.classes)
        m.update(fold=k, train_min=(time.time() - t0) / 60, classes=t.classes)
        log(f'{args.model} fold {k}: acc {m["accuracy"]:.2f} f1 {m["f1"]:.2f}')
        save_json(m, f'{out}/fold{k}.json')
        np.savez_compressed(f'{out}/preds{k}.npz', y=t.y[te], prob=prob, idx=te)


# --------------------------------------------------------------------------
def concat_tables(a, b):
    off = 0 if a.groups is None else a.groups.max() + 1
    ga = a.groups if a.groups is not None else np.zeros(len(a), np.int64)
    gb = (b.groups + off + 1) if b.groups is not None else np.full(len(b), off + 1, np.int64)
    t = D.Table(np.vstack([a.num, b.num]), np.vstack([a.cat, b.cat]), np.concatenate([a.y, b.y]),
                a.classes, np.concatenate([a.files, b.files]),
                np.concatenate([a.order_key, b.order_key + a.order_key.max() + 1]),
                np.concatenate([ga, gb]), a.num_names, a.cat_names)
    return t, np.arange(len(a)), np.arange(len(a), len(a) + len(b))


def cmd_cross(args):
    cfg = D.load_config(args.config)
    s = D.load_dataset(args.source, cfg, binary=args.binary, max_rows=args.max_rows)
    tg = D.load_dataset(args.target, cfg, binary=args.binary, max_rows=args.max_rows)
    classes = s.classes + [c for c in tg.classes if c not in s.classes]
    for tbl in (s, tg):                       # common label space
        tbl.y = np.array([classes.index(tbl.classes[i]) for i in range(len(tbl.classes))])[tbl.y]
        tbl.classes = classes
    common, only_s, only_t = D.align_features(s, tg)
    log(f'shared features: {len(common)}; source-only: {only_s}; target-only: {only_t}')
    log(f'classes: {classes}')
    t, S, Tg = concat_tables(s, tg)
    seq = D.build_sequences(t, args.T)
    dev = device_of(args)
    out = f'{args.out}/cross/{args.source}__{args.target}{"_bin" if args.binary else ""}'
    for k in range(args.folds):
        seed = args.seed + k
        if done(f'{out}/seed{k}.json', args):
            continue
        set_seed(seed)
        s_tr, s_te = D.make_folds(s, 5, seed)[0]
        g_tr, g_te = D.make_folds(tg, 5, seed)[0]
        s_tr, s_te, g_tr, g_te = S[s_tr], S[s_te], Tg[g_tr], Tg[g_te]
        s_tr2, s_va = split_val(t, seq, s_tr, seed, args)
        pp, store = prepare_fold(t, seq, s_tr2, args, seed, args.smote)
        tr_ids = np.concatenate([s_tr2, store.synthetic_ids()])
        model = build_model(args.model, t.num.shape[1], pp.cardinalities, len(classes), args.T)
        model, _ = train_model(model, store, tr_ids, s_va, len(classes), dev, seed=seed, log=log,
                               **train_kwargs(args))
        zs = metrics(t.y[g_te], predict(model, store, g_te, dev), classes)
        src_before = metrics(t.y[s_te], predict(model, store, s_te, dev), classes)
        base = {n: v.clone() for n, v in model.state_dict().items()}
        lab = D.subsample(g_tr, t.y, args.label_frac, seed)
        r = dict(seed=seed, zero_shot=zs, source_before=src_before, n_labels=len(lab),
                 shared_features=common)
        # all adaptation variants from the same trained source model:
        # ewc = replay + EWC (full AML-IDS) | replay = buffer only | ewconly = EWC only | naive
        for var in args.adapt_configs.split(','):
            model.load_state_dict(base)
            use_buf, use_ewc = var in ('ewc', 'replay'), var in ('ewc', 'ewconly')
            buf = None
            if use_buf:
                buf = ReplayBuffer(args.buffer, args.balanced, seed)
                buf.add(s_tr2, t.y[s_tr2])
            ewc = None
            if use_ewc:
                ewc = EWC(model)
                ewc.consolidate(store, s_tr2, dev, n=args.fisher_n, seed=seed)
            sec = adapt(model, store, lab, dev, buf, ewc, args.lam if use_ewc else 0.0,
                        args.adapt_epochs, args.adapt_lr, args.bs, 0.3 if buf else 0.0, seed)
            ad = metrics(t.y[g_te], predict(model, store, g_te, dev), classes)
            sa = metrics(t.y[s_te], predict(model, store, s_te, dev), classes)
            r[f'adapted_{var}'] = ad
            r[f'source_after_{var}'] = sa
            r[f'adapt_sec_{var}'] = sec
            log(f'seed {seed} {var}: target {zs["accuracy"]:.2f} -> {ad["accuracy"]:.2f} '
                f'(F1 {zs["f1"]:.1f} -> {ad["f1"]:.1f}); source F1 {src_before["f1"]:.1f} -> {sa["f1"]:.1f}')
        r['adapted'] = r.get('adapted_ewc', next(v for k, v in r.items() if k.startswith('adapted_')))
        save_json(r, f'{out}/seed{k}.json')


# --------------------------------------------------------------------------
def phase_splits(t, phases, seed):
    tr, te = [], []
    for p in phases:
        idx = D.phase_indices(t, p)
        if len(idx) == 0:
            raise ValueError(f'No rows for phase {p}; check file names / --phases')
        sub = D.Table(t.num[idx], t.cat[idx], t.y[idx], t.classes, t.files[idx], t.order_key[idx],
                      None if t.groups is None else t.groups[idx], t.num_names, t.cat_names)
        a, b = D.make_folds(sub, 5, seed)[0]
        tr.append(idx[a]); te.append(idx[b])
    return tr, te


def parse_phases(s):
    return [p.split(',') for p in s.split('|')]


def base_model_for_phases(t, seq, args, dev, seed, tr0):
    tr2, va = split_val(t, seq, tr0, seed, args)
    pp, store = prepare_fold(t, seq, tr2, args, seed, args.smote)
    model = build_model(args.model, t.num.shape[1], pp.cardinalities, len(t.classes), args.T)
    model, _ = train_model(model, store, np.concatenate([tr2, store.synthetic_ids()]), va,
                           len(t.classes), dev, seed=seed, log=log, **train_kwargs(args))
    return model, store, tr2


def cmd_phases(args):
    """Tables 8 and 9: sequential phases; vary lambda_EWC, buffer size and balancing."""
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, max_rows=args.max_rows)
    seq = D.build_sequences(t, args.T)
    dev = device_of(args)
    phases = parse_phases(args.phases)
    configs = [(float(l), int(b), bal) for l in args.lams.split(',')
               for b in args.buffer_sizes.split(',') for bal in ([False, True] if args.try_balanced
                                                                 else [args.balanced])]
    out = f'{args.out}/phases/{args.dataset}/{args.tag or "run"}'
    for k in range(args.folds):
        seed = args.seed + k
        if all(done(f'{out}/seed{k}_lam{l:g}_buf{b}_bal{int(c)}.json', args) for l, b, c in configs):
            continue
        set_seed(seed)
        tr, te = phase_splits(t, phases, seed)
        model, store, tr0 = base_model_for_phases(t, seq, args, dev, seed, tr[0])
        base = {n: v.clone() for n, v in model.state_dict().items()}
        ev = lambda x: metrics(t.y[x], predict(model, store, x, dev), t.classes)
        m0 = [ev(x) for x in te]
        for lam, bsz, bal in configs:
            model.load_state_dict(base)
            buf = ReplayBuffer(bsz, bal, seed) if bsz > 0 else None
            if buf:
                buf.add(tr0, t.y[tr0])
            ewc = EWC(model)
            ewc.consolidate(store, tr0, dev, n=args.fisher_n, seed=seed)
            learned = {0: m0[0]}
            secs = []
            for j in range(1, len(phases)):
                lab = D.subsample(tr[j], t.y, args.label_frac, seed)
                secs.append(adapt(model, store, lab, dev, buf, ewc, lam, args.adapt_epochs,
                                  args.adapt_lr, args.bs, 0.3 if buf else 0.0, seed))
                if buf:
                    buf.add(lab, t.y[lab])
                ewc.consolidate(store, lab, dev, n=args.fisher_n, seed=seed)
                learned[j] = ev(te[j])
            final = [ev(x) for x in te]
            old = range(len(phases) - 1)
            r = dict(seed=seed, lam=lam, buffer=bsz, balanced=bal,
                     acc_before=[m['accuracy'] for m in m0],
                     f1_before=[m['f1'] for m in m0],
                     acc_final=[m['accuracy'] for m in final],
                     f1_final=[m['f1'] for m in final],
                     old_acc=float(np.mean([final[j]['accuracy'] for j in old])),
                     new_acc=final[-1]['accuracy'],
                     old_f1=float(np.mean([final[j]['f1'] for j in old])),
                     new_f1=final[-1]['f1'],
                     forgetting=float(np.mean([learned[j]['accuracy'] - final[j]['accuracy'] for j in old])),
                     forgetting_f1=float(np.mean([learned[j]['f1'] - final[j]['f1'] for j in old])),
                     per_class_before=[m['per_class_recall'] for m in m0],
                     per_class_final=[m['per_class_recall'] for m in final],
                     adapt_min=float(np.sum(secs) / 60))
            log(f'seed {seed} lam {lam:g} buf {bsz} bal {bal}: old F1 {r["old_f1"]:.1f} new F1 {r["new_f1"]:.1f} '
                f'forgetting F1 {r["forgetting_f1"]:.1f} (acc {r["forgetting"]:.2f}) ({r["adapt_min"]:.1f} min)')
            save_json(r, f'{out}/seed{k}_lam{lam:g}_buf{bsz}_bal{int(bal)}.json')


def cmd_stream(args):
    """Table 7 drift rows: prequential evaluation on a time-ordered stream of phase 2."""
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, max_rows=args.max_rows)
    seq = D.build_sequences(t, args.T)
    dev = device_of(args)
    phases = parse_phases(args.phases)
    out = f'{args.out}/stream/{args.dataset}/{args.tag or "run"}'
    for k in range(args.folds):
        seed = args.seed + k
        if all(done(f'{out}/seed{k}_{c.replace(":", "-")}.json', args) for c in args.configs.split(',')):
            continue
        set_seed(seed)
        tr, te = phase_splits(t, phases[:1], seed)
        model, store, tr0 = base_model_for_phases(t, seq, args, dev, seed, tr[0])
        base = {n: v.clone() for n, v in model.state_dict().items()}
        # reference error level: base model on held-out data of the training phase
        ref_err = 100 - metrics(t.y[te[0]], predict(model, store, te[0], dev), t.classes)['accuracy']
        stream = D.phase_indices(t, phases[1])
        stream = stream[np.argsort(t.order_key[stream], kind='stable')]
        for conf in args.configs.split(','):
            det_name, strat = conf.split(':')
            # none: no update | naive: fine-tune only | ewconly: EWC, no replay
            # replay: replay buffer, lambda=0 | ewc: replay buffer + EWC (full AML-IDS)
            use_buf = strat in ('replay', 'ewc')
            use_ewc = strat in ('ewconly', 'ewc')
            lam = args.lam if use_ewc else 0.0
            model.load_state_dict(base)
            rng = np.random.default_rng(seed)
            det = {'ph': lambda: PageHinkley(args.ph_delta, args.ph_lambda),
                   'adwin': lambda: ADWINDetector(), 'none': lambda: None}[det_name]()
            buf = ReplayBuffer(args.buffer, args.balanced, seed) if use_buf else None
            if buf:
                buf.add(tr0, t.y[tr0])
            ewc = EWC(model) if use_ewc else None
            if ewc:
                ewc.consolidate(store, tr0, dev, n=args.fisher_n, seed=seed)
            probs, window, events, secs, errs = [], [], 0, 0.0, []
            baseline, recover, since = None, False, 0

            def do_adapt():
                nonlocal secs, events, window, since
                new = np.concatenate(window[-args.window:])
                if len(new) == 0:
                    return
                secs += adapt(model, store, new, dev, buf, ewc, lam, args.adapt_epochs,
                              args.adapt_lr, args.bs, 0.3 if buf else 0.0, seed)
                if buf:
                    buf.add(new, t.y[new])
                if ewc:
                    ewc.consolidate(store, new, dev, n=args.fisher_n, seed=seed)
                events += 1
                window, since = [], 0

            for b in iterate(stream, args.stream_batch, False):
                p = predict(model, store, b, dev)
                probs.append(p)
                lab = b[rng.random(len(b)) < args.label_frac]
                window.append(lab)
                if det is None or strat == 'none' or len(lab) == 0:
                    continue
                err = float((p[np.isin(b, lab)].argmax(1) != t.y[lab]).mean() * 100)
                errs.append(err)
                since += 1
                if det.update(err):
                    baseline = ref_err
                    recover = True
                    do_adapt()
                elif recover and since >= args.recover_every:
                    # recovery mode: keep adapting while error stays above the pre-drift level
                    recent = float(np.mean(errs[-args.recover_every:]))
                    if recent > baseline + args.recover_margin:
                        do_adapt()
                    else:
                        recover = False
            m = metrics(t.y[stream], np.concatenate(probs), t.classes)
            ret = metrics(t.y[te[0]], predict(model, store, te[0], dev), t.classes)
            r = dict(seed=seed, config=conf, stream=m, retention=ret, drift_events=events,
                     adapt_min=secs / 60, baseline_err=baseline)
            log(f'seed {seed} {conf}: stream acc {m["accuracy"]:.2f} F1 {m["f1"]:.2f} '
                f'events {events} retention acc {ret["accuracy"]:.2f} F1 {ret["f1"]:.2f}')
            save_json(r, f'{out}/seed{k}_{conf.replace(":", "-")}.json')


# --------------------------------------------------------------------------
def cmd_latency(args):
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, max_rows=args.max_rows)
    ppf = D.packets_per_flow(t)
    dev = device_of(args)
    cards = [len(set(t.cat[:, j])) + 1 for j in range(t.cat.shape[1])]
    res = {'packets_per_flow': ppf, 'device': str(dev)}
    for name in args.models.split(','):
        model = build_model(name, t.num.shape[1], cards, len(t.classes), args.T).to(dev).eval()
        r = {'params_m': count_params(model) / 1e6}
        for bs in (1, 1024):
            xn = torch.randn(bs, args.T, t.num.shape[1], device=dev)
            xc = torch.zeros(bs, args.T, t.cat.shape[1], dtype=torch.long, device=dev)
            with torch.no_grad():
                for _ in range(50):
                    model(xn, xc)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                reps = args.reps if bs == 1 else max(20, args.reps // 20)
                t0 = time.perf_counter()
                for _ in range(reps):
                    model(xn, xc)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
            per_flow_ms = (time.perf_counter() - t0) / reps / bs * 1000
            r[f'ms_per_flow_b{bs}'] = per_flow_ms
            r[f'flows_per_s_b{bs}'] = 1000 / per_flow_ms
            if ppf:
                r[f'us_per_packet_b{bs}'] = per_flow_ms * 1000 / ppf
                r[f'packets_per_s_b{bs}'] = 1000 / per_flow_ms * ppf
        if dev.type == 'cuda':                       # peak memory of one training step, batch 256
            torch.cuda.reset_peak_memory_stats()
            model.train()
            xn = torch.randn(256, args.T, t.num.shape[1], device=dev)
            xc = torch.zeros(256, args.T, t.cat.shape[1], dtype=torch.long, device=dev)
            model(xn, xc).sum().backward()
            r['train_step_gpu_gb'] = torch.cuda.max_memory_allocated() / 1e9
        res[name] = r
        log(name, json.dumps(r))
    save_json(res, f'{args.out}/latency/{args.dataset}.json')


def cmd_branches(args):
    """Per-branch, per-class contribution: zero one CNN branch at a time (Section 5.4)."""
    cfg = D.load_config(args.config)
    t = D.load_dataset(args.dataset, cfg, binary=args.binary, max_rows=args.max_rows)
    seq = D.build_sequences(t, args.T)
    dev = device_of(args)
    src = f'{args.out}/indomain/{args.dataset}/{args.tag or "aml_ids"}'
    folds = official_or_folds(t, cfg[args.dataset], args.folds, args.seed)
    res = []
    for k, (tr, te) in enumerate(folds):
        seed = args.seed + k
        tr2, _ = split_val(t, seq, tr, seed, args)
        pp, store = prepare_fold(t, seq, tr2, args, seed, False)
        model = build_model('aml_ids', t.num.shape[1], pp.cardinalities, len(t.classes), args.T)
        model.load_state_dict(torch.load(f'{src}/model{k}.pt', map_location='cpu'))
        model.to(dev)
        full = metrics(t.y[te], predict(model, store, te, dev), t.classes)['per_class_recall']
        row = {'fold': k, 'full': full}
        for i, s in enumerate((3, 5, 7)):
            model.branch_mask.fill_(1.0)
            model.branch_mask[i] = 0.0
            m = metrics(t.y[te], predict(model, store, te, dev), t.classes)['per_class_recall']
            row[f'without_{s}x{s}'] = {c: full[c] - m.get(c, 0) for c in full}
        model.branch_mask.fill_(1.0)
        res.append(row)
        log(f'fold {k}: recall drop per class when a branch is removed: ' +
            json.dumps({key: {c: round(v, 2) for c, v in row[key].items()}
                        for key in row if key.startswith('without')}))
    save_json(res, f'{args.out}/branches/{args.dataset}.json')


# --------------------------------------------------------------------------
def load_runs(pattern, key):
    vals = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        for part in key.split('.'):
            r = r[part]
        vals.append(float(r))
    return np.array(vals)


def cmd_stats(args):
    """Paired t-test (Bonferroni), Cohen's d, and McNemar on pooled predictions."""
    from scipy.stats import chi2, ttest_rel
    a = load_runs(args.a, args.key)
    b = load_runs(args.b or args.a, args.key_b or args.key)
    assert len(a) == len(b) and len(a) > 1, (len(a), len(b))
    t_, p = ttest_rel(a, b)
    d = (a - b).mean() / (a - b).std(ddof=1)
    print(f'A: {a.mean():.2f} ± {a.std(ddof=1):.2f}   B: {b.mean():.2f} ± {b.std(ddof=1):.2f}')
    print(f'paired t = {t_:.3f}, p = {p:.2e}, Bonferroni (m={args.m}) p = {min(1, p * args.m):.2e},'
          f' Cohen d = {d:.2f}')
    if args.preds_a and args.preds_b:
        n01 = n10 = 0
        for fa, fb in zip(sorted(glob.glob(args.preds_a)), sorted(glob.glob(args.preds_b))):
            A, B = np.load(fa), np.load(fb)
            assert (A['idx'] == B['idx']).all()
            ca, cb = A['prob'].argmax(1) == A['y'], B['prob'].argmax(1) == B['y']
            n01 += int((ca & ~cb).sum()); n10 += int((~ca & cb).sum())
        stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
        print(f'McNemar: n01={n01} n10={n10} chi2={stat:.2f} p={chi2.sf(stat, 1):.2e}')


def cmd_plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    src = f'{args.out}/indomain/{args.dataset}/{args.tag or "aml_ids"}'
    os.makedirs(f'{args.out}/figures', exist_ok=True)
    curves = [json.load(open(f)) for f in sorted(glob.glob(f'{src}/curves*.json'))]
    L = min(len(c) for c in curves)
    get = lambda k: np.mean([[e[k] for e in c[:L]] for c in curves], axis=0)
    ep = np.arange(1, L + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, get('train_loss'), label='Training'); ax[0].plot(ep, get('val_loss'), label='Validation')
    ax[0].set(title='(a) Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
    ax[1].plot(ep, get('train_acc'), label='Training'); ax[1].plot(ep, get('val_acc'), label='Validation')
    ax[1].set(title='(b) Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    stop = int(np.mean([np.argmax([e['val_f1'] for e in c]) + 1 for c in curves]))
    for a in ax:
        a.axvline(stop, ls='--', c='gray', label=f'Best epoch (mean {stop})')
        a.grid(alpha=.3); a.legend()
    fig.suptitle(f'Training and validation convergence, {args.dataset} ({len(curves)}-fold mean)')
    fig.tight_layout(); fig.savefig(f'{args.out}/figures/fig5_curves_{args.dataset}.png', dpi=300)

    P = [np.load(f) for f in sorted(glob.glob(f'{src}/preds*.npz'))]
    y = np.concatenate([p['y'] for p in P]); prob = np.concatenate([p['prob'] for p in P])
    classes = json.load(open(f'{src}/fold0.json'))['classes']
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, c in enumerate(classes):
        if (y == i).any() and (y != i).any():
            fpr, tpr, _ = roc_curve(y == i, prob[:, i])
            ax.plot(fpr, tpr, label=c)
    ax.plot([0, 1], [0, 1], 'k--', lw=.8)
    ax.set(title=f'ROC curves (one-vs-rest), {args.dataset}', xlabel='False positive rate',
           ylabel='True positive rate'); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f'{args.out}/figures/fig2_roc_{args.dataset}.png', dpi=300)

    cm = np.zeros((len(classes), len(classes)))
    for p in P:
        from sklearn.metrics import confusion_matrix
        cm += confusion_matrix(p['y'], p['prob'].argmax(1), labels=range(len(classes)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if cmn[i, j] > .5 else 'black')
    ax.set(title=f'Normalized confusion matrix, {args.dataset}', xlabel='Predicted', ylabel='True')
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(f'{args.out}/figures/confusion_{args.dataset}.png', dpi=300)
    log('figures written to', f'{args.out}/figures')


def ms(v):
    return f'{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}' if len(v) > 1 else f'{np.mean(v):.1f}'


def cmd_tables(args):
    lines = []
    for ds in sorted(glob.glob(f'{args.out}/indomain/*')):
        lines += [f'\n## In-domain: {os.path.basename(ds)}\n',
                  '| Model | Accuracy | Precision | Recall | F1 | FPR | AUC-ROC | Train (min) |',
                  '|---|---|---|---|---|---|---|---|']
        for md in sorted(glob.glob(f'{ds}/*')):
            R = [json.load(open(f)) for f in sorted(glob.glob(f'{md}/fold*.json'))]
            if not R:
                continue
            g = lambda k: [r[k] for r in R if k in r]
            lines.append(f'| {os.path.basename(md)} | {ms(g("accuracy"))} | {ms(g("precision"))} | '
                         f'{ms(g("recall"))} | {ms(g("f1"))} | {ms(g("fpr")) if g("fpr") else "-"} | '
                         f'{ms(g("auc_roc")) if g("auc_roc") else "-"} | {ms(g("train_min"))} |')
    for cd in sorted(glob.glob(f'{args.out}/cross/*')):
        R = [json.load(open(f)) for f in sorted(glob.glob(f'{cd}/seed*.json'))]
        if not R:
            continue
        lines += [f'\n## Cross-dataset: {os.path.basename(cd)}\n',
                  '| Setting | Target acc | Target F1 | Source F1 after (retention) |', '|---|---|---|---|',
                  f'| zero-shot | {ms([r["zero_shot"]["accuracy"] for r in R])} | '
                  f'{ms([r["zero_shot"]["f1"] for r in R])} | {ms([r["source_before"]["f1"] for r in R])} |']
        for var in sorted({k[8:] for r in R for k in r if k.startswith('adapted_')}):
            S = [r for r in R if f'adapted_{var}' in r]
            lines.append(f'| adapt: {var} | {ms([r[f"adapted_{var}"]["accuracy"] for r in S])} | '
                         f'{ms([r[f"adapted_{var}"]["f1"] for r in S])} | '
                         f'{ms([r[f"source_after_{var}"]["f1"] for r in S])} |')
    for pd_ in sorted(glob.glob(f'{args.out}/phases/*/*')):
        R = [json.load(open(f)) for f in sorted(glob.glob(f'{pd_}/*.json'))]
        lines += [f'\n## Phases: {pd_}\n', '| λ_EWC | Buffer | Balanced | Old F1 | New F1 | Forgetting F1 (pp) | '
                  'Old acc | New acc | Forgetting acc (pp) | Adapt (min) |', '|---|---|---|---|---|---|---|---|---|---|']
        keys = sorted({(r['lam'], r['buffer'], r['balanced']) for r in R})
        for key in keys:
            S = [r for r in R if (r['lam'], r['buffer'], r['balanced']) == key]
            g = lambda k: ms([r[k] for r in S]) if all(k in r for r in S) else '-'
            lines.append(f'| {key[0]:g} | {key[1]} | {key[2]} | {g("old_f1")} | {g("new_f1")} | '
                         f'{g("forgetting_f1")} | {g("old_acc")} | {g("new_acc")} | {g("forgetting")} | '
                         f'{g("adapt_min")} |')
    for sd in sorted(glob.glob(f'{args.out}/stream/*/*')):
        R = [json.load(open(f)) for f in sorted(glob.glob(f'{sd}/*.json'))]
        lines += [f'\n## Stream: {sd}\n', '| Config | Stream acc | Stream F1 | FPR | Retention acc | '
                  'Retention F1 | Events |', '|---|---|---|---|---|---|---|']
        for conf in sorted({r['config'] for r in R}):
            S = [r for r in R if r['config'] == conf]
            lines.append(f'| {conf} | {ms([r["stream"]["accuracy"] for r in S])} | '
                         f'{ms([r["stream"]["f1"] for r in S])} | '
                         f'{ms([r["stream"].get("fpr", 0) for r in S])} | '
                         f'{ms([r["retention"]["accuracy"] for r in S])} | '
                         f'{ms([r["retention"]["f1"] for r in S])} | '
                         f'{ms([r["drift_events"] for r in S])} |')
        for conf in sorted({r['config'] for r in R}):
            S = [r for r in R if r['config'] == conf]
            cls = sorted({c for r in S for c in r['stream']['per_class_recall']})
            lines.append(f'\n{conf} per-class recall — stream: ' + ', '.join(
                f'{c} {np.mean([r["stream"]["per_class_recall"].get(c, 0) for r in S]):.1f}' for c in cls) +
                ' | retention: ' + ', '.join(
                f'{c} {np.mean([r["retention"]["per_class_recall"].get(c, 0) for r in S]):.1f}'
                for c in sorted({c for r in S for c in r['retention']['per_class_recall']})))
    for lf in sorted(glob.glob(f'{args.out}/latency/*.json')):
        lines += [f'\n## Latency: {lf}\n', '```', json.dumps(json.load(open(lf)), indent=1), '```']
    txt = '\n'.join(lines)
    with open(f'{args.out}/tables.md', 'w') as f:
        f.write(txt)
    print(txt)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['indomain', 'classical', 'cross', 'phases', 'stream', 'latency',
                                    'branches', 'stats', 'plots', 'tables'])
    ap.add_argument('--config', default='configs/datasets.yaml')
    ap.add_argument('--dataset'); ap.add_argument('--source'); ap.add_argument('--target')
    ap.add_argument('--model', default='aml_ids')
    ap.add_argument('--tag', help='results sub-folder name (default: model name)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--device')
    ap.add_argument('--binary', action='store_true')
    ap.add_argument('--overwrite', action='store_true', help='recompute existing results')
    ap.add_argument('--block-val', action='store_true',
                    help='validation split from contiguous blocks instead of random sequences (strict model selection)')
    ap.add_argument('--purge-boundaries', action='store_true',
                    help='indomain: drop training sequences whose context contains test-fold records (strict CV)')
    ap.add_argument('--max-rows', type=int, default=None,
                    help='stratified subsample of rows after loading (for limited RAM)')
    ap.add_argument('--T', type=int, default=10, help='flows per sequence')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    # training
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--smote', dest='smote', action='store_true', default=True)
    ap.add_argument('--no-smote', dest='smote', action='store_false')
    ap.add_argument('--smote-min', type=int, default=1000)
    ap.add_argument('--smote-target', type=int, default=1000)
    ap.add_argument('--svm-n', type=int, default=50000)
    # adaptation
    ap.add_argument('--lam', type=float, default=5000)
    ap.add_argument('--lams', default='0,100,1000,5000,10000,50000')
    ap.add_argument('--buffer', type=int, default=10000)
    ap.add_argument('--adapt-configs', default='ewc,replay,ewconly,naive',
                    help='cross: adaptation variants evaluated from the same source model')
    ap.add_argument('--buffer-sizes', default='10000')
    ap.add_argument('--balanced', action='store_true')
    ap.add_argument('--try-balanced', action='store_true')
    ap.add_argument('--label-frac', type=float, default=0.10)
    ap.add_argument('--fisher-n', type=int, default=1000)
    ap.add_argument('--adapt-epochs', type=int, default=5)
    ap.add_argument('--adapt-lr', type=float, default=1e-4)
    ap.add_argument('--phases', default='Monday,Tuesday,Wednesday|Thursday,Friday')
    # stream
    ap.add_argument('--configs', default='none:none,ph:naive,ph:ewconly,ph:replay,ph:ewc,adwin:ewc')
    ap.add_argument('--stream-batch', type=int, default=2000)
    ap.add_argument('--window', type=int, default=20, help='batches of labels used per adaptation')
    ap.add_argument('--ph-delta', type=float, default=0.5)
    ap.add_argument('--ph-lambda', type=float, default=50)
    ap.add_argument('--recover-every', type=int, default=5, help='batches between recovery checks')
    ap.add_argument('--recover-margin', type=float, default=5.0,
                    help='keep adapting while error exceeds the pre-drift error by this many points')
    # latency
    ap.add_argument('--models', default='aml_ids,cnn_lstm_attn,transformer,mamba')
    ap.add_argument('--reps', type=int, default=1000)
    # stats
    ap.add_argument('--a'); ap.add_argument('--b'); ap.add_argument('--key', default='accuracy')
    ap.add_argument('--key-b'); ap.add_argument('--m', type=int, default=1)
    ap.add_argument('--preds-a'); ap.add_argument('--preds-b')
    args = ap.parse_args()
    globals()['cmd_' + args.cmd](args)


if __name__ == '__main__':
    main()
