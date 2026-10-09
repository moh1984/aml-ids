"""Data pipeline for AML-IDS: loading, cleaning, label harmonisation, scaling,
sequence construction, fold generation and SMOTE for minority classes.

All statistics (scaling, category vocabularies, SMOTE) are fitted on training
indices only, to avoid information leakage.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------
# Label harmonisation
# --------------------------------------------------------------------------
def cic_label(raw):
    s = str(raw).lower()
    if 'benign' in s:
        return 'Benign'
    if 'web' in s or 'xss' in s or 'sql' in s:
        return 'WebAttack'
    if 'ddos' in s:
        return 'DDoS'
    if 'dos' in s:
        return 'DoS'
    if 'heartbleed' in s:
        return 'Other'
    if 'portscan' in s:
        return 'PortScan'
    if 'bot' in s:
        return 'Bot'
    if 'infilt' in s:
        return 'Infiltration'
    if 'brute' in s or 'patator' in s:
        return 'BruteForce'
    return 'Other'


NSL_MAP = {
    'normal': 'Normal',
    **{a: 'DoS' for a in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2',
                          'mailbomb', 'processtable', 'udpstorm']},
    **{a: 'Probe' for a in ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']},
    **{a: 'R2L' for a in ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy',
                          'warezclient', 'warezmaster', 'sendmail', 'named', 'snmpgetattack',
                          'snmpguess', 'xlock', 'xsnoop', 'worm']},
    **{a: 'U2R' for a in ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'ps',
                          'sqlattack', 'xterm', 'httptunnel']},
}

NSL_COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty']

BENIGN_NAMES = {'Benign', 'Normal'}

# --------------------------------------------------------------------------
# Canonical feature names (lets CICIDS2017 and CSE-CIC-IDS2018 share columns)
# --------------------------------------------------------------------------
_TOK = {'tot': ['total'], 'totlen': ['total', 'length'], 'pkts': ['packet'], 'pkt': ['packet'],
        'packets': ['packet'], 'byts': ['bytes'], 'len': ['length'], 'cnt': ['count'],
        'avg': ['average'], 'seg': ['segment'], 'blk': ['bulk'], 'b': ['bulk'], 'rt': ['rate'],
        'var': ['variance'], 'forward': ['fwd'], 'backward': ['bwd'], 'dst': ['destination'],
        'src': ['source'], 'of': [], 's': ['persec'], 'win': ['win']}


def canonical(name):
    toks = re.findall(r'[a-z0-9]+', str(name).lower())
    out = []
    for t in toks:
        out.extend(_TOK.get(t, [t]))
    return '_'.join(sorted(out))


# columns that identify a flow rather than describe it (never used as features)
ID_COLS = {canonical(c) for c in ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
                                  'Source Port', 'Src Port', 'Timestamp', 'Label', 'Protocol']}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_config(path='configs/datasets.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def _is_label(c):
    return str(c).strip().lower() == 'label'


def _read_csvs(patterns, max_rows=None, chunk=500_000):
    """Reads CSVs.  With max_rows, very large datasets (CSE-CIC-IDS2018) are sampled while
    reading so they never have to fit in memory: pass 1 counts labels, pass 2 keeps each row
    with a label-dependent probability (same fraction for every class, but at least ~200 rows
    of rare classes)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    files = sorted(sum([glob.glob(p) for p in patterns], []))
    if not files:
        raise FileNotFoundError(f'No files match {patterns}')
    prob = None
    if max_rows:
        counts = pd.Series(dtype=np.int64)
        for f in files:
            lab = pd.read_csv(f, usecols=_is_label, encoding='latin-1', dtype=str).iloc[:, 0]
            counts = counts.add(lab.str.strip().value_counts(), fill_value=0)
        counts = counts.drop(labels=[k for k in counts.index if k.lower() == 'label'],
                             errors='ignore')
        total = counts.sum()
        if total > max_rows:
            frac = max_rows / total
            prob = {k: min(1.0, max(frac, 200.0 / v)) for k, v in counts.items()}
    rng = np.random.default_rng(0)
    frames = []
    for f in files:
        if prob is None:
            parts = [pd.read_csv(f, low_memory=False, encoding='latin-1')]
        else:
            parts = []
            for df in pd.read_csv(f, low_memory=False, encoding='latin-1', chunksize=chunk):
                lab = df[[c for c in df.columns if _is_label(c)][0]].astype(str).str.strip()
                p = lab.map(prob).fillna(0.0).values
                parts.append(df[rng.random(len(df)) < p])
        df = pd.concat(parts, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
        df['__file'] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class Table:
    """Container for one cleaned dataset."""

    def __init__(self, num, cat, y, classes, files, order_key, groups, num_names, cat_names):
        self.num = num            # float32 (N, dn)
        self.cat = cat            # object/str (N, dc) raw categories
        self.y = y                # int64 (N,)
        self.classes = classes    # list[str]
        self.files = files        # np.array of source file names
        self.order_key = order_key
        self.groups = groups      # host-pair id per row or None
        self.num_names = num_names
        self.cat_names = cat_names

    def __len__(self):
        return len(self.y)


def load_dataset(name, cfg, binary=False, classes=None, max_rows=None):
    c = cfg[name]
    fmt = c['format']
    if fmt == 'nsl':
        frames = []
        for split in ('train', 'test'):
            df = pd.read_csv(c[split], header=None, names=NSL_COLS)
            df['__file'] = split
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        labels = df['label'].map(lambda s: NSL_MAP.get(str(s).strip(), 'Other'))
        cat_cols = ['protocol_type', 'service', 'flag']
        drop = {'label', 'difficulty', '__file'}
    elif fmt == 'unsw':
        df = _read_csvs(c['files'])
        labels = df['attack_cat'].fillna('Normal').astype(str).str.strip().replace(
            {'Backdoors': 'Backdoor', '': 'Normal'})
        cat_cols = ['proto', 'service', 'state']
        drop = {'id', 'label', 'attack_cat', '__file'}
    elif fmt == 'cic':
        df = _read_csvs(c['files'], max_rows)
        lab_col = [col for col in df.columns if col.lower() == 'label'][0]
        df = df[df[lab_col].astype(str).str.strip().str.lower() != 'label']       # repeated header rows (2018)
        df = df.reset_index(drop=True)
        labels = df[lab_col].map(cic_label)
        cat_cols = []
        drop = {'__file', lab_col}
    elif fmt == 'generic':
        df = _read_csvs(c['files'])
        labels = df[c['label_col']].astype(str).str.strip()
        cat_cols = c.get('cat_cols', [])
        drop = set(c.get('drop_cols', [])) | {c['label_col'], '__file'}
    else:
        raise ValueError(fmt)

    # ordering (time if available, otherwise file order then row order)
    time_col = c.get('time_col')
    if time_col and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors='coerce', dayfirst=c.get('dayfirst', True))
        order_key = ts.values.astype('datetime64[ns]').astype(np.int64)
    else:
        order_key = np.arange(len(df), dtype=np.int64)
        if c.get('file_order'):
            # chronological rank of each file (e.g. Monday..Friday), row order within a file
            fo = [p.lower() for p in c['file_order']]
            rank = {f: next((i for i, p in enumerate(fo) if p in f.lower()), len(fo))
                    for f in df['__file'].unique()}
            order_key = df['__file'].map(rank).values.astype(np.int64) * 10**10 + order_key

    groups = None
    gcols = c.get('group_cols') or []
    if gcols and all(g in df.columns for g in gcols):
        a = df[gcols[0]].astype(str)
        b = df[gcols[1]].astype(str)
        pair = np.where(a < b, a + '|' + b, b + '|' + a)            # undirected host pair
        groups = pd.factorize(pair)[0].astype(np.int64)

    feat_cols = [col for col in df.columns if col not in drop and col not in cat_cols
                 and canonical(col) not in ID_COLS and col != time_col and col not in gcols]
    num = df[feat_cols].apply(pd.to_numeric, errors='coerce').astype(np.float64)
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).values
    cat = df[cat_cols].astype(str).values if cat_cols else np.zeros((len(df), 0), dtype=object)

    if binary:
        labels = labels.map(lambda s: 'Benign' if s in BENIGN_NAMES else 'Attack')
    if classes is None:
        classes = sorted(labels.unique(), key=lambda s: (s not in BENIGN_NAMES, s))
    idx = {k: i for i, k in enumerate(classes)}
    y = labels.map(lambda s: idx.get(s, -1)).values.astype(np.int64)
    keep = y >= 0
    if max_rows and keep.sum() > max_rows:          # stratified subsample for limited RAM
        rng = np.random.default_rng(0)
        ids = np.where(keep)[0]
        frac = max_rows / len(ids)
        sel = []
        for c_ in np.unique(y[ids]):
            ic = ids[y[ids] == c_]
            k = min(len(ic), max(int(round(len(ic) * frac)), min(len(ic), 200)))
            sel.append(rng.choice(ic, k, replace=False))
        keep = np.zeros(len(y), dtype=bool); keep[np.concatenate(sel)] = True
    files = df['__file'].values
    t = Table(num[keep], cat[keep], y[keep], list(classes), files[keep], order_key[keep],
              None if groups is None else groups[keep], feat_cols, cat_cols)
    return t


def align_features(src, tgt):
    """Keep only numeric features that both tables share (by canonical name)."""
    cs = {canonical(n): i for i, n in enumerate(src.num_names)}
    ct = {canonical(n): i for i, n in enumerate(tgt.num_names)}
    common = [k for k in cs if k in ct]
    if len(common) < 0.5 * min(len(cs), len(ct)):
        raise ValueError(f'Only {len(common)} shared features; the datasets use different schemas. '
                         'Use NetFlow-v2 versions or a common extractor for cross-dataset tests.')
    only_s = sorted(set(cs) - set(common))
    only_t = sorted(set(ct) - set(common))
    for tbl, m in ((src, cs), (tgt, ct)):
        cols = [m[k] for k in common]
        tbl.num = tbl.num[:, cols]
        tbl.num_names = [tbl.num_names[i] for i in cols]
    return common, only_s, only_t


# --------------------------------------------------------------------------
# Preprocessing fitted on training indices
# --------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, clip=10.0):
        self.clip = clip

    def fit(self, t, idx):
        x = t.num[idx]
        self.med = np.median(x, axis=0)
        q1, q3 = np.percentile(x, [25, 75], axis=0)
        iqr = q3 - q1
        std = x.std(axis=0)
        self.scale = np.where(iqr > 1e-9, iqr, np.where(std > 1e-9, std, 1.0)).astype(np.float32)
        self.vocab = []
        for j in range(t.cat.shape[1]):
            vals = pd.unique(t.cat[idx, j])
            self.vocab.append({v: i + 1 for i, v in enumerate(vals)})   # 0 = unknown
        return self

    def transform(self, t):
        xn = (t.num - self.med) / self.scale
        xn = np.clip(xn, -self.clip, self.clip).astype(np.float32)
        xc = np.zeros((len(t), t.cat.shape[1]), dtype=np.int64)
        for j, voc in enumerate(self.vocab):
            xc[:, j] = pd.Series(t.cat[:, j]).map(voc).fillna(0).astype(np.int64).values
        return xn, xc

    @property
    def cardinalities(self):
        return [len(v) + 1 for v in self.vocab]


# --------------------------------------------------------------------------
# Sequences: row i -> indices of the previous T-1 flows of the same host pair
# (or the previous T-1 flows overall when no host information exists) + i.
# Index N (one past the end) points to an all-zero padding row.
# --------------------------------------------------------------------------
def build_sequences(t, T):
    n = len(t)
    order = np.argsort(t.order_key, kind='stable')
    g = t.groups[order] if t.groups is not None else np.zeros(n, dtype=np.int64)
    df = pd.DataFrame({'g': g, 'i': order})
    seq = np.full((n, T), n, dtype=np.int64)
    for k in range(T):
        prev = df.groupby('g')['i'].shift(k)
        seq[order, T - 1 - k] = prev.fillna(n).astype(np.int64).values
    return seq


def chunk_groups(t, chunk=5000):
    """Contiguous blocks in time order, used as CV groups to limit leakage between
    overlapping sequences in train and test."""
    order = np.argsort(t.order_key, kind='stable')
    grp = np.empty(len(t), dtype=np.int64)
    grp[order] = np.arange(len(t)) // chunk
    return grp


def make_folds(t, n_splits=5, seed=0, chunk=5000):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(t))
    chunk = max(1, min(chunk, len(t) // (n_splits * 20)))   # >= 20 groups per fold
    return [(tr, te) for tr, te in sgkf.split(idx, t.y, chunk_groups(t, chunk))]


def purge_boundaries(train_ids, other_ids, seq):
    """Removes training sequences whose context (the preceding T-1 records) contains records of `other_ids`
    (e.g. the test fold). Sequences are built before splitting, so at a block boundary where a training block
    follows a test block, up to T-1 training sequences would otherwise contain test-record features as context."""
    n = seq.shape[0]
    mark = np.zeros(n + 1, dtype=bool)          # index n is the zero-padding row
    mark[other_ids] = True
    keep = ~mark[seq[train_ids]].any(axis=1)
    return train_ids[keep], int((~keep).sum())


def val_split(y, idx, frac=0.1, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    ys = y[idx]
    # classes with a single sample cannot be stratified; keep them in train
    counts = np.bincount(ys)
    ok = counts[ys] >= 2
    a, b = next(sss.split(idx[ok], ys[ok]))
    return np.concatenate([idx[ok][a], idx[~ok]]), idx[ok][b]


def val_split_blocks(t, idx, frac=0.1, seed=0, chunk=5000):
    """Validation split made of contiguous blocks of the training fold (instead of random sequences), so that
    validation sequences overlap training sequences only at block boundaries (combine with purge_boundaries
    to remove that overlap as well)."""
    n_splits = max(2, int(round(1 / frac)))
    chunk = max(1, min(chunk, len(idx) // (n_splits * 20)))   # >= 20 blocks per split, as in make_folds
    grp = chunk_groups(t, chunk)[idx]
    a, b = next(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, t.y[idx], grp))
    return idx[a], idx[b]


def subsample(idx, y, frac, seed):
    """Stratified random subset (e.g. the 10% of labelled target data)."""
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[idx]):
        ids = idx[y[idx] == c]
        k = max(1, int(round(len(ids) * frac)))
        out.append(rng.choice(ids, k, replace=False))
    return np.concatenate(out)


# --------------------------------------------------------------------------
# SMOTE on flattened sequences (training data only)
# --------------------------------------------------------------------------
def smote_sequences(xn, xc, seq, y, train_idx, min_count, target, k=5, seed=0):
    """Returns synthetic (num_seq (M,T,dn), cat_seq (M,T,dc), y (M,)) for classes whose
    training count is below `min_count`, oversampled up to `target` samples."""
    rng = np.random.default_rng(seed)
    pad_n = np.vstack([xn, np.zeros((1, xn.shape[1]), np.float32)])
    pad_c = np.vstack([xc, np.zeros((1, xc.shape[1]), np.int64)])
    outs_n, outs_c, outs_y = [], [], []
    for c in np.unique(y[train_idx]):
        ids = train_idx[y[train_idx] == c]
        if len(ids) >= min_count or len(ids) < 2:
            continue
        need = target - len(ids)
        S = pad_n[seq[ids]]                       # (n_c, T, dn)
        flat = S.reshape(len(ids), -1)
        kk = min(k, len(ids) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(flat)
        _, nb = nn.kneighbors(flat)
        base = rng.integers(0, len(ids), need)
        mate = nb[base, rng.integers(1, kk + 1, need)]
        lam = rng.random((need, 1)).astype(np.float32)
        new = flat[base] + lam * (flat[mate] - flat[base])
        outs_n.append(new.reshape(need, *S.shape[1:]).astype(np.float32))
        outs_c.append(pad_c[seq[ids[base]]])
        outs_y.append(np.full(need, c, dtype=np.int64))
    if not outs_y:
        return None
    return np.concatenate(outs_n), np.concatenate(outs_c), np.concatenate(outs_y)


def phase_indices(t, patterns):
    """Rows whose source file name contains any of the given substrings (e.g. weekdays)."""
    m = np.zeros(len(t), dtype=bool)
    for p in patterns:
        m |= np.array([p.lower() in f.lower() for f in t.files])
    return np.where(m)[0]


def packets_per_flow(t):
    names = [canonical(n) for n in t.num_names]
    fwd = canonical('Total Fwd Packets')
    bwd = canonical('Total Backward Packets')
    if fwd in names and bwd in names:
        return float((t.num[:, names.index(fwd)] + t.num[:, names.index(bwd)]).mean())
    return None
