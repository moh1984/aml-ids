"""AML-IDS and baseline models.

Input to every model: x_num (B, T, dn) float, x_cat (B, T, dc) long.
The spatial (CNN) path uses the last flow of the sequence; the temporal path uses
the whole sequence.  Shapes are documented in Table 2 of the manuscript.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEncoder(nn.Module):
    def __init__(self, dn, cards, emb_dim=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cards])
        self.out_dim = dn + emb_dim * len(cards)

    def forward(self, xn, xc):
        if len(self.embs) == 0:
            return xn
        e = [emb(xc[..., j]) for j, emb in enumerate(self.embs)]
        return torch.cat([xn] + e, dim=-1)


class AMLIDS(nn.Module):
    """Multi-scale CNN + (Bi)LSTM + multi-head attention.

    Flags reproduce the ablation rows of Table 7:
      scales=(3,), bidirectional=False, attention=False  -> base single-scale CNN-LSTM
      scales=(3,5,7)                                        -> + multi-scale filters
      bidirectional=True                                    -> + BiLSTM
      attention=True                                        -> + multi-head attention
    spatial=False gives an LSTM-only model; temporal=False a CNN-only model.
    """

    def __init__(self, dn, cards, n_classes, T, scales=(3, 5, 7), filters=64, hidden=256,
                 layers=2, bidirectional=True, attention=True, heads=4, fc=(512, 256),
                 dropout=0.3, spatial=True, temporal=True, emb_dim=8):
        super().__init__()
        self.enc = InputEncoder(dn, cards, emb_dim)
        d = self.enc.out_dim
        self.side = math.ceil(math.sqrt(d))
        self.pad = self.side ** 2 - d
        self.spatial, self.temporal = spatial, temporal
        fused = 0
        if spatial:
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Conv2d(1, filters, k, stride=1, padding=k // 2),
                              nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
                              nn.AdaptiveMaxPool2d(1))
                for k in scales])
            self.register_buffer('branch_mask', torch.ones(len(scales)))
            fused += filters * len(scales)
        if temporal:
            self.lstm = nn.LSTM(d, hidden, num_layers=layers, batch_first=True,
                                bidirectional=bidirectional)
            h = hidden * (2 if bidirectional else 1)
            self.attn = nn.MultiheadAttention(h, heads, batch_first=True) if attention else None
            fused += h
        dims = [fused] + list(fc)
        head = []
        for a, b in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        head.append(nn.Linear(dims[-1], n_classes))
        self.head = nn.Sequential(*head)
        self.layers = layers

    def spatial_features(self, x_last):
        B = x_last.shape[0]
        z = F.pad(x_last, (0, self.pad)).view(B, 1, self.side, self.side)
        outs = [br(z).flatten(1) * self.branch_mask[i] for i, br in enumerate(self.branches)]
        return torch.cat(outs, dim=1)                                    # (B, 64*len(scales))

    def forward(self, xn, xc):
        x = self.enc(xn, xc)                                              # (B, T, d)
        parts = []
        if self.spatial:
            parts.append(self.spatial_features(x[:, -1]))
        if self.temporal:
            H, _ = self.lstm(x)                                           # (B, T, h)
            if self.attn is not None:
                A, _ = self.attn(H, H, H, need_weights=False)
                parts.append(A.mean(dim=1))                               # (B, h)
            else:
                parts.append(H[:, -1])
        return self.head(torch.cat(parts, dim=1))

    # parameters updated during post-drift adaptation (final LSTM layer + head)
    def adapt_parameters(self):
        ps = list(self.head.parameters())
        if self.temporal:
            last = f'_l{self.layers - 1}'
            ps += [p for n, p in self.lstm.named_parameters() if n.endswith(last) or
                   n.endswith(last + '_reverse')]
        return ps


class TransformerIDS(nn.Module):
    def __init__(self, dn, cards, n_classes, T, d_model=128, heads=4, layers=4, ff=256,
                 dropout=0.1, emb_dim=8):
        super().__init__()
        self.enc = InputEncoder(dn, cards, emb_dim)
        self.proj = nn.Linear(self.enc.out_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, T, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, xn, xc):
        z = self.proj(self.enc(xn, xc)) + self.pos[:, :xn.shape[1]]
        return self.head(self.tr(z).mean(dim=1))

    def adapt_parameters(self):
        return list(self.head.parameters())


class SelectiveSSM(nn.Module):
    """Minimal pure-PyTorch Mamba (S6) block. Uses mamba_ssm.Mamba if installed."""

    def __init__(self, d, state=16, expand=2, conv=4):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # noqa
            self.fast = Mamba(d_model=d, d_state=state, d_conv=conv, expand=expand)
            return
        except Exception:
            self.fast = None
        E = d * expand
        self.dt_rank = math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * E)
        self.conv = nn.Conv1d(E, E, conv, groups=E, padding=conv - 1)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1).float()).repeat(E, 1))
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d)
        self.state = state

    def forward(self, x):
        if self.fast is not None:
            return self.fast(x)
        B, T, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dbc = self.x_proj(u)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.state, self.state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                  # (B, T, E)
        A = -torch.exp(self.A_log)                                         # (E, N)
        h = torch.zeros(B, u.shape[-1], self.state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t, :, None] * A)
            h = dA * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * u[:, t])
        y = torch.stack(ys, dim=1) * F.silu(z)
        return self.out_proj(y)


class MambaIDS(nn.Module):
    def __init__(self, dn, cards, n_classes, T, d_model=128, layers=2, emb_dim=8):
        super().__init__()
        self.enc = InputEncoder(dn, cards, emb_dim)
        self.proj = nn.Linear(self.enc.out_dim, d_model)
        self.blocks = nn.ModuleList([SelectiveSSM(d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, xn, xc):
        z = self.proj(self.enc(xn, xc))
        for blk, norm in zip(self.blocks, self.norms):
            z = z + blk(norm(z))
        return self.head(z.mean(dim=1))

    def adapt_parameters(self):
        return list(self.head.parameters())


# name -> constructor kwargs; used by the experiment scripts and Table 4/5/7
MODELS = {
    'aml_ids':        dict(cls='AMLIDS'),
    # ablation (Table 7)
    'abl_base':       dict(cls='AMLIDS', scales=(3,), bidirectional=False, attention=False),
    'abl_multiscale': dict(cls='AMLIDS', bidirectional=False, attention=False),
    'abl_bilstm':     dict(cls='AMLIDS', attention=False),
    'abl_attention':  dict(cls='AMLIDS'),          # + SMOTE is switched in the trainer
    # deep baselines (Tables 4-5)
    'cnn':            dict(cls='AMLIDS', scales=(3,), temporal=False),
    'lstm':           dict(cls='AMLIDS', spatial=False, bidirectional=False, attention=False,
                           layers=1),
    'cnn_lstm_attn':  dict(cls='AMLIDS', scales=(3,), bidirectional=False, attention=True,
                           layers=1),
    'transformer':    dict(cls='TransformerIDS'),
    'mamba':          dict(cls='MambaIDS'),
}


def build_model(name, dn, cards, n_classes, T):
    spec = dict(MODELS[name])
    cls = {'AMLIDS': AMLIDS, 'TransformerIDS': TransformerIDS, 'MambaIDS': MambaIDS}[spec.pop('cls')]
    return cls(dn, cards, n_classes, T, **spec)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
