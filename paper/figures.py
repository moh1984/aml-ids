"""Draws all figures of the paper at their printed size (text width 6.1 in), so font sizes are true point sizes.
Run from the repository root after paper/summarize.py:  python paper/figures.py
Each figure is placed in the paper at the size it is drawn (6.0 in wide; confusion matrices 3.9 and 3.3 in)."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})
OUT = 'paper/figures/'
S = json.load(open('paper/numbers_static.json'))
A = json.load(open('paper/numbers_adaptation.json'))

# ---------------- Figure 1: architecture ----------------
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 51); ax.axis('off')

def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.0', fc=fc, ec='#333', lw=0.8))
    ax.text(x + w / 2, y + h - 1.4, title, ha='center', va='top', fontsize=8, weight='bold')
    ax.text(x + w / 2, y + (h - 4.2) / 2, sub, ha='center', va='center', fontsize=7, linespacing=1.3)

def arr(x1, y1, x2, y2, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#333', ls=ls))

ax.text(0.5, 49.6, 'Detection path', ha='left', fontsize=8.5, style='italic', color='#444')
box(0.5, 24, 17, 22, 'Flow records', 'CICFlowMeter\nfeatures (78 for\nCIC datasets)', '#e8eef7')
box(20.5, 24, 18.5, 22, 'Preprocessing', 'IQR scaling,\nclip ±10;\nembeddings (d=8);\nT = 10 sequences;\nSMOTE (train)', '#e8eef7')
box(42, 36, 21, 10, 'Multi-scale CNN', '9×9 map; 3×3, 5×5,\n7×7 × 64; GMP → 192', '#fdf1d8')
box(42, 24, 21, 10, 'BiLSTM + attention', '2 × 256 BiLSTM,\n4 heads → 512', '#fdf1d8')
box(66, 24, 17, 22, 'Fusion & head', '704 → 512\n→ 256 → C\nsoftmax', '#fdf1d8')
box(86, 24, 13.5, 22, 'Output', 'class label\nand alert', '#e6f4ea')
arr(17.8, 35, 20.2, 35); arr(39.3, 38, 41.7, 41); arr(39.3, 32, 41.7, 29)
arr(63.3, 41, 65.7, 38); arr(63.3, 29, 65.7, 32); arr(83.3, 35, 85.7, 35)
ax.plot([0.5, 99.5], [20.8, 20.8], color='#bbb', lw=0.7, ls='--')
ax.text(0.5, 18.3, 'Drift-aware adaptation loop (after deployment)', ha='left', fontsize=8.5, style='italic', color='#444')
box(0.5, 0.8, 22, 14.5, 'Labelled feedback', '10% of each batch\nof 2,000 flows', '#f3e8f7')
box(25.5, 0.8, 23.5, 14.5, 'Page-Hinkley test', 'error e_t (%), δ = 0.5,\nλ_PH = 50; recovery:\nadapt every 5 batches\nwhile e_t > e_ref + 5', '#f3e8f7')
box(52, 0.8, 22, 14.5, 'Fine-tuning', 'last LSTM layer +\nhead; Adam 1e-4,\n5 epochs; 70% new /\n30% replay', '#f3e8f7')
box(77, 0.8, 22.5, 14.5, 'Balanced replay', 'reservoir sampling,\nequal per-class\nquotas; 10,000\nflows', '#f3e8f7')
arr(22.8, 8, 25.2, 8); arr(49.3, 8, 51.7, 8); arr(76.7, 8, 74.3, 8)
arr(66, 15.6, 60, 23.6, ls='--'); ax.text(64.5, 18.3, 'update', fontsize=7, color='#444')
fig.tight_layout(pad=0.2); fig.savefig(OUT + 'fig1_architecture.png', dpi=300); plt.close(fig)

# ---------------- Figure 2: ROC curves (needs the per-sample predictions, not included in the repository) ----------------
from sklearn.metrics import roc_curve, auc
P = sorted(glob.glob('results_preds/cicids2017_aml_ids/preds*.npz'))
if P:
    D = [np.load(f) for f in P]
    y = np.concatenate([d['y'] for d in D]); prob = np.concatenate([d['prob'] for d in D])
    cls = json.load(open('results/indomain/cicids2017/aml_ids/fold0.json'))['classes']
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.6))
    colors = plt.cm.tab10(np.arange(len(cls)))
    for i, c in enumerate(cls):
        if not ((y == i).any() and (y != i).any()):
            continue
        fpr, tpr, _ = roc_curve(y == i, prob[:, i]); a = auc(fpr, tpr) * 100
        for k in (0, 1):
            ax[k].plot(fpr * 100, tpr * 100, lw=1.1, color=colors[i], label=f'{c} ({a:.1f})' if k == 0 else None)
    ax[0].plot([0, 100], [0, 100], 'k--', lw=0.7)
    ax[0].set(title='(a) Full range', xlabel='False positive rate (%)', ylabel='True positive rate (%)')
    ax[1].set(title='(b) Low false-positive region', xlabel='False positive rate (%)', xlim=(0, 2), ylim=(40, 101))
    for a_ in ax: a_.grid(alpha=0.3)
    ax[0].legend(loc='lower right', fontsize=6.3, title='Class (AUC, %)', title_fontsize=6.5, ncol=2,
                 columnspacing=0.6, handlelength=1.2, borderpad=0.3, labelspacing=0.25, framealpha=0.95)
    fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig2_roc_cicids2017.png', dpi=300); plt.close(fig)
else:
    print('results_preds/ not found: Figure 2 (ROC) skipped')

# ---------------- Figures 3 and 4: normalized confusion matrices ----------------
def confusion(ds, fname):
    R = [json.load(open(f)) for f in sorted(glob.glob(f'results/indomain/{ds}/aml_ids/fold*.json'))]
    cm = sum(np.array(r['confusion'], float) for r in R); cls = R[0]['classes']
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(cls)
    fig, ax = plt.subplots(figsize=(3.6, 3.1) if n > 6 else (3.3, 2.75))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cls, rotation=40, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(cls)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6.2 if n > 6 else 7.5,
                    color='white' if cmn[i, j] > 0.5 else 'black')
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
    cb = fig.colorbar(im, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.3); fig.savefig(OUT + fname, dpi=300); plt.close(fig)

confusion('cicids2017', 'fig3_confusion_cicids2017.png')
confusion('nslkdd', 'fig4_confusion_nslkdd.png')

# ---------------- Figure 5: convergence ----------------
C = [json.load(open(f)) for f in sorted(glob.glob('results/indomain/cicids2017/aml_ids/curves*.json'))]
L = min(len(c) for c in C); ep = np.arange(1, L + 1)
get = lambda k: np.mean([[e[k] for e in c[:L]] for c in C], axis=0)
best = int(np.mean([np.argmax([e['val_f1'] for e in c]) + 1 for c in C]))
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.5))
ax[0].plot(ep, get('train_loss'), label='Training'); ax[0].plot(ep, get('val_loss'), label='Validation')
ax[0].set(title='(a) Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
ax[1].plot(ep, get('train_acc'), label='Training'); ax[1].plot(ep, get('val_acc'), label='Validation')
ax[1].set(title='(b) Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
for a in ax:
    a.axvline(best, ls='--', c='gray', lw=0.9, label=f'Best epoch (mean {best})'); a.grid(alpha=0.3); a.legend()
fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig5_curves_cicids2017.png', dpi=300); plt.close(fig)

# ---------------- Figure 6: adaptation ----------------
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.9))
d = {}
for r in A['lambda']:
    d.setdefault((r['lam'], r['buffer']), r)
xs = sorted({k[0] for k in d}); pos = {l: i for i, l in enumerate(xs)}
for buf, lab, c in ((0, 'no replay', '#c0392b'), (10000, 'replay (10,000)', '#2471a3')):
    L_ = [l for l in xs if (l, buf) in d]
    ax[0].errorbar([pos[l] for l in L_], [d[(l, buf)]['forg_f1'][0] for l in L_],
                   yerr=[d[(l, buf)]['forg_f1'][1] for l in L_], marker='o', ms=3.5, capsize=2, lw=1.1, label=lab, color=c)
ticks = ['0'] + [f'{l:g}' if l < 1e5 else f'1e{int(np.log10(l))}' for l in xs[1:]]
ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels(ticks, rotation=60, fontsize=7)
ax[0].set(title='(a) Forgetting vs. λ_EWC', xlabel='λ_EWC', ylabel='Forgetting, macro-F1 (pp)')
ax[0].grid(alpha=0.3); ax[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.3))
conf = [('none:none', 'static'), ('ph:naive', 'fine-tune'), ('ph:ewconly', 'EWC only'), ('ph:replay', 'replay'), ('ph:ewc', 'replay+EWC')]
cls = ['Benign', 'BruteForce', 'DoS']; w = 0.16
for i, (c, lab) in enumerate(conf):
    ax[1].bar(np.arange(3) + (i - 2) * w, [A['stream'][c]['ret_recall'][k] for k in cls], w, label=lab)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(cls); ax[1].set_ylim(0, 105)
ax[1].set(title='(b) Retention after the stream', ylabel='Recall on phase A (%)')
ax[1].grid(axis='y', alpha=0.3)
ax[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=7.5, columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig6_adaptation.png', dpi=300); plt.close(fig)
print('figures written to', OUT)
