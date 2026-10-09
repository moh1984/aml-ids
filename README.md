# AML-IDS: Drift-Aware Deep Intrusion Detection with Class-Balanced Replay

Code, configuration, and result logs for the paper

> M. Alkhazaleh, M. Baklizi, M. Zghoul, and M. N. AlShaikh Hasan, "Drift-Aware Deep Intrusion Detection with
> Class-Balanced Replay for Evolving Network Traffic," *IAES International Journal of Artificial Intelligence (IJ-AI)*,
> under review (paper ID 32546).

AML-IDS couples a multi-scale CNN-BiLSTM flow classifier with a post-deployment adaptation loop:

- a **Page-Hinkley test with a recovery mode** decides when to update the model and keeps updating while the error
  stays above its in-distribution level;
- a **class-balanced replay buffer** (reservoir sampling with equal per-class quotas) protects previously learned attacks;
- only the **last BiLSTM layer and the classifier** are fine-tuned.

Elastic weight consolidation (EWC) is implemented as an evaluated alternative; in our experiments it had no measurable
effect on forgetting for λ from 10² to 10¹⁰, whereas class balancing of the buffer reduced forgetting from 3.8 to 0.6
macro-F1 points.

## Main results (from `results/`)

| Experiment | Result |
|---|---|
| CICIDS2017, macro-F1 (5 folds) | AML-IDS 80.8 ± 3.7 (75.8 without SMOTE); LSTM 80.1; Transformer 73.7; random forest 91.5 |
| UNSW-NB15, macro-F1 (5 folds, shuffled order) | AML-IDS 56.2 ± 1.0; Transformer 59.4; random forest 58.1 |
| NSL-KDD test set, macro-F1 (5 seeds) | AML-IDS 55.4 ± 1.0; random forest 46.8 |
| Transfer CICIDS2017 → CSE-CIC-IDS2018, 10% labels | target macro-F1 59.1 → 98.0; source retained 99.5 (naive fine-tuning 94.1) |
| Prequential stream, Thursday–Friday of CICIDS2017 | accuracy 74.8% → 95.5%; earlier DoS recall 93.6% (naive fine-tuning 12.4%) |
| Replay-buffer balancing (three-phase experiment) | forgetting 3.8 → 0.6 macro-F1 points at every buffer size |
| Throughput on one NVIDIA T4 (batch 1,024) | ≈24,000 flows/s; amortized 2.1 µs per packet (per-flow time / mean packets per flow) |

## Repository layout

```
amlids/            data pipeline, models, training, drift detection, replay, EWC
configs/           dataset paths and file order
run.py             command-line entry point for every experiment
run_all.sh         runs every experiment behind the paper's tables and figures (2-GPU machine)
notebooks/         Kaggle notebook used to run the experiments (7 stages)
results/           JSON results of the runs reported in the paper (no model weights)
paper/             summarize.py and figures.py regenerate the paper's numbers and Figures 1–6
                   (Figure 2 requires the release asset results_preds.zip)
tests/             synthetic data generator and smoke test
requirements-lock.txt  exact package versions of the Kaggle GPU environment used for the paper
```

## Installation

```bash
pip install -r requirements.txt     # PyTorch with CUDA recommended
# exact versions used for the paper (Kaggle GPU image, Python 3.12.13, CUDA 12.8, cuDNN 9.10.2):
# pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
bash tests/smoke.sh                 # ~5 min on CPU, synthetic data; must print SMOKE TEST PASSED
```

## Data

All datasets are public. Place the files as follows (paths can be changed in `configs/datasets.yaml`):

| Dataset | Files | Folder |
|---|---|---|
| CICIDS2017 | the eight daily CSV files (MachineLearningCVE or TrafficLabelling) | `data/CICIDS2017/` |
| CSE-CIC-IDS2018 | the ten daily "Processed Traffic Data for ML" CSV files | `data/CSE-CIC-IDS2018/` |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | `data/UNSW-NB15/` |
| NSL-KDD | `KDDTrain+.txt`, `KDDTest+.txt` | `data/NSL-KDD/` |

Use the versions that keep the original per-day files: the drift experiments are defined by weekday.

## Reproducing the paper

`run_all.sh` runs every experiment behind the tables and figures, with the settings of the paper, and then rebuilds
the numbers and figures. Individual commands:

| Paper item | Command |
|---|---|
| Table 5, Figures 2–5 | `python run.py indomain --dataset <cicids2017\|unsw_shuffled\|nslkdd> --model <aml_ids\|cnn\|lstm\|cnn_lstm_attn\|transformer\|mamba> [--no-smote]` and `python run.py classical --dataset <ds> --model <rf\|svm>` |
| Table 5, last row (AML-IDS without SMOTE) | `indomain --model abl_attention --no-smote` on each dataset |
| Section 4.3, UNSW-NB15 order check | `rf`, `abl_attention --no-smote`, and `lstm --no-smote` on `unsw_nb15` (file order) |
| Table 6 (ablation) | `indomain --dataset cicids2017 --model abl_base / abl_multiscale / abl_bilstm / abl_attention --no-smote` |
| Table 7 (λ_EWC) | `phases --tag lambda --lams 0,100,1000,5000,10000,50000 --buffer-sizes 0,10000` and `phases --tag lambda_large --lams 100000,...,10000000000 --buffer-sizes 0` |
| Table 8 (buffer) | `phases --tag buffer --phases "Monday,Tuesday\|Wednesday\|Friday" --buffer-sizes 2000,...,50000 --try-balanced` |
| Table 9 (stream) | `stream --balanced --configs none:none,ph:naive,ph:ewconly,ph:replay,ph:ewc,adwin:ewc` |
| Table 10 (transfer) | `cross --source cicids2017 --target cicids2018 --binary --max-rows 3000000 --balanced` |
| Table 11 (cost) | `latency --dataset cicids2017` |
| Section 5.2 branch analysis | `branches --dataset cicids2017` |

All deep-learning runs in the paper used `--steps-per-epoch 1000 --max-epochs 60 --patience 10`. `unsw_shuffled` is
UNSW-NB15 with each official partition shuffled separately (`random_state=0`); `run_all.sh` creates its files.

**What a fresh run reproduces.** The paper's in-domain CICIDS2017 values are per-fold means of two runs made with the
default, non-deterministic cuDNN kernels. The current code uses deterministic kernels, so a fresh run follows the same
protocol but will not match those two runs bit for bit; differences are within the run-to-run variation reported in
Section 4.3 of the paper. `RERUN=1 bash run_all.sh` also makes the second run. The exact reported numbers are
regenerated from the stored results by `paper/summarize.py` (next section).

**On Kaggle:** upload the repository as a private dataset, import `notebooks/kaggle_aml_ids.ipynb`, attach the four
datasets, select GPU T4 ×2, set `STAGE` (1–7) in the first cell, and run with *Save & Run All (Commit)*. The notebook is
the one used for the paper; each stage stops itself after 11 h and saves all completed results.

## Regenerating the reported numbers

```bash
python paper/summarize.py    # -> paper/numbers_static.json, numbers_adaptation.json, numbers_fair_comparison.json
python paper/figures.py      # -> paper/figures/fig1 ... fig6 (fig2 only if results_preds/ is present)
```

Macro-averaged metrics are recomputed from the saved confusion matrices over the classes present in each test fold.
CICIDS2017 values are per-fold means of two runs (`results/indomain` and `results/indomain_rerun`). Figures 1 and 3–6 are rebuilt from `results/` alone. Figure 2 (ROC) needs the per-sample predictions of AML-IDS on
CICIDS2017 (88 MB), which are not stored in the repository because of their size; they are attached to the GitHub
release as `results_preds.zip`. Unzip it in the repository root (it creates `results_preds/cicids2017_aml_ids/`)
and run `python paper/figures.py`; without it, Figure 2 is skipped and the other figures are still produced.

## Levels of reproducibility

| What | How | Exact? |
|---|---|---|
| Every number in the paper | `python paper/summarize.py` on the stored `results/` | yes |
| Figures 1, 3–6 | `python paper/figures.py` on the stored `results/` | yes |
| Figure 2 (ROC) | same, after unzipping the release asset `results_preds.zip` | yes |
| Re-running the experiments | `bash run_all.sh` (add `RERUN=1` for the second CICIDS2017 run) | protocol-level |

Re-running reproduces the protocol, not necessarily the stored numbers bit for bit. The in-domain CICIDS2017 runs
(Table 5), the UNSW-NB15 file-order runs, and the NSL-KDD baseline runs used the default, non-deterministic cuDNN
kernels. All later runs (the adaptation experiments of Tables 7–10, the ablation of Table 6, the no-SMOTE rows, and the
shuffled UNSW-NB15 runs) used the deterministic kernels of the current code; on the same hardware and package versions
(`requirements-lock.txt`) they should repeat exactly, as two independent runs of the λ = 0 setting did.

## Notes

- **SMOTE.** AML-IDS is trained with SMOTE (`indomain --model aml_ids`), the deep baselines with `--no-smote`.
  The AML-IDS architecture without SMOTE is `abl_attention`; `paper/summarize.py` writes the resulting fair comparison
  to `paper/numbers_fair_comparison.json`.

- **Validation split.** Early stopping uses a random, stratified 10% of the training sequences (as in the paper).
  Because sequences overlap, validation scores are slightly optimistic; this affects model selection only, not the
  test folds. `--block-val` draws the validation split from contiguous blocks instead, and together with
  `--purge-boundaries` also removes training sequences whose context contains validation records.
- **Sequence overlap at fold boundaries.** Sequences are built before splitting. With the contiguous blocks used
  in the paper, overlap is confined to block boundaries: where a training block follows a test block, the first nine
  training sequences contain test-record features (never labels) as context (at most 0.18% of training sequences on
  CICIDS2017 and 0.35% on UNSW-NB15). `--purge-boundaries` removes such sequences for strictly separated folds; it is
  off by default so that the scripts reproduce the paper's numbers. If sequences are built per host pair (files with
  IP addresses), overlap is no longer confined to boundaries and `--purge-boundaries` should be used.
- **Determinism.** The in-domain runs in the paper used the default (non-deterministic) cuDNN kernels, so models with
  convolutional layers varied slightly between runs (AML-IDS macro-F1 81.8 and 79.8 on CICIDS2017). The current code
  enables deterministic kernels.
- **Sequences.** The public CICIDS2017 files contain no IP addresses, so sequences use the nine preceding flows in
  chronological order rather than per host pair. UNSW-NB15 and NSL-KDD have no timestamps, so file order is used.
  NSL-KDD records are highly interleaved and non-temporal in file order; UNSW-NB15 records cluster by class, which raises the macro-F1 of
  sequence models by 1–4 points and hides many of their false alarms. The UNSW-NB15 results reported in the paper are
  therefore those of `results/indomain/unsw_shuffled/` (the records of each official partition shuffled separately with `random_state=0`, then concatenated); the file-order
  runs in `results/indomain/unsw_nb15/` are kept for the order check.
- **Not included.** The IDS-MTran and GNN baselines of the original submission were not re-implemented, and the
  multi-class transfer to CSE-CIC-IDS2018 is not reported.

## License

MIT (see `LICENSE`).
