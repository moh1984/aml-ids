#!/usr/bin/env bash
# Runs every experiment behind the tables and figures of the paper, with the settings used for the reported results,
# on a machine with 2 GPUs (two jobs in parallel). After it finishes, paper/summarize.py and paper/figures.py rebuild
# all reported numbers and figures from results/.
#
# Reproducibility note: the paper's in-domain CICIDS2017 values are per-fold means of TWO runs made with the default
# (non-deterministic) cuDNN kernels. The current code uses deterministic kernels, so a new run reproduces the protocol
# but not those two runs bit for bit; expect differences within the run-to-run variation reported in Section 4.3.
# Set RERUN=1 to also make the second CICIDS2017 run (results/indomain_rerun, results/latency_rerun).
set -e
mkdir -p logs
FAST="--steps-per-epoch 1000 --max-epochs 60 --patience 10"   # training budget used in the paper (Table 3)
G() { CUDA_VISIBLE_DEVICES=$1 python run.py "${@:2}"; }

# ---------------------------------------------------------------------------------------------------------------
# UNSW-NB15 is evaluated on a copy in which each official partition is shuffled separately (its files cluster
# records by class; Section 4.3). The loader then concatenates the two partitions.
python - <<'PY'
import os, glob, pandas as pd, yaml
cfg = yaml.safe_load(open('configs/datasets.yaml'))
os.makedirs('data/UNSW-NB15-shuffled', exist_ok=True)
for pat in cfg['unsw_nb15']['files']:
    for f in glob.glob(pat):
        pd.read_csv(f).sample(frac=1.0, random_state=0).to_csv('data/UNSW-NB15-shuffled/' + os.path.basename(f), index=False)
PY

# ---------------------------------------------------------------------------------------------------------------
# Table 5 and Figures 2-5: in-domain benchmarks (5 folds / 5 seeds), all models, including the last row of Table 5
# (AML-IDS architecture without SMOTE = abl_attention)
indomain_all() {   # $1 = dataset, $2 = output root
  python run.py classical --dataset $1 --model rf  --out $2 > logs/$1_rf.log  2>&1 &
  python run.py classical --dataset $1 --model svm --out $2 > logs/$1_svm.log 2>&1 &
  G 0 indomain --dataset $1 --model aml_ids $FAST --out $2                  > logs/$1_aml.log 2>&1 &
  G 1 indomain --dataset $1 --model transformer --no-smote $FAST --out $2   > logs/$1_trf.log 2>&1; wait
  G 0 indomain --dataset $1 --model cnn_lstm_attn --no-smote $FAST --out $2 > logs/$1_cla.log 2>&1 &
  G 1 indomain --dataset $1 --model mamba --no-smote $FAST --out $2         > logs/$1_mba.log 2>&1; wait
  G 0 indomain --dataset $1 --model lstm --no-smote $FAST --out $2          > logs/$1_lstm.log 2>&1 &
  G 1 indomain --dataset $1 --model cnn --no-smote $FAST --out $2           > logs/$1_cnn.log 2>&1; wait
}
for DS in cicids2017 unsw_shuffled nslkdd; do
  indomain_all $DS results
  python run.py plots --dataset $DS
done
G 0 indomain --dataset unsw_shuffled --model abl_attention --no-smote $FAST > logs/unsw_nosmote.log 2>&1 &
G 1 indomain --dataset nslkdd --model abl_attention --no-smote $FAST        > logs/nsl_nosmote.log 2>&1; wait
G 0 latency  --dataset cicids2017 > logs/latency.log 2>&1      # Table 11
G 0 branches --dataset cicids2017 > logs/branches.log 2>&1     # branch-removal analysis (Section 5.2)

# Optional second CICIDS2017 run (the paper averages two runs per fold)
if [ "${RERUN:-0}" = "1" ]; then
  indomain_all cicids2017 results_rerun
  G 0 latency --dataset cicids2017 --out results_rerun > logs/latency_rerun.log 2>&1
  mkdir -p results/indomain_rerun results/latency_rerun
  cp -r results_rerun/indomain/cicids2017 results/indomain_rerun/ && cp results_rerun/latency/*.json results/latency_rerun/
fi

# ---------------------------------------------------------------------------------------------------------------
# Section 4.3: record-order check on UNSW-NB15 in its original file order
python run.py classical --dataset unsw_nb15 --model rf > logs/order_rf.log 2>&1 &
G 0 indomain --dataset unsw_nb15 --model abl_attention --no-smote $FAST > logs/order_aml_nosmote.log 2>&1 &
G 1 indomain --dataset unsw_nb15 --model lstm --no-smote $FAST          > logs/order_lstm.log 2>&1; wait

# Table 6: ablation on CICIDS2017 (the last row is aml_ids above; abl_attention is also the CICIDS2017 no-SMOTE row)
G 0 indomain --dataset cicids2017 --model abl_base --no-smote $FAST       > logs/abl1.log 2>&1 &
G 1 indomain --dataset cicids2017 --model abl_multiscale --no-smote $FAST > logs/abl2.log 2>&1; wait
G 0 indomain --dataset cicids2017 --model abl_bilstm --no-smote $FAST     > logs/abl3.log 2>&1 &
G 1 indomain --dataset cicids2017 --model abl_attention --no-smote $FAST  > logs/abl4.log 2>&1; wait

# ---------------------------------------------------------------------------------------------------------------
# Tables 7-8 and Figure 6(a): forgetting, EWC coefficient, replay-buffer size and balancing
G 0 phases --dataset cicids2017 --tag lambda --folds 3 --phases "Monday,Tuesday,Wednesday|Thursday,Friday" \
    --lams 0,100,1000,5000,10000,50000 --buffer-sizes 0,10000 $FAST > logs/lambda.log 2>&1 &
G 1 phases --dataset cicids2017 --tag buffer --folds 3 --phases "Monday,Tuesday|Wednesday|Friday" \
    --lams 5000 --buffer-sizes 2000,5000,10000,20000,50000 --try-balanced $FAST > logs/buffer.log 2>&1; wait
# lambda=0 without buffer is already part of the 'lambda' run above, so it is not repeated here
G 0 phases --dataset cicids2017 --tag lambda_large --folds 3 --phases "Monday,Tuesday,Wednesday|Thursday,Friday" \
    --lams 100000,1000000,10000000,100000000,1000000000,10000000000 --buffer-sizes 0 $FAST > logs/lambda_large.log 2>&1 &
# Table 9 and Figure 6(b): prequential stream
G 1 stream --dataset cicids2017 --folds 3 --balanced \
    --configs none:none,ph:naive,ph:ewconly,ph:replay,ph:ewc,adwin:ewc $FAST > logs/stream.log 2>&1; wait

# Table 10: transfer CICIDS2017 -> CSE-CIC-IDS2018 (binary, 5 seeds, 4 adaptation strategies)
G 0 cross --source cicids2017 --target cicids2018 --binary --max-rows 3000000 --balanced $FAST > logs/cross.log 2>&1

# ---------------------------------------------------------------------------------------------------------------
python run.py tables
python paper/summarize.py && python paper/figures.py
