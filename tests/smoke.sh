#!/usr/bin/env bash
# Quick end-to-end check on synthetic data (CPU, ~5 minutes). Run from the project root.
set -e
python tests/make_synthetic.py
Q="--max-epochs 2 --patience 2 --folds 2 --smote-min 100 --smote-target 100 --fisher-n 30 --adapt-epochs 1 --out results_smoke"
python run.py indomain  --dataset cicids2017 --model aml_ids $Q
python run.py indomain  --dataset nslkdd --model aml_ids $Q
python run.py indomain  --dataset unsw_nb15 --model transformer --no-smote $Q
python run.py classical --dataset unsw_nb15 --model rf --folds 2 --out results_smoke
python run.py cross     --source cicids2017 --target cicids2018 $Q
python run.py phases    --dataset cicids2017 --lams 0,5000 --buffer-sizes 500 $Q --folds 1
python run.py stream    --dataset cicids2017 --stream-batch 100 $Q --folds 1
python run.py latency   --dataset cicids2017 --reps 20 --models aml_ids --out results_smoke
python run.py plots     --dataset cicids2017 --out results_smoke
python run.py tables    --out results_smoke > /dev/null
echo "SMOKE TEST PASSED (numbers on synthetic data are meaningless)"
rm -rf data results_smoke
