#!/usr/bin/env bash
set -euo pipefail
ASSETS=(CSI300 SP500 CSI1000 BTCUSD)
MODELS=(LSTM Transformer Autoformer PatchTST iTransformer)
for asset in "${ASSETS[@]}"; do
  for model in "${MODELS[@]}"; do
    python code/final_experiment/run_experiment.py --asset "$asset" --model "$model" --mode full
  done
done
