#!/usr/bin/env bash
set -euo pipefail
# Window-length robustness: LSTM and standard Transformer; W = 252 and 756 (W = 504 is the baseline).
for asset in CSI300 SP500 CSI1000 BTCUSD; do
  for model in LSTM Transformer; do
    python code/robustness_experiment/run_window_robustness.py --asset "$asset" --model "$model" --window 252 --mode full
    python code/robustness_experiment/run_window_robustness.py --asset "$asset" --model "$model" --window 756 --mode full
    python code/robustness_experiment/run_seq_len_robustness.py --asset "$asset" --model "$model" --seq_len 20
    python code/robustness_experiment/run_seq_len_robustness.py --asset "$asset" --model "$model" --seq_len 120
  done
  # LSTM re-tuning interval robustness; 63 is the baseline.
  python code/robustness_experiment/run_retune_robustness.py --asset "$asset" --tune_every 21
  python code/robustness_experiment/run_retune_robustness.py --asset "$asset" --tune_every 126
done
