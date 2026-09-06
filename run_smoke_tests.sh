#!/usr/bin/env bash
set -euo pipefail

python code/test_environment.py
python code/check_official_sources.py

required_data=(
  data/CSI300_daily_data.csv
  data/SP500_daily_data.csv
  data/CSI1000_daily_data.csv
  data/BTC_USD_daily_data.csv
)

missing=0
for path in "${required_data[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing=1
  fi
done

if [[ "$missing" -eq 0 ]]; then
  python code/check_data.py
  python code/verify_returns.py
  python code/test_rolling_protocol.py
else
  echo
  echo "Raw third-party market data are not distributed with this repository."
  echo "Data-dependent checks were skipped."
  echo "See data/README.md, place the four prepared CSV files in data/, and rerun this script."
fi
