# -*- coding: utf-8 -*-
"""
report_model_sizes.py -- trainable parameter counts of the selected configurations.

For every asset and model, the script reads the hyperparameters recorded in the
archived forecast files, builds the corresponding network for each distinct
configuration, and reports the minimum, median and maximum number of trainable
parameters. It trains nothing, so it runs in seconds on CPU.

Usage
-----
    python code/report_model_sizes.py
    python code/report_model_sizes.py --out tables/model_sizes.csv

The values feed the "Trainable parameters" column of Table 3.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_experiment"))

from config import RESULT_DIR  # noqa: E402
from model_factory import build_model  # noqa: E402

ASSETS = ["CSI300", "SP500", "CSI1000", "BTCUSD"]
MODELS = ["LSTM", "Transformer", "Autoformer", "PatchTST", "iTransformer"]


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULT_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for asset in ASSETS:
        for model_name in MODELS:
            path = os.path.join(args.results, "full", asset, f"{model_name}_predictions.csv")
            if not os.path.isfile(path):
                print(f"skipped (missing): {path}")
                continue

            df = pd.read_csv(path)
            configs = df["HyperParameters"].map(json.loads)
            distinct = {}
            for params in configs:
                key = json.dumps(params, sort_keys=True)
                distinct.setdefault(key, (params, 0))
                distinct[key] = (params, distinct[key][1] + 1)

            counts = []
            for key, (params, _) in distinct.items():
                counts.append(count_parameters(build_model(model_name, params)))
            counts = np.asarray(counts, dtype=float)

            rows.append({
                "Asset": asset,
                "Model": model_name,
                "DistinctConfigurations": len(distinct),
                "ParametersMin": int(counts.min()),
                "ParametersMedian": int(np.median(counts)),
                "ParametersMax": int(counts.max()),
            })
            print(f"{asset:8s} {model_name:13s} configs={len(distinct):3d} "
                  f"params={int(counts.min()):>7,} - {int(np.median(counts)):>7,} - {int(counts.max()):>7,}")

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
