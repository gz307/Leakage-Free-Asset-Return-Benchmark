# -*- coding: utf-8 -*-
"""
verify_provenance.py -- spot-check that the archived forecasts are reproducible
from the released pipeline.

For a sample of out-of-sample origins, the script rebuilds the rolling split,
re-uses the hyperparameters recorded in the archived forecast file, re-estimates
the network with the same seed, and compares the resulting forecast with the
archived value. It does not retrain the whole sample.

Usage
-----
    python code/verify_provenance.py --asset CSI300 --model LSTM --n 12
    python code/verify_provenance.py --asset BTCUSD --model Autoformer --n 6 --device cpu

Output
------
    results/provenance/PROVENANCE_{ASSET}_{MODEL}.csv
    a summary line with the maximum absolute deviation

Interpretation
--------------
    max |deviation| < tol  -> the archived forecast is reproduced by the pipeline
    larger deviations      -> the archived value was produced by a different code
                              path, hyperparameters, or seed; investigate before
                              claiming provenance
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_experiment"))

from config import (  # noqa: E402
    WINDOW, VAL_SIZE, SEQ_LEN, MAX_EPOCHS, BATCH_SIZE, PATIENCE, SEED, RESULT_DIR,
)
from core import (  # noqa: E402
    set_seed, get_device, load_asset, make_split, scale_split,
    train_model, predict_original_scale,
)
from model_factory import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True,
                    choices=["CSI300", "SP500", "CSI1000", "BTCUSD"])
    ap.add_argument("--model", required=True,
                    choices=["LSTM", "Transformer", "Autoformer", "PatchTST", "iTransformer"])
    ap.add_argument("--n", type=int, default=12, help="number of origins to re-execute")
    ap.add_argument("--seed", type=int, default=SEED, help="must match the archived run")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="tolerance in log-return units")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--pick", default="random", choices=["random", "even"],
                    help="random origins, or evenly spaced across the sample")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch
            pass

    pred_path = os.path.join(RESULT_DIR, "full", args.asset,
                             f"{args.model}_predictions.csv")
    if not os.path.isfile(pred_path):
        raise SystemExit(f"archived forecasts not found: {pred_path}")
    archived = pd.read_csv(pred_path)
    archived["Date"] = pd.to_datetime(archived["Date"])

    returns, dates = load_asset(args.asset)
    date_to_index = {d.normalize(): i for i, d in enumerate(dates)}

    rows = archived[archived["Date"].isin(pd.DatetimeIndex(date_to_index.keys()))]
    if args.pick == "even":
        pos = np.linspace(0, len(rows) - 1, args.n).round().astype(int)
        sample = rows.iloc[pos]
    else:
        rng = np.random.default_rng(0)
        sample = rows.iloc[np.sort(rng.choice(len(rows), size=min(args.n, len(rows)),
                                              replace=False))]

    out_rows = []
    for _, rec in sample.iterrows():
        origin = date_to_index[rec["Date"].normalize()]
        params = json.loads(rec["HyperParameters"])

        split = make_split(returns, dates, origin, WINDOW, VAL_SIZE, SEQ_LEN)
        scaled = scale_split(split)

        set_seed(args.seed)
        model = build_model(args.model, params)
        model, val_loss, best_epoch = train_model(
            model=model,
            train_X=scaled["train_X"], train_y=scaled["train_y"],
            val_X=scaled["val_X"], val_y=scaled["val_y"],
            lr=params["lr"], epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            patience=PATIENCE, device=device,
        )
        pred = predict_original_scale(model, scaled["test_X"],
                                      scaled["y_scaler"], device)

        archived_pred = float(rec["Prediction"])
        out_rows.append({
            "Date": rec["Date"].date(),
            "ArchivedPrediction": archived_pred,
            "ReexecutedPrediction": pred,
            "AbsDifference": abs(pred - archived_pred),
            "RelativeDifference": abs(pred - archived_pred) / max(abs(archived_pred), 1e-12),
            "ArchivedValLoss": float(rec["BestValLoss"]),
            "ReexecutedValLoss": float(val_loss),
            "ArchivedBestEpoch": int(rec["BestEpoch"]),
            "ReexecutedBestEpoch": int(best_epoch),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(out_rows)
    out_dir = os.path.join(RESULT_DIR, "..", "provenance")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"PROVENANCE_{args.asset}_{args.model}.csv")
    result.to_csv(out_file, index=False)

    max_abs = result["AbsDifference"].max()
    max_rel = result["RelativeDifference"].max()
    print()
    print(result.to_string(index=False))
    print()
    print(f"origins re-executed        : {len(result)}")
    print(f"max absolute deviation     : {max_abs:.3e}  (log-return units)")
    print(f"max relative deviation     : {max_rel:.3e}")
    print(f"tolerance                  : {args.tol:.1e}")
    verdict = "PASS" if max_abs <= args.tol else "FAIL"
    print(f"verdict                    : {verdict}")
    print(f"saved                      : {out_file}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
