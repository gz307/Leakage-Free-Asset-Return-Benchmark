# -*- coding: utf-8 -*-
"""
make_tables.py -- regenerate Table 4, Table 5 and Table 6 of the manuscript
from archived out-of-sample forecast files.

Usage
-----
    python make_tables.py --results results/final --out tables

Inputs (must exist)
-------------------
    results/final/analysis/{ASSET}_OOS_ALL.csv
        columns: Date, True, HM, PER, LSTM, Transformer, Autoformer, PatchTST, iTransformer
    results/robustness/analysis/WINDOW_ROBUSTNESS_LONG.csv
    results/robustness/analysis/SEQLEN_ROBUSTNESS_LONG.csv
    results/robustness/analysis/RETUNE_ROBUSTNESS_LONG.csv

Outputs
-------
    tables/table4.csv            point estimates (MSE, MAE, R2OS, DA, N)
    tables/table5.csv            predictive-accuracy tests vs HM and PER
    tables/table5_bandwidth.csv  Clark-West with lag 1 vs automatic bandwidth
    tables/table6.csv            robustness of R2OS to W, L and the re-tuning interval

Statistical conventions (must match the manuscript)
---------------------------------------------------
* Forecast loss is squared error.
* R2OS = 1 - SSE(model) / SSE(HM); HM has R2OS = 0 by construction.
* Newey-West (Bartlett) HAC variance.
  - Primary specification: one lag (as stated in Section 3.6 of the manuscript).
  - Sensitivity: Newey-West (1994) automatic bandwidth
        floor(4 * (n/100)^(2/9)), bounded to [1, n-1].
* Clark-West: f_t = e_HM,t^2 - e_m,t^2 + (HM_t - m_t)^2,
  one-sided upper-tail standard-normal p-value.
* Realised loss differential: d_t = e_m,t^2 - e_HM,t^2 (positive = model worse than HM),
  two-sided p-value, Holm-adjusted over the 20 model-asset comparisons.
* DM-HLN against PER: d_t = e_PER,t^2 - e_m,t^2 (positive = model better than PER),
  HLN small-sample correction, one-sided p-value from the standard normal.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

ASSETS = ["CSI300", "SP500", "CSI1000", "BTCUSD"]
MODELS = ["LSTM", "Transformer", "Autoformer", "PatchTST", "iTransformer"]


# ----------------------------------------------------------------------
# HAC helpers
# ----------------------------------------------------------------------
def nw_variance(x, lag):
    """Newey-West long-run variance with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    s = float(np.dot(xc, xc) / n)
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        s += 2.0 * w * float(np.dot(xc[ell:], xc[:-ell]) / n)
    return s


def auto_bandwidth(n):
    """Newey-West (1994) plug-in bandwidth, bounded to [1, n-1]."""
    return int(max(1, min(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)), n - 1)))


def normal_sf(z):
    """Upper-tail standard-normal probability."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def holm(pvalues):
    """Holm step-down adjustment."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * p[idx], 1.0))
        adjusted[idx] = running
    return adjusted


# ----------------------------------------------------------------------
# Table 4
# ----------------------------------------------------------------------
def build_table4(frames):
    rows = []
    for asset, df in frames.items():
        y = df["True"].to_numpy()
        sse_hm = np.sum((y - df["HM"].to_numpy()) ** 2)
        for model in ["HM", "PER"] + MODELS:
            pred = df[model].to_numpy()
            err = y - pred
            rows.append({
                "Asset": asset,
                "Model": model,
                "MSE": float(np.mean(err ** 2)),
                "MAE": float(np.mean(np.abs(err))),
                "R2OS": float(1.0 - np.sum(err ** 2) / sse_hm),
                "DA": float(np.mean(np.sign(y) == np.sign(pred))),
                "N": int(y.size),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Table 5
# ----------------------------------------------------------------------
def build_table5(frames, lag=1):
    rows = []
    for asset, df in frames.items():
        y = df["True"].to_numpy()
        n = y.size
        hm = df["HM"].to_numpy()
        per = df["PER"].to_numpy()
        e_hm = y - hm
        e_per = y - per
        sse_hm = np.sum(e_hm ** 2)
        bw = auto_bandwidth(n)
        for model in MODELS:
            pred = df[model].to_numpy()
            e_m = y - pred

            # Clark-West adjusted differential
            f = e_hm ** 2 - e_m ** 2 + (hm - pred) ** 2
            cw_lag1 = f.mean() / math.sqrt(nw_variance(f, 1) / n)
            cw_auto = f.mean() / math.sqrt(nw_variance(f, bw) / n)

            # Realised squared-loss differential, positive = model worse than HM
            d = e_m ** 2 - e_hm ** 2
            se_lag1 = math.sqrt(nw_variance(d, lag) / n)
            se_auto = math.sqrt(nw_variance(d, bw) / n)
            z_lag1 = d.mean() / se_lag1
            z_auto = d.mean() / se_auto

            # DM-HLN against PER, positive = model better than PER
            dm_d = e_per ** 2 - e_m ** 2
            dm = dm_d.mean() / math.sqrt(nw_variance(dm_d, 1) / n)
            hln = math.sqrt((n + 1 - 2 + 1.0 / n) / n)

            rows.append({
                "Asset": asset,
                "Model": model,
                "N": int(n),
                "R2OS": float(1.0 - np.sum(e_m ** 2) / sse_hm),
                "CW_lag1": float(cw_lag1),
                "p_CW_lag1": float(normal_sf(cw_lag1)),
                "CW_auto": float(cw_auto),
                "p_CW_auto": float(normal_sf(cw_auto)),
                "dMSE_x1e4": float(d.mean() * 1e4),
                "dMSE_SE_x1e4": float(se_lag1 * 1e4),
                "dMSE_CI95_low": float((d.mean() - 1.96 * se_lag1) * 1e4),
                "dMSE_CI95_high": float((d.mean() + 1.96 * se_lag1) * 1e4),
                "p_dMSE_lag1": float(2.0 * normal_sf(abs(z_lag1))),
                "p_dMSE_auto": float(2.0 * normal_sf(abs(z_auto))),
                "DM_HLN": float(dm * hln),
                "p_DM": float(normal_sf(dm * hln)),
                "HAC_bandwidth_auto": int(bw),
            })
    table = pd.DataFrame(rows)
    table["p_CW_holm"] = holm(table["p_CW_lag1"])
    table["p_dMSE_holm"] = holm(table["p_dMSE_lag1"])
    table["sig_CW_holm"] = table["p_CW_holm"] < 0.05
    table["sig_dMSE_holm"] = table["p_dMSE_holm"] < 0.05
    return table


# ----------------------------------------------------------------------
# Table 6
# ----------------------------------------------------------------------
def build_table6(results_root):
    """results_root is the robustness results directory (…/results/robustness)."""
    anal = os.path.join(results_root, "analysis")
    out = []

    path = os.path.join(anal, "WINDOW_ROBUSTNESS_LONG.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        col_r2 = [c for c in df.columns if c.lower() in ("ros2", "r_os2", "r2os")][0]
        col_w = [c for c in df.columns if c.lower() in ("window", "w")][0]
        pivot = df.pivot_table(index=["Asset", "Model"], columns=col_w, values=col_r2)
        for (asset, model), row in pivot.iterrows():
            out.append({"Panel": "A: window length W", "Asset": asset, "Model": model,
                        **{f"value_{int(k)}": float(v) for k, v in row.dropna().items()}})

    path = os.path.join(anal, "SEQLEN_ROBUSTNESS_LONG.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        col_r2 = [c for c in df.columns if c.lower() in ("ros2", "r_os2", "r2os")][0]
        col_l = [c for c in df.columns if c.lower() in ("seqlen", "seq_len", "seqlength", "l")][0]
        pivot = df.pivot_table(index=["Asset", "Model"], columns=col_l, values=col_r2)
        for (asset, model), row in pivot.iterrows():
            out.append({"Panel": "B: input length L", "Asset": asset, "Model": model,
                        **{f"value_{int(k)}": float(v) for k, v in row.dropna().items()}})

    path = os.path.join(anal, "RETUNE_ROBUSTNESS_LONG.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        col_r2 = [c for c in df.columns if c.lower() in ("ros2", "r_os2", "r2os")][0]
        col_k = [c for c in df.columns
                 if c.lower() in ("tuneevery", "retune", "interval", "tune_every", "k")][0]
        pivot = df.pivot_table(index=["Asset", "Model"], columns=col_k, values=col_r2)
        for (asset, model), row in pivot.iterrows():
            out.append({"Panel": "C: re-tuning interval", "Asset": asset, "Model": model,
                        **{f"value_{int(k)}": float(v) for k, v in row.dropna().items()}})

    return pd.DataFrame(out)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final")
    ap.add_argument("--robustness", default="results/robustness")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frames = {}
    for asset in ASSETS:
        path = os.path.join(args.results, "analysis", f"{asset}_OOS_ALL.csv")
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        frames[asset] = df.sort_values("Date").reset_index(drop=True)

    table4 = build_table4(frames)
    table5 = build_table5(frames)
    table6 = build_table6(args.robustness)

    table4.to_csv(os.path.join(args.out, "table4.csv"), index=False, float_format="%.10f")
    table5.to_csv(os.path.join(args.out, "table5.csv"), index=False, float_format="%.10f")
    table6.to_csv(os.path.join(args.out, "table6.csv"), index=False, float_format="%.10f")

    bw = table5[["Asset", "Model", "CW_lag1", "p_CW_lag1", "CW_auto", "p_CW_auto",
                 "HAC_bandwidth_auto"]]
    bw.to_csv(os.path.join(args.out, "table5_bandwidth.csv"), index=False, float_format="%.10f")

    print(f"Table 4: {len(table4)} rows -> {args.out}/table4.csv")
    print(f"Table 5: {len(table5)} rows -> {args.out}/table5.csv")
    print(f"Table 6: {len(table6)} rows -> {args.out}/table6.csv")
    print()
    print("Clark-West, Holm-adjusted, minimum p = "
          f"{table5['p_CW_holm'].min():.4f} "
          f"({int(table5['sig_CW_holm'].sum())} significant at 5%)")
    print("Realised loss differential, Holm-adjusted, "
          f"{int(table5['sig_dMSE_holm'].sum())} of {len(table5)} significant at 5%, "
          f"{int((table5['dMSE_x1e4'] > 0).sum())} positive (model worse than HM)")


if __name__ == "__main__":
    main()
