# -*- coding: utf-8 -*-
"""
make_descriptive_stats.py -- regenerate Table 1 from the raw return files.

The four raw price files are not redistributed with the replication archive, because
the data providers restrict redistribution. Place them in a folder and point the script
at it:

    python analysis/make_descriptive_stats.py --data /path/to/data --out tables

The script expects the files produced by the retrieval procedure described in
data/DATA_MANIFEST.csv:
    CSI300_daily_data.csv, SP500_daily_data.csv, CSI1000_daily_data.csv, BTC_USD_daily_data.csv
each with columns Date, Close, Log_Return.

Kurtosis is reported as EXCESS kurtosis (Fisher), following the manuscript's Table 1;
the value for a normal distribution is zero.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

FILES = {
    "CSI 300": "CSI300_daily_data.csv",
    "S&P 500": "SP500_daily_data.csv",
    "CSI 1000": "CSI1000_daily_data.csv",
    "BTC/USD": "BTC_USD_daily_data.csv",
}

# Values printed in Table 1 of the manuscript, used as a consistency check.
TABLE1 = {
    "CSI 300": dict(n=4015, start="2010-01-04", end="2026-07-17",
                    mean=0.006, sd=1.364, skew=-0.570, kurt=5.354),
    "S&P 500": dict(n=4162, start="2010-01-04", end="2026-07-17",
                    mean=0.046, sd=1.087, skew=-0.609, kurt=13.477),
    "CSI 1000": dict(n=2856, start="2014-10-20", end="2026-07-17",
                     mean=0.006, sd=1.751, skew=-0.842, kurt=4.871),
    "BTC/USD": dict(n=4320, start="2014-09-18", end="2026-07-16",
                    mean=0.114, sd=3.502, skew=-0.707, kurt=11.886),
}

TOL = 0.02      # tolerance on the printed precision of Table 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding the four raw CSV files")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    failures = 0

    for name, fname in FILES.items():
        path = os.path.join(args.data, fname)
        if not os.path.isfile(path):
            print(f"missing: {path}")
            failures += 1
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.replace("/", "-", regex=False))
        r = df["Log_Return"].astype(float)
        row = dict(
            Asset=name,
            Start=str(df["Date"].min().date()),
            End=str(df["Date"].max().date()),
            N=len(df),
            Mean_pct=round(r.mean() * 100, 6),
            SD_pct=round(r.std(ddof=1) * 100, 6),
            Skewness=round(r.skew(), 6),
            Excess_kurtosis=round(r.kurt(), 6),
        )
        rows.append(row)

        exp = TABLE1[name]
        checks = [("N", row["N"], exp["n"]), ("start", row["Start"], exp["start"]),
                  ("end", row["End"], exp["end"])]
        numeric = [("mean", row["Mean_pct"], exp["mean"]), ("sd", row["SD_pct"], exp["sd"]),
                   ("skew", row["Skewness"], exp["skew"]),
                   ("kurt", row["Excess_kurtosis"], exp["kurt"])]
        ok = all(a == b for _, a, b in checks) and all(abs(a - b) <= TOL for _, a, b in numeric)
        print(f"{name:<10} {'PASS' if ok else 'FAIL'}"
              f"  N={row['N']}  mean={row['Mean_pct']:.3f}%  sd={row['SD_pct']:.3f}%"
              f"  skew={row['Skewness']:.3f}  exkurt={row['Excess_kurtosis']:.3f}")
        if not ok:
            failures += 1
            for label, a, b in numeric:
                if abs(a - b) > TOL:
                    print(f"    {label}: computed {a:.3f}, Table 1 {b:.3f}")

    table = pd.DataFrame(rows)
    out_file = os.path.join(args.out, "table1.csv")
    table.to_csv(out_file, index=False)
    print()
    print("saved:", out_file)
    print(f"assets checked: {len(table)}   failures: {failures}")
    print("note: kurtosis is excess kurtosis (normal distribution = 0)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
