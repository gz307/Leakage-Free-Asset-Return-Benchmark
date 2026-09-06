# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


PROJECT_ROOT = os.environ.get("STOCK_EMH_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)


FILES = {

    "CSI300":
        "CSI300_daily_data.csv",

    "SP500":
        "SP500_daily_data.csv",

    "CSI1000":
        "CSI1000_daily_data.csv",

    "BTCUSD":
        "BTC_USD_daily_data.csv"
}


print("=" * 80)
print("LOG RETURN VERIFICATION")
print("=" * 80)


all_pass = True


for asset, filename in FILES.items():

    path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_csv(
        path
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df["Log_Return"] = pd.to_numeric(
        df["Log_Return"],
        errors="coerce"
    )


    # ========================================================
    # ========================================================

    recalculated = np.log(
        df["Close"]
        /
        df["Close"].shift(1)
    )


    original = (
        df["Log_Return"]
        .iloc[1:]
        .to_numpy()
    )

    recalculated = (
        recalculated
        .iloc[1:]
        .to_numpy()
    )


    diff = np.abs(
        original
        -
        recalculated
    )


    max_diff = np.nanmax(
        diff
    )

    mean_diff = np.nanmean(
        diff
    )


    # --------------------------------------------------------
    # --------------------------------------------------------

    tolerance = 1e-6

    passed = (
        max_diff
        < tolerance
    )


    if not passed:
        all_pass = False


    print("\n" + "-" * 80)

    print(
        "Asset:",
        asset
    )

    print(
        "Max absolute difference:",
        max_diff
    )

    print(
        "Mean absolute difference:",
        mean_diff
    )

    print(
        "Status:",
        "PASS"
        if passed
        else "CHECK REQUIRED"
    )


print("\n" + "=" * 80)

if all_pass:

    print(
        "ALL RETURN SERIES VERIFIED"
    )

else:

    print(
        "SOME RETURN SERIES REQUIRE CHECKING"
    )

print("=" * 80)
