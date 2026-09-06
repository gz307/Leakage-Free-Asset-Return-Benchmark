# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from common.config import (
    DATA_DIR,
    ASSETS
)


def load_asset(
    asset
):
    """Load one asset series and return aligned dates and log returns.

    Returns the prepared dataframe, a NumPy array of log returns, and the
    corresponding DatetimeIndex.
    """

    if asset not in ASSETS:

        raise ValueError(
            f"Unknown asset: {asset}"
        )


    filename = (
        ASSETS[asset]["file"]
    )

    path = os.path.join(
        DATA_DIR,
        filename
    )


    if not os.path.isfile(
        path
    ):

        raise FileNotFoundError(
            f"Data file not found:\n{path}"
        )


    df = pd.read_csv(
        path
    )


    required = [
        "Date",
        "Log_Return"
    ]


    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"{asset} Missing required column: "
                f"{column}"
            )


    df = df[
        [
            "Date",
            "Log_Return"
        ]
    ].copy()


    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise"
    )


    df["Log_Return"] = (
        pd.to_numeric(
            df["Log_Return"],
            errors="raise"
        )
    )


    # ========================================================
    # ========================================================

    if df["Date"].duplicated().any():

        raise ValueError(
            f"{asset}: Duplicate dates detected"
        )


    if not df[
        "Date"
    ].is_monotonic_increasing:

        raise ValueError(
            f"{asset}: Dates are not in ascending order"
        )


    if df[
        "Log_Return"
    ].isna().any():

        raise ValueError(
            f"{asset}: Log_Return NaN values detected"
        )


    if not np.isfinite(
        df["Log_Return"].values
    ).all():

        raise ValueError(
            f"{asset}: Log_Return Infinite values detected"
        )


    returns = (
        df["Log_Return"]
        .to_numpy(
            dtype=np.float64
        )
    )


    dates = pd.DatetimeIndex(
        df["Date"]
    )


    return (
        df,
        returns,
        dates
    )
