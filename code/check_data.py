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

print(
    "FINANCIAL DATA CHECK"
)

print("=" * 80)


for asset, filename in FILES.items():

    print("\n")
    print("-" * 80)

    print(
        "ASSET:",
        asset
    )

    print("-" * 80)

    path = os.path.join(
        DATA_DIR,
        filename
    )

    print(
        "File:",
        path
    )


    if not os.path.isfile(
        path
    ):

        print(
            "STATUS: FILE NOT FOUND"
        )

        continue


    df = pd.read_csv(
        path
    )


    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )


    print("\nFirst 3 rows:")

    print(
        df.head(3)
    )


    print("\nLast 3 rows:")

    print(
        df.tail(3)
    )


    # ========================================================
    # Log_Return
    # ========================================================

    if (
        "Log_Return"
        not in df.columns
    ):

        print(
            "\nWARNING:"
            " Log_Return column not found."
        )

        continue


    returns = pd.to_numeric(
        df["Log_Return"],
        errors="coerce"
    )


    print(
        "\nLog_Return missing:",
        returns.isna().sum()
    )


    print(
        "Log_Return inf:",
        np.isinf(
            returns.dropna()
        ).sum()
    )


    print(
        "Mean:",
        returns.mean()
    )


    print(
        "Std:",
        returns.std()
    )


    print(
        "Min:",
        returns.min()
    )


    print(
        "Max:",
        returns.max()
    )


    # ========================================================
    # ========================================================

    possible_dates = [
        "Date",
        "date",
        "Datetime",
        "datetime"
    ]


    date_col = None


    for candidate in possible_dates:

        if candidate in df.columns:

            date_col = candidate
            break


    if date_col is None:

        first_column = df.columns[0]

        candidate_dates = pd.to_datetime(
            df[first_column],
            errors="coerce"
        )

        if (
            candidate_dates
            .notna()
            .mean()
            > 0.9
        ):

            date_col = first_column


    if date_col is not None:

        dates = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )


        print(
            "\nDate column:",
            date_col
        )


        print(
            "Invalid dates:",
            dates.isna().sum()
        )


        valid_dates = (
            dates
            .dropna()
        )


        if len(
            valid_dates
        ) > 0:

            print(
                "Start:",
                valid_dates.min()
            )

            print(
                "End:",
                valid_dates.max()
            )

            print(
                "Duplicate dates:",
                valid_dates.duplicated().sum()
            )

            print(
                "Monotonic increasing:",
                valid_dates.is_monotonic_increasing
            )

    else:

        print(
            "\nWARNING:"
            " Date column could not be identified."
        )


print("\n" + "=" * 80)

print(
    "DATA CHECK FINISHED"
)

print("=" * 80)
