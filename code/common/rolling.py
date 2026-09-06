# -*- coding: utf-8 -*-

import numpy as np


def make_rolling_split(
    returns,
    dates,
    origin,
    window,
    val_size,
    seq_len
):
    """Construct one leakage-free rolling forecast origin.

    `origin` is the position of the out-of-sample target. The model may use
    only the preceding `window` observations, while `returns[origin]` is the
    forecast target and is excluded from all fitting and validation inputs.
    """

    if origin < window:

        raise ValueError(
            "origin must be at least WINDOW"
        )


    # ========================================================
    # ========================================================

    window_start = (
        origin
        -
        window
    )

    window_end = origin

    window_data = returns[
        window_start:
        window_end
    ]

    window_dates = dates[
        window_start:
        window_end
    ]


    assert (
        len(window_data)
        ==
        window
    )


    # ========================================================
    # OOS target
    # ========================================================

    y_true = float(
        returns[origin]
    )

    target_date = (
        dates[origin]
    )


    # ========================================================
    # HM
    #
    # ========================================================

    hm_pred = float(
        np.mean(
            window_data
        )
    )


    # ========================================================
    # PER
    #
    # ========================================================

    per_pred = float(
        window_data[-1]
    )


    # ========================================================
    #
    #
    # | model fitting | validation |
    #
    # ========================================================

    val_target_start = (
        window
        -
        val_size
    )


    train_X = []
    train_y = []
    train_target_dates = []

    val_X = []
    val_y = []
    val_target_dates = []


    # ========================================================
    #
    #   [t-L, ..., t-1]
    #
    #   t
    #
    # ========================================================

    for target_pos in range(
        seq_len,
        window
    ):

        x = window_data[
            target_pos-seq_len:
            target_pos
        ]

        y = window_data[
            target_pos
        ]

        y_date = window_dates[
            target_pos
        ]


        if (
            target_pos
            <
            val_target_start
        ):

            train_X.append(
                x
            )

            train_y.append(
                y
            )

            train_target_dates.append(
                y_date
            )

        else:

            val_X.append(
                x
            )

            val_y.append(
                y
            )

            val_target_dates.append(
                y_date
            )


    train_X = np.asarray(
        train_X,
        dtype=np.float32
    )[..., None]


    train_y = np.asarray(
        train_y,
        dtype=np.float32
    ).reshape(
        -1,
        1
    )


    val_X = np.asarray(
        val_X,
        dtype=np.float32
    )[..., None]


    val_y = np.asarray(
        val_y,
        dtype=np.float32
    ).reshape(
        -1,
        1
    )


    # ========================================================
    #
    # ========================================================

    test_X = (
        window_data[
            -seq_len:
        ]
        .astype(
            np.float32
        )
        .reshape(
            1,
            seq_len,
            1
        )
    )


    # ========================================================
    # ========================================================

    last_known_date = (
        window_dates[-1]
    )


    if not (
        last_known_date
        <
        target_date
    ):

        raise RuntimeError(
            "DATA LEAKAGE: "
            "target date <= last known date"
        )


    # Training target
    if len(
        train_target_dates
    ) > 0:

        if not (
            max(
                train_target_dates
            )
            <
            target_date
        ):

            raise RuntimeError(
                "DATA LEAKAGE "
                "in training targets"
            )


    # Validation target
    if len(
        val_target_dates
    ) > 0:

        if not (
            max(
                val_target_dates
            )
            <
            target_date
        ):

            raise RuntimeError(
                "DATA LEAKAGE "
                "in validation targets"
            )


    return {

        # ---------------------------------------------
        # OOS
        # ---------------------------------------------

        "origin":
            origin,

        "target_date":
            target_date,

        "y_true":
            y_true,


        # ---------------------------------------------
        # Information boundary
        # ---------------------------------------------

        "window_start_date":
            window_dates[0],

        "last_known_date":
            last_known_date,


        # ---------------------------------------------
        # Benchmarks
        # ---------------------------------------------

        "hm_pred":
            hm_pred,

        "per_pred":
            per_pred,


        # ---------------------------------------------
        # Neural data
        # ---------------------------------------------

        "train_X":
            train_X,

        "train_y":
            train_y,

        "val_X":
            val_X,

        "val_y":
            val_y,

        "test_X":
            test_X,


        # ---------------------------------------------
        # Audit dates
        # ---------------------------------------------

        "train_target_dates":
            train_target_dates,

        "val_target_dates":
            val_target_dates
    }
