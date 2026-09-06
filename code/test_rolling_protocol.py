# -*- coding: utf-8 -*-

from common.config import (
    WINDOW,
    VAL_SIZE,
    SEQ_LEN
)

from common.data import (
    load_asset
)

from common.rolling import (
    make_rolling_split
)


ASSETS = [
    "CSI300",
    "SP500",
    "CSI1000",
    "BTCUSD"
]


print("=" * 90)

print(
    "ROLLING-ORIGIN "
    "NO-LEAKAGE AUDIT"
)

print("=" * 90)


for asset in ASSETS:

    print("\n")
    print("-" * 90)

    print(
        "ASSET:",
        asset
    )

    print("-" * 90)


    (
        df,
        returns,
        dates
    ) = load_asset(
        asset
    )


    # ========================================================
    #
    # ========================================================

    origins = [

        WINDOW,

        (
            WINDOW
            +
            (
                len(returns)
                -
                WINDOW
            )
            // 2
        ),

        len(
            returns
        ) - 1
    ]


    for origin in origins:

        split = make_rolling_split(
            returns=returns,
            dates=dates,
            origin=origin,
            window=WINDOW,
            val_size=VAL_SIZE,
            seq_len=SEQ_LEN
        )


        print("\nOrigin index:")
        print(
            split["origin"]
        )


        print(
            "Window start:",
            split[
                "window_start_date"
            ]
        )


        print(
            "Last known date:",
            split[
                "last_known_date"
            ]
        )


        print(
            "Target date:",
            split[
                "target_date"
            ]
        )


        print(
            "Train X:",
            split[
                "train_X"
            ].shape
        )


        print(
            "Train y:",
            split[
                "train_y"
            ].shape
        )


        print(
            "Validation X:",
            split[
                "val_X"
            ].shape
        )


        print(
            "Validation y:",
            split[
                "val_y"
            ].shape
        )


        print(
            "OOS X:",
            split[
                "test_X"
            ].shape
        )


        print(
            "HM:",
            split[
                "hm_pred"
            ]
        )


        print(
            "PER:",
            split[
                "per_pred"
            ]
        )


        # ====================================================
        # ====================================================

        assert (
            split[
                "last_known_date"
            ]
            <
            split[
                "target_date"
            ]
        )


        assert (
            split[
                "train_X"
            ].shape[1]
            ==
            SEQ_LEN
        )


        assert (
            split[
                "val_X"
            ].shape[1]
            ==
            SEQ_LEN
        )


        assert (
            split[
                "test_X"
            ].shape
            ==
            (
                1,
                SEQ_LEN,
                1
            )
        )


        # ====================================================
        #
        # train target positions:
        # 60 ... 377
        #
        #
        # 378 - 60 = 318
        #
        # validation target positions:
        # 378 ... 503
        #
        # ====================================================

        expected_train = (
            (
                WINDOW
                -
                VAL_SIZE
            )
            -
            SEQ_LEN
        )


        expected_val = (
            VAL_SIZE
        )


        assert (
            len(
                split[
                    "train_y"
                ]
            )
            ==
            expected_train
        )


        assert (
            len(
                split[
                    "val_y"
                ]
            )
            ==
            expected_val
        )


        print(
            "Leakage check: PASS"
        )


print("\n" + "=" * 90)

print(
    "ALL ASSETS PASSED "
    "ROLLING-ORIGIN AUDIT"
)

print("=" * 90)
