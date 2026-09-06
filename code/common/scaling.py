# -*- coding: utf-8 -*-

import numpy as np

from sklearn.preprocessing import StandardScaler


def fit_and_transform(
    train_X,
    val_X,
    test_X
):
    """Standardize rolling-window inputs without information leakage.

    The scaler is fit on the training inputs only; validation and test inputs
    are transformed with the fitted parameters and are never used to refit it.
    """

    # ========================================================
    # ========================================================

    if train_X.ndim != 3:
        raise ValueError(
            "train_X must have shape [N, L, C]"
        )

    if val_X.ndim != 3:
        raise ValueError(
            "val_X must have shape [N, L, C]"
        )

    if test_X.ndim != 3:
        raise ValueError(
            "test_X must have shape [N, L, C]"
        )


    # ========================================================
    # ========================================================

    if train_X.shape[-1] != 1:

        raise ValueError(
            "The main experiment requires d=1"
        )


    # ========================================================
    # Fit ONLY on training inputs
    # ========================================================

    scaler = StandardScaler()

    train_flat = train_X.reshape(
        -1,
        1
    )

    scaler.fit(
        train_flat
    )


    # ========================================================
    # Transform
    # ========================================================

    train_scaled = scaler.transform(
        train_X.reshape(-1, 1)
    ).reshape(
        train_X.shape
    )


    val_scaled = scaler.transform(
        val_X.reshape(-1, 1)
    ).reshape(
        val_X.shape
    )


    test_scaled = scaler.transform(
        test_X.reshape(-1, 1)
    ).reshape(
        test_X.shape
    )


    # ========================================================
    # Finite check
    # ========================================================

    for name, x in [

        (
            "train",
            train_scaled
        ),

        (
            "validation",
            val_scaled
        ),

        (
            "test",
            test_scaled
        )
    ]:

        if not np.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"{name} scaling "
                "produced NaN/Inf"
            )


    return (
        train_scaled.astype(
            np.float32
        ),

        val_scaled.astype(
            np.float32
        ),

        test_scaled.astype(
            np.float32
        ),

        scaler
    )
