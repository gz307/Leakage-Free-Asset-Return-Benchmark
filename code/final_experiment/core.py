# -*- coding: utf-8 -*-

import os
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.preprocessing import (
    StandardScaler
)

from config import (
    DATA_DIR,
    ASSETS
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Data
# ============================================================

def load_asset(asset):

    if asset not in ASSETS:

        raise ValueError(
            f"Unknown asset: {asset}"
        )

    path = os.path.join(
        DATA_DIR,
        ASSETS[asset]
    )

    if not os.path.isfile(path):

        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    if df["Date"].duplicated().any():

        raise RuntimeError(
            f"{asset}: duplicate dates"
        )

    returns = (
        df["Log_Return"]
        .astype(float)
        .values
    )

    if not np.isfinite(
        returns
    ).all():

        raise RuntimeError(
            f"{asset}: invalid returns"
        )

    dates = pd.DatetimeIndex(
        df["Date"]
    )

    return (
        returns,
        dates
    )


# ============================================================
# Rolling split
# ============================================================

def make_split(
    returns,
    dates,
    origin,
    window,
    val_size,
    seq_len
):

    window_start = (
        origin - window
    )

    history = returns[
        window_start:origin
    ]

    history_dates = dates[
        window_start:origin
    ]

    if len(history) != window:

        raise RuntimeError(
            "Wrong window length"
        )


    # ========================================================
    # OOS
    # ========================================================

    y_true = float(
        returns[origin]
    )

    target_date = (
        dates[origin]
    )

    last_known_date = (
        history_dates[-1]
    )


    if not (
        last_known_date
        <
        target_date
    ):

        raise RuntimeError(
            "DATA LEAKAGE"
        )


    # ========================================================
    # Baselines
    # ========================================================

    hm = float(
        np.mean(history)
    )

    per = float(
        history[-1]
    )


    # ========================================================
    # Train / validation
    # ========================================================

    validation_start = (
        window
        -
        val_size
    )


    train_X = []
    train_y = []

    val_X = []
    val_y = []


    for t in range(
        seq_len,
        window
    ):

        x = history[
            t-seq_len:t
        ]

        y = history[t]


        if t < validation_start:

            train_X.append(x)

            train_y.append(y)

        else:

            val_X.append(x)

            val_y.append(y)


    train_X = np.asarray(
        train_X,
        dtype=np.float32
    )[..., None]

    train_y = np.asarray(
        train_y,
        dtype=np.float32
    ).reshape(-1, 1)


    val_X = np.asarray(
        val_X,
        dtype=np.float32
    )[..., None]

    val_y = np.asarray(
        val_y,
        dtype=np.float32
    ).reshape(-1, 1)


    test_X = (
        history[-seq_len:]
        .astype(np.float32)
        .reshape(
            1,
            seq_len,
            1
        )
    )


    return {

        "target_date":
            target_date,

        "last_known_date":
            last_known_date,

        "window_start_date":
            history_dates[0],

        "true":
            y_true,

        "HM":
            hm,

        "PER":
            per,

        "train_X":
            train_X,

        "train_y":
            train_y,

        "val_X":
            val_X,

        "val_y":
            val_y,

        "test_X":
            test_X
    }


# ============================================================
# Scaling
#
#
#
# ============================================================

def scale_split(split):

    x_scaler = StandardScaler()

    y_scaler = StandardScaler()


    # ========================================================
    # Fit ONLY training data
    # ========================================================

    x_scaler.fit(
        split["train_X"]
        .reshape(-1, 1)
    )

    y_scaler.fit(
        split["train_y"]
    )


    train_X = x_scaler.transform(
        split["train_X"]
        .reshape(-1, 1)
    ).reshape(
        split["train_X"].shape
    )


    val_X = x_scaler.transform(
        split["val_X"]
        .reshape(-1, 1)
    ).reshape(
        split["val_X"].shape
    )


    test_X = x_scaler.transform(
        split["test_X"]
        .reshape(-1, 1)
    ).reshape(
        split["test_X"].shape
    )


    train_y = y_scaler.transform(
        split["train_y"]
    )


    val_y = y_scaler.transform(
        split["val_y"]
    )


    return {

        "train_X":
            train_X.astype(
                np.float32
            ),

        "train_y":
            train_y.astype(
                np.float32
            ),

        "val_X":
            val_X.astype(
                np.float32
            ),

        "val_y":
            val_y.astype(
                np.float32
            ),

        "test_X":
            test_X.astype(
                np.float32
            ),

        "x_scaler":
            x_scaler,

        "y_scaler":
            y_scaler
    }


# ============================================================
# Generic trainer
# ============================================================

def train_model(
    model,
    train_X,
    train_y,
    val_X,
    val_y,
    lr,
    epochs,
    batch_size,
    patience,
    device
):

    model = model.to(
        device
    )


    dataset = TensorDataset(

        torch.tensor(
            train_X,
            dtype=torch.float32
        ),

        torch.tensor(
            train_y,
            dtype=torch.float32
        )
    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False
    )


    val_X_t = torch.tensor(
        val_X,
        dtype=torch.float32,
        device=device
    )

    val_y_t = torch.tensor(
        val_y,
        dtype=torch.float32,
        device=device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    loss_fn = nn.MSELoss()


    best_loss = np.inf

    best_state = None

    best_epoch = None

    counter = 0


    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()


        for bx, by in loader:

            bx = bx.to(device)

            by = by.to(device)


            optimizer.zero_grad(
                set_to_none=True
            )


            pred = model(bx)


            if pred.shape != by.shape:

                raise RuntimeError(
                    f"Prediction shape "
                    f"{pred.shape}, "
                    f"target {by.shape}"
                )


            loss = loss_fn(
                pred,
                by
            )


            if not torch.isfinite(loss):

                raise RuntimeError(
                    "NaN/Inf training loss"
                )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()


        # ====================================================
        # Validation
        # ====================================================

        model.eval()


        with torch.no_grad():

            val_pred = model(
                val_X_t
            )

            val_loss = loss_fn(
                val_pred,
                val_y_t
            ).item()


        if val_loss < best_loss:

            best_loss = (
                val_loss
            )

            best_epoch = (
                epoch
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            counter = 0

        else:

            counter += 1


        if counter >= patience:

            break


    if best_state is None:

        raise RuntimeError(
            "Training failed"
        )


    model.load_state_dict(
        best_state
    )


    return (
        model,
        float(best_loss),
        int(best_epoch)
    )


# ============================================================
# One-step prediction
# ============================================================

def predict_original_scale(
    model,
    test_X,
    y_scaler,
    device
):

    model.eval()


    x = torch.tensor(
        test_X,
        dtype=torch.float32,
        device=device
    )


    with torch.no_grad():

        pred_scaled = (
            model(x)
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )


    pred = y_scaler.inverse_transform(
        pred_scaled
    )[0, 0]


    if not np.isfinite(pred):

        raise RuntimeError(
            "Invalid OOS prediction"
        )


    return float(pred)
