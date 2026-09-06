# -*- coding: utf-8 -*-

import copy
import random

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# ============================================================
# Random seed
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


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
# Generic training
# ============================================================

def train_with_early_stopping(
    model,
    train_X,
    train_y,
    val_X,
    val_y,
    learning_rate,
    max_epochs,
    batch_size,
    patience,
    device
):
    """Train one model under the rolling-origin protocol.

    Early stopping is based only on validation MSE.
    """

    model = model.to(
        device
    )


    # ========================================================
    # Tensor
    # ========================================================

    train_X_t = torch.tensor(
        train_X,
        dtype=torch.float32
    )

    train_y_t = torch.tensor(
        train_y,
        dtype=torch.float32
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


    dataset = TensorDataset(
        train_X_t,
        train_y_t
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,

        shuffle=False
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    criterion = nn.MSELoss()


    # ========================================================
    # Early stopping
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_state = None

    best_epoch = None

    no_improve = 0


    for epoch in range(
        1,
        max_epochs + 1
    ):

        # ====================================================
        # Train
        # ====================================================

        model.train()


        for bx, by in loader:

            bx = bx.to(
                device
            )

            by = by.to(
                device
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            pred = model(
                bx
            )


            if pred.shape != by.shape:

                raise RuntimeError(
                    "Model output shape "
                    f"{pred.shape} != "
                    f"target shape {by.shape}"
                )


            loss = criterion(
                pred,
                by
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Training loss became "
                    "NaN/Inf"
                )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
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

            val_loss = criterion(
                val_pred,
                val_y_t
            ).item()


        if not np.isfinite(
            val_loss
        ):

            raise RuntimeError(
                "Validation loss became "
                "NaN/Inf"
            )


        # ====================================================
        # Save best
        # ====================================================

        if (
            val_loss
            <
            best_val_loss
        ):

            best_val_loss = (
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

            no_improve = 0

        else:

            no_improve += 1


        if (
            no_improve
            >=
            patience
        ):

            break


    if best_state is None:

        raise RuntimeError(
            "No valid model state"
        )


    # ========================================================
    # Restore best
    # ========================================================

    model.load_state_dict(
        best_state
    )


    return (
        model,
        best_val_loss,
        best_epoch
    )


# ============================================================
# Prediction
# ============================================================

def predict_one(
    model,
    test_X,
    device
):

    model.eval()


    x = torch.tensor(
        test_X,
        dtype=torch.float32,
        device=device
    )


    with torch.no_grad():

        pred = model(
            x
        )


    pred = (
        pred
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


    if len(pred) != 1:

        raise RuntimeError(
            "One-step OOS must "
            "produce exactly one value"
        )


    if not np.isfinite(
        pred[0]
    ):

        raise RuntimeError(
            "OOS prediction is NaN/Inf"
        )


    return float(
        pred[0]
    )
