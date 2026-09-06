# -*- coding: utf-8 -*-

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import optuna

from tqdm import tqdm

from config import *

from core import (
    set_seed,
    get_device,
    load_asset,
    make_split,
    scale_split,
    train_model,
    predict_original_scale
)

from model_factory import (
    build_model
)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--asset",
    required=True,
    choices=list(
        ASSETS.keys()
    )
)


parser.add_argument(
    "--model",
    required=True,
    choices=[
        "LSTM",
        "Transformer",
        "Autoformer",
        "PatchTST",
        "iTransformer"
    ]
)


parser.add_argument(
    "--mode",
    default="full",
    choices=[
        "smoke",
        "full"
    ]
)


args = parser.parse_args()


ASSET = args.asset

MODEL_NAME = args.model

MODE = args.mode


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.path.join(
    RESULT_DIR,
    MODE,
    ASSET
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


PRED_FILE = os.path.join(

    OUTPUT_DIR,

    f"{MODEL_NAME}_predictions.csv"
)


TUNE_FILE = os.path.join(

    OUTPUT_DIR,

    f"{MODEL_NAME}_tuning.csv"
)


# ============================================================
# Device
# ============================================================

device = get_device()


print("=" * 80)

print(
    f"Asset = {ASSET}"
)

print(
    f"Model = {MODEL_NAME}"
)

print(
    f"Mode = {MODE}"
)

print(
    f"Device = {device}"
)

print("=" * 80)


# ============================================================
# Load data
# ============================================================

returns, dates = load_asset(
    ASSET
)


# ============================================================
# Search space
# ============================================================

def suggest_params(
    trial
):

    if MODEL_NAME == "LSTM":

        return {

            "hidden_size":
                trial.suggest_categorical(
                    "hidden_size",
                    [16, 32, 64]
                ),

            "num_layers":
                trial.suggest_int(
                    "num_layers",
                    1,
                    3
                ),

            "dropout":
                trial.suggest_float(
                    "dropout",
                    0.0,
                    0.3
                ),

            "lr":
                trial.suggest_float(
                    "lr",
                    1e-5,
                    1e-3,
                    log=True
                )
        }


    params = {

        "d_model":
            trial.suggest_categorical(
                "d_model",
                [16, 32, 64]
            ),

        "nhead":
            trial.suggest_categorical(
                "nhead",
                [1, 2, 4]
            ),

        "num_layers":
            trial.suggest_int(
                "num_layers",
                1,
                3
            ),

        "d_ff":
            trial.suggest_categorical(
                "d_ff",
                [32, 64, 128, 256]
            ),

        "dropout":
            trial.suggest_float(
                "dropout",
                0.0,
                0.3
            ),

        "lr":
            trial.suggest_float(
                "lr",
                1e-5,
                1e-3,
                log=True
            )
    }


    if MODEL_NAME == "Autoformer":

        params[
            "moving_avg"
        ] = trial.suggest_categorical(

            "moving_avg",

            [3, 5, 7, 13]
        )


    if MODEL_NAME == "PatchTST":

        params[
            "patch_len"
        ] = trial.suggest_categorical(

            "patch_len",

            [5, 10, 12, 15, 20]
        )

        params[
            "stride"
        ] = trial.suggest_categorical(

            "stride",

            [2, 5, 6, 10]
        )


    return params


# ============================================================
# Tune
# ============================================================

def tune_parameters(
    scaled,
    tune_seed
):

    def objective(
        trial
    ):

        params = suggest_params(
            trial
        )


        set_seed(
            tune_seed
        )


        model = build_model(
            MODEL_NAME,
            params
        )


        model, best_loss, _ = (
            train_model(

                model=model,

                train_X=scaled[
                    "train_X"
                ],

                train_y=scaled[
                    "train_y"
                ],

                val_X=scaled[
                    "val_X"
                ],

                val_y=scaled[
                    "val_y"
                ],

                lr=params[
                    "lr"
                ],

                epochs=OPTUNA_EPOCHS,

                batch_size=BATCH_SIZE,

                patience=OPTUNA_PATIENCE,

                device=device
            )
        )


        del model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        return best_loss


    sampler = (
        optuna.samplers.TPESampler(
            seed=tune_seed
        )
    )


    study = optuna.create_study(

        direction="minimize",

        sampler=sampler
    )


    n_trials = (

        2
        if MODE == "smoke"
        else OPTUNA_TRIALS
    )


    study.optimize(

        objective,

        n_trials=n_trials,

        show_progress_bar=False
    )


    return (
        study.best_params,
        float(
            study.best_value
        )
    )


# ============================================================
# Resume
# ============================================================

existing = None


if os.path.isfile(
    PRED_FILE
):

    existing = pd.read_csv(
        PRED_FILE
    )


completed_dates = set()


if existing is not None:

    completed_dates = set(

        pd.to_datetime(
            existing["Date"]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


# ============================================================
# Recover latest hyperparameters
# ============================================================

current_params = None


if os.path.isfile(
    TUNE_FILE
):

    tune_df = pd.read_csv(
        TUNE_FILE
    )

    if len(tune_df) > 0:

        current_params = json.loads(

            tune_df.iloc[-1][
                "BestParams"
            ]
        )


# ============================================================
# Origins
# ============================================================

all_origins = list(

    range(
        WINDOW,
        len(returns),
        STRIDE
    )
)


if MODE == "smoke":

    all_origins = (
        all_origins[:5]
    )


# ============================================================
# Main rolling
# ============================================================

for index, origin in enumerate(

    tqdm(
        all_origins
    )
):

    date_string = (
        dates[origin]
        .strftime(
            "%Y-%m-%d"
        )
    )


    if (
        date_string
        in
        completed_dates
    ):

        continue


    # ========================================================
    # Split
    # ========================================================

    split = make_split(

        returns,

        dates,

        origin,

        WINDOW,

        VAL_SIZE,

        SEQ_LEN
    )


    scaled = scale_split(
        split
    )


    # ========================================================
    # Tune every 63 observations
    # ========================================================

    relative_origin = (
        origin
        -
        WINDOW
    )


    need_tune = (

        current_params
        is None

        or

        (
            relative_origin
            %
            TUNE_EVERY
            ==
            0
        )
    )


    if need_tune:

        print(
            f"\nTUNING "
            f"{ASSET} "
            f"{MODEL_NAME} "
            f"{date_string}"
        )


        tuning_seed = (

            SEED

            +
            relative_origin
        )


        current_params, best_search_loss = (
            tune_parameters(

                scaled,

                tuning_seed
            )
        )


        tune_record = pd.DataFrame([{

            "Date":
                date_string,

            "Asset":
                ASSET,

            "Model":
                MODEL_NAME,

            "BestValidationLoss":
                best_search_loss,

            "BestParams":
                json.dumps(
                    current_params
                )
        }])


        tune_record.to_csv(

            TUNE_FILE,

            mode="a",

            index=False,

            header=(
                not os.path.exists(
                    TUNE_FILE
                )
            )
        )


    # ========================================================
    # Daily refit
    # ========================================================

    set_seed(
        SEED
    )


    model = build_model(

        MODEL_NAME,

        current_params
    )


    start_time = (
        time.time()
    )


    model, val_loss, best_epoch = (

        train_model(

            model=model,

            train_X=scaled[
                "train_X"
            ],

            train_y=scaled[
                "train_y"
            ],

            val_X=scaled[
                "val_X"
            ],

            val_y=scaled[
                "val_y"
            ],

            lr=current_params[
                "lr"
            ],

            epochs=MAX_EPOCHS,

            batch_size=BATCH_SIZE,

            patience=PATIENCE,

            device=device
        )
    )


    # ========================================================
    # OOS prediction
    # ========================================================

    prediction = (
        predict_original_scale(

            model=model,

            test_X=scaled[
                "test_X"
            ],

            y_scaler=scaled[
                "y_scaler"
            ],

            device=device
        )
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # Save immediately
    # ========================================================

    record = pd.DataFrame([{

        "Date":
            date_string,

        "Asset":
            ASSET,

        "Model":
            MODEL_NAME,

        "WindowStart":
            split[
                "window_start_date"
            ],

        "LastKnownDate":
            split[
                "last_known_date"
            ],

        "True":
            split[
                "true"
            ],

        "HM":
            split[
                "HM"
            ],

        "PER":
            split[
                "PER"
            ],

        "Prediction":
            prediction,

        "BestValLoss":
            val_loss,

        "BestEpoch":
            best_epoch,

        "TrainingSeconds":
            elapsed,

        "HyperParameters":
            json.dumps(
                current_params
            )
    }])


    record.to_csv(

        PRED_FILE,

        mode="a",

        index=False,

        header=(
            not os.path.exists(
                PRED_FILE
            )
        )
    )


    completed_dates.add(
        date_string
    )


    del model


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


print("\n" + "=" * 80)

print(
    "EXPERIMENT FINISHED"
)

print(
    f"{ASSET} - {MODEL_NAME}"
)

print(
    f"Saved to:\n{PRED_FILE}"
)

print("=" * 80)
