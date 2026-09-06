# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse

import pandas as pd
import torch
import optuna
from tqdm import tqdm


PROJECT_ROOT = os.environ.get("STOCK_EMH_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


FINAL_CODE_DIR = os.path.join(
    PROJECT_ROOT,
    "code",
    "final_experiment"
)

sys.path.insert(
    0,
    FINAL_CODE_DIR
)

from core import (
    set_seed,
    get_device,
    load_asset,
    make_split,
    scale_split,
    train_model,
    predict_original_scale
)

from model_factory import build_model


# ============================================================
# Fixed protocol
# ============================================================

WINDOW = 504
VAL_SIZE = 126
SEQ_LEN = 60

MAX_EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10

OPTUNA_TRIALS = 20
OPTUNA_EPOCHS = 30
OPTUNA_PATIENCE = 5

SEED = 42

COMMON_OOS_START = 504


ASSETS = [
    "CSI300",
    "SP500",
    "CSI1000",
    "BTCUSD"
]

RETUNE_VALUES = [
    21,
    126
]


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--asset",
    required=True,
    choices=ASSETS
)

parser.add_argument(
    "--tune_every",
    required=True,
    type=int,
    choices=RETUNE_VALUES
)

parser.add_argument(
    "--mode",
    default="full",
    choices=["smoke", "full"]
)

args = parser.parse_args()

ASSET = args.asset
TUNE_EVERY = args.tune_every
MODE = args.mode

MODEL_NAME = "LSTM"


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "robustness",
    "retune",
    f"R{TUNE_EVERY}",
    MODE,
    ASSET
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

PRED_FILE = os.path.join(
    OUTPUT_DIR,
    "LSTM_predictions.csv"
)

TUNE_FILE = os.path.join(
    OUTPUT_DIR,
    "LSTM_tuning.csv"
)


# ============================================================
# Hard safety
# ============================================================

final_full = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "results",
        "final",
        "full"
    )
)

if os.path.abspath(
    OUTPUT_DIR
).startswith(
    final_full
):

    raise RuntimeError(
        "SAFETY ERROR: final/full write blocked."
    )


# ============================================================
# LSTM search
# ============================================================

def suggest_params(
    trial
):

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


device = get_device()


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
            "LSTM",
            params
        )

        model, val_loss, _ = train_model(
            model=model,

            train_X=scaled["train_X"],
            train_y=scaled["train_y"],

            val_X=scaled["val_X"],
            val_y=scaled["val_y"],

            lr=params["lr"],

            epochs=(
                5
                if MODE == "smoke"
                else OPTUNA_EPOCHS
            ),

            batch_size=BATCH_SIZE,

            patience=(
                2
                if MODE == "smoke"
                else OPTUNA_PATIENCE
            ),

            device=device
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(
            val_loss
        )


    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=tune_seed
        )
    )


    study.optimize(
        objective,
        n_trials=(
            2
            if MODE == "smoke"
            else OPTUNA_TRIALS
        ),
        show_progress_bar=False
    )


    return (
        study.best_params,
        float(study.best_value)
    )


# ============================================================
# Start
# ============================================================

print("=" * 80)
print("RETUNING-FREQUENCY ROBUSTNESS")
print("=" * 80)
print("Asset:", ASSET)
print("Model: LSTM")
print("Window:", WINDOW)
print("SeqLen:", SEQ_LEN)
print("TuneEvery:", TUNE_EVERY)
print("Mode:", MODE)
print("Device:", device)
print("Output:", OUTPUT_DIR)
print("=" * 80)


returns, dates = load_asset(
    ASSET
)


# ============================================================
# Resume
# ============================================================

completed_dates = set()

if os.path.isfile(
    PRED_FILE
):

    old = pd.read_csv(
        PRED_FILE
    )

    completed_dates = set(
        pd.to_datetime(
            old["Date"]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )


current_params = None

if os.path.isfile(
    TUNE_FILE
):

    tuning = pd.read_csv(
        TUNE_FILE
    )

    if len(tuning) > 0:

        current_params = json.loads(
            tuning.iloc[-1][
                "BestParams"
            ]
        )


origins = list(
    range(
        COMMON_OOS_START,
        len(returns)
    )
)

if MODE == "smoke":
    origins = origins[:5]


# ============================================================
# Rolling
# ============================================================

for origin in tqdm(origins):

    date_string = (
        dates[origin]
        .strftime("%Y-%m-%d")
    )

    if date_string in completed_dates:
        continue


    split = make_split(
        returns=returns,
        dates=dates,
        origin=origin,
        window=WINDOW,
        val_size=VAL_SIZE,
        seq_len=SEQ_LEN
    )


    scaled = scale_split(
        split
    )


    relative_origin = (
        origin
        -
        COMMON_OOS_START
    )


    need_tune = (
        current_params is None
        or
        relative_origin
        %
        TUNE_EVERY
        ==
        0
    )


    if need_tune:

        tune_seed = (
            SEED
            +
            relative_origin
        )

        print(
            f"\nTUNING "
            f"{ASSET} LSTM "
            f"R={TUNE_EVERY} "
            f"{date_string}"
        )


        current_params, search_loss = (
            tune_parameters(
                scaled,
                tune_seed
            )
        )


        tuning_row = pd.DataFrame([{

            "Date":
                date_string,

            "Asset":
                ASSET,

            "Model":
                "LSTM",

            "Window":
                WINDOW,

            "SeqLen":
                SEQ_LEN,

            "TuneEvery":
                TUNE_EVERY,

            "BestValidationLoss":
                search_loss,

            "BestParams":
                json.dumps(
                    current_params
                )
        }])


        tuning_row.to_csv(
            TUNE_FILE,
            mode="a",
            index=False,
            header=(
                not os.path.exists(
                    TUNE_FILE
                )
            )
        )


    set_seed(
        SEED
    )


    model = build_model(
        "LSTM",
        current_params
    )


    start = time.time()


    model, val_loss, best_epoch = train_model(
        model=model,

        train_X=scaled["train_X"],
        train_y=scaled["train_y"],

        val_X=scaled["val_X"],
        val_y=scaled["val_y"],

        lr=current_params["lr"],

        epochs=(
            5
            if MODE == "smoke"
            else MAX_EPOCHS
        ),

        batch_size=BATCH_SIZE,

        patience=(
            2
            if MODE == "smoke"
            else PATIENCE
        ),

        device=device
    )


    prediction = predict_original_scale(
        model=model,
        test_X=scaled["test_X"],
        y_scaler=scaled["y_scaler"],
        device=device
    )


    elapsed = (
        time.time()
        -
        start
    )


    result_row = pd.DataFrame([{

        "Date":
            date_string,

        "Asset":
            ASSET,

        "Model":
            "LSTM",

        "Experiment":
            "RetuneRobustness",

        "Window":
            WINDOW,

        "SeqLen":
            SEQ_LEN,

        "TuneEvery":
            TUNE_EVERY,

        "WindowStart":
            split["window_start_date"],

        "LastKnownDate":
            split["last_known_date"],

        "True":
            split["true"],

        "HM":
            split["HM"],

        "PER":
            split["PER"],

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


    result_row.to_csv(
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
print("RETUNING ROBUSTNESS FINISHED")
print(ASSET, f"R={TUNE_EVERY}")
print("Saved:", PRED_FILE)
print("=" * 80)
