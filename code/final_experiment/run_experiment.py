# -*- coding: utf-8 -*-
"""
Rolling-origin runner.

The defaults reproduce the reported baseline exactly. Three optional arguments
were added for the replication experiments; when they are omitted the behaviour
is identical to the original runner.

Baseline (unchanged):
    python code/final_experiment/run_experiment.py --asset CSI300 --model LSTM --mode full

Seed replication (results go to results/<tag>/):
    python code/final_experiment/run_experiment.py --asset CSI300 --model LSTM --mode full --seed 7 --tag seed7

Tuning-budget sensitivity:
    python code/final_experiment/run_experiment.py --asset CSI300 --model Transformer --mode full --budget 60 --tag budget60

Smoke test (five origins only):
    python code/final_experiment/run_experiment.py --asset CSI300 --model LSTM --mode smoke --tag smokecheck
"""

import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    ASSETS,
    BATCH_SIZE,
    MAX_EPOCHS,
    OPTUNA_EPOCHS,
    OPTUNA_PATIENCE,
    OPTUNA_TRIALS,
    PATIENCE,
    PROJECT_ROOT,
    RESULT_DIR,
    SEED,
    SEQ_LEN,
    STRIDE,
    TUNE_EVERY,
    VAL_SIZE,
    WINDOW,
)
from core import (
    get_device,
    load_asset,
    make_split,
    predict_original_scale,
    scale_split,
    set_seed,
    train_model,
)
from model_factory import build_model

MODEL_NAMES = ["LSTM", "Transformer", "Autoformer", "PatchTST", "iTransformer"]


# ============================================================
# Search space
# ============================================================

def suggest_params(trial, model_name):
    """Same search space and same suggestion order as the original runner."""

    if model_name == "LSTM":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        }

    params = {
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
        "nhead": trial.suggest_categorical("nhead", [1, 2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "d_ff": trial.suggest_categorical("d_ff", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    }

    if model_name == "Autoformer":
        params["moving_avg"] = trial.suggest_categorical("moving_avg", [3, 5, 7, 13])

    if model_name == "PatchTST":
        params["patch_len"] = trial.suggest_categorical("patch_len", [5, 10, 12, 15, 20])
        params["stride"] = trial.suggest_categorical("stride", [2, 5, 6, 10])

    return params


# ============================================================
# Hyperparameter search
# ============================================================

def tune_parameters(scaled, tune_seed, model_name, device, n_trials):
    """Run one Optuna search and return the best configuration."""

    def objective(trial):
        params = suggest_params(trial, model_name)
        set_seed(tune_seed)
        model = build_model(model_name, params)
        model, best_loss, _ = train_model(
            model=model,
            train_X=scaled["train_X"],
            train_y=scaled["train_y"],
            val_X=scaled["val_X"],
            val_y=scaled["val_y"],
            lr=params["lr"],
            epochs=OPTUNA_EPOCHS,
            batch_size=BATCH_SIZE,
            patience=OPTUNA_PATIENCE,
            device=device,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_loss

    sampler = optuna.samplers.TPESampler(seed=tune_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=list(ASSETS.keys()))
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--mode", default="full", choices=["smoke", "full"])
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed used for initialisation and for the Optuna sampler")
    parser.add_argument("--budget", type=int, default=OPTUNA_TRIALS,
                        help="number of Optuna trials per search")
    parser.add_argument("--tag", default="",
                        help="run group name; empty writes to results/final (baseline)")
    args = parser.parse_args()

    asset = args.asset
    model_name = args.model
    mode = args.mode
    run_seed = args.seed
    trial_budget = args.budget

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    base_dir = RESULT_DIR if not args.tag else os.path.join(PROJECT_ROOT, "results", args.tag)
    output_dir = os.path.join(base_dir, mode, asset)
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, f"{model_name}_predictions.csv")
    tune_file = os.path.join(output_dir, f"{model_name}_tuning.csv")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = get_device()

    print("=" * 80)
    print(f"Asset      = {asset}")
    print(f"Model      = {model_name}")
    print(f"Mode       = {mode}")
    print(f"Seed       = {run_seed}")
    print(f"Budget     = {trial_budget}")
    print(f"Tag        = {args.tag or '(baseline)'}")
    print(f"Device     = {device}")
    print(f"Output     = {output_dir}")
    print("=" * 80)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    returns, dates = load_asset(asset)

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------
    completed_dates = set()
    if os.path.isfile(pred_file):
        existing = pd.read_csv(pred_file)
        completed_dates = set(
            pd.to_datetime(existing["Date"]).dt.strftime("%Y-%m-%d")
        )

    current_params = None
    if os.path.isfile(tune_file):
        tune_df = pd.read_csv(tune_file)
        if len(tune_df) > 0:
            current_params = json.loads(tune_df.iloc[-1]["BestParams"])

    # --------------------------------------------------------
    # Forecast origins
    # --------------------------------------------------------
    all_origins = list(range(WINDOW, len(returns), STRIDE))
    if mode == "smoke":
        all_origins = all_origins[:5]

    n_trials = 2 if mode == "smoke" else trial_budget

    # --------------------------------------------------------
    # Rolling loop
    # --------------------------------------------------------
    for origin in tqdm(all_origins):
        date_string = dates[origin].strftime("%Y-%m-%d")

        if date_string in completed_dates:
            continue

        split = make_split(returns, dates, origin, WINDOW, VAL_SIZE, SEQ_LEN)
        scaled = scale_split(split)

        relative_origin = origin - WINDOW
        need_tune = (current_params is None) or (relative_origin % TUNE_EVERY == 0)

        if need_tune:
            print(f"\nTUNING {asset} {model_name} {date_string}")
            tuning_seed = run_seed + relative_origin
            current_params, best_search_loss = tune_parameters(
                scaled, tuning_seed, model_name, device, n_trials
            )
            tune_record = pd.DataFrame([{
                "Date": date_string,
                "Asset": asset,
                "Model": model_name,
                "Seed": run_seed,
                "TrialBudget": n_trials,
                "BestValidationLoss": best_search_loss,
                "BestParams": json.dumps(current_params),
            }])
            tune_record.to_csv(
                tune_file,
                mode="a",
                index=False,
                header=(not os.path.isfile(tune_file)),
            )

        # ----------------------------------------------------
        # Daily refit and forecast
        # ----------------------------------------------------
        set_seed(run_seed)
        model = build_model(model_name, current_params)

        start_time = time.time()
        model, val_loss, best_epoch = train_model(
            model=model,
            train_X=scaled["train_X"],
            train_y=scaled["train_y"],
            val_X=scaled["val_X"],
            val_y=scaled["val_y"],
            lr=current_params["lr"],
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            device=device,
        )
        prediction = predict_original_scale(
            model=model,
            test_X=scaled["test_X"],
            y_scaler=scaled["y_scaler"],
            device=device,
        )
        elapsed = time.time() - start_time

        record = pd.DataFrame([{
            "Date": date_string,
            "Asset": asset,
            "Model": model_name,
            "WindowStart": split["window_start_date"],
            "LastKnownDate": split["last_known_date"],
            "True": split["true"],
            "HM": split["HM"],
            "PER": split["PER"],
            "Prediction": prediction,
            "BestValLoss": val_loss,
            "BestEpoch": best_epoch,
            "TrainingSeconds": elapsed,
            "Seed": run_seed,
            "TrialBudget": n_trials,
            "HyperParameters": json.dumps(current_params),
        }])
        record.to_csv(
            pred_file,
            mode="a",
            index=False,
            header=(not os.path.isfile(pred_file)),
        )

        completed_dates.add(date_string)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print(f"{asset} - {model_name} - seed {run_seed} - budget {n_trials}")
    print(f"Saved to:\n{pred_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
