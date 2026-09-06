# -*- coding: utf-8 -*-

import os


# ============================================================
#

#
# The project root is resolved automatically
# Example:

# ============================================================

PROJECT_ROOT = os.environ.get("STOCK_EMH_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

OFFICIAL_DIR = os.path.join(
    PROJECT_ROOT,
    "official_models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "final"
)


# ============================================================
# ============================================================

ASSETS = {

    "CSI300":
        "CSI300_daily_data.csv",

    "SP500":
        "SP500_daily_data.csv",

    "CSI1000":
        "CSI1000_daily_data.csv",

    "BTCUSD":
        "BTC_USD_daily_data.csv"
}


# ============================================================
# ============================================================

WINDOW = 504

VAL_SIZE = 126

SEQ_LEN = 60

HORIZON = 1

STRIDE = 1


# ============================================================
# Training
# ============================================================

MAX_EPOCHS = 50

BATCH_SIZE = 64

PATIENCE = 10


# ============================================================
# Optuna
#
# ============================================================

TUNE_EVERY = 63

OPTUNA_TRIALS = 20

OPTUNA_EPOCHS = 30

OPTUNA_PATIENCE = 5


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
