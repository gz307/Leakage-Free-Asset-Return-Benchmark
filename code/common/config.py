# -*- coding: utf-8 -*-

import os


# ============================================================
#
# ============================================================

PROJECT_ROOT = os.environ.get("STOCK_EMH_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ============================================================
# Directories
# ============================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

OFFICIAL_MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "official_models"
)


# ============================================================
#
# ============================================================

ASSETS = {

    "CSI300": {
        "file":
            "CSI300_daily_data.csv"
    },

    "SP500": {
        "file":
            "SP500_daily_data.csv"
    },

    "CSI1000": {
        "file":
            "CSI1000_daily_data.csv"
    },

    "BTCUSD": {
        "file":
            "BTC_USD_daily_data.csv"
    }
}


# ============================================================
# ============================================================

WINDOW = 504

VAL_SIZE = 126

SEQ_LEN = 60

HORIZON = 1

STRIDE = 1


# ============================================================
# ============================================================

MAX_EPOCHS = 50

BATCH_SIZE = 64

PATIENCE = 10

SEED = 42
