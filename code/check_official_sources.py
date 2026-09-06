# -*- coding: utf-8 -*-

import os


PROJECT_ROOT = os.environ.get("STOCK_EMH_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


SOURCES = {

    "Autoformer": os.path.join(
        PROJECT_ROOT,
        "official_models",
        "Autoformer",
        "models",
        "Autoformer.py"
    ),

    "PatchTST": os.path.join(
        PROJECT_ROOT,
        "official_models",
        "PatchTST",
        "PatchTST_supervised",
        "models",
        "PatchTST.py"
    ),

    "iTransformer": os.path.join(
        PROJECT_ROOT,
        "official_models",
        "iTransformer",
        "model",
        "iTransformer.py"
    )
}


print("=" * 70)
print("OFFICIAL MODEL SOURCE CHECK")
print("=" * 70)


all_ok = True


for name, path in SOURCES.items():

    exists = os.path.isfile(
        path
    )

    print("\nModel:", name)

    print(
        "Source:",
        path
    )

    print(
        "Exists:",
        exists
    )

    if not exists:
        all_ok = False


print("\n" + "=" * 70)


if all_ok:

    print(
        "ALL OFFICIAL MODEL SOURCES FOUND"
    )

else:

    raise RuntimeError(
        "One or more official model source files are missing"
    )


print("=" * 70)
