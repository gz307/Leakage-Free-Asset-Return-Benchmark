# -*- coding: utf-8 -*-

import sys
import os

print("=" * 60)
print("Python environment check")
print("=" * 60)

print("Python:")
print(sys.version)

print("\nPython executable:")
print(sys.executable)

print("\nWorking directory:")
print(os.getcwd())


print("\n" + "=" * 60)
print("NumPy")
print("=" * 60)

import numpy as np

print("NumPy version:", np.__version__)


print("\n" + "=" * 60)
print("Pandas")
print("=" * 60)

import pandas as pd

print("Pandas version:", pd.__version__)


print("\n" + "=" * 60)
print("Scikit-learn")
print("=" * 60)

import sklearn

print(
    "Scikit-learn version:",
    sklearn.__version__
)


print("\n" + "=" * 60)
print("SciPy")
print("=" * 60)

import scipy

print("SciPy version:", scipy.__version__)


print("\n" + "=" * 60)
print("PyTorch")
print("=" * 60)

import torch

print("PyTorch version:", torch.__version__)
print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("\n" + "=" * 60)
print("Tensor test")
print("=" * 60)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

x = torch.randn(
    8,
    60,
    1
).to(device)

print("Input shape:", x.shape)
print("Device:", x.device)
print("Finite:", torch.isfinite(x).all().item())


print("\n" + "=" * 60)
print("Basic environment test passed")
print("=" * 60)
