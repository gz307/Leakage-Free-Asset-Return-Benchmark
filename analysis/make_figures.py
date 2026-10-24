# -*- coding: utf-8 -*-
"""
make_figures_print.py -- Figures 1-6 at final print size (v3, larger type).

Design rules
------------
* Canvas is 6.5 inches wide (= 16.5 cm, the text width) and is inserted at that
  width, so the point sizes below are approximately the point sizes on the page.
  (bbox_inches="tight" trims a little margin, which scales everything up by
  roughly 10 per cent, so the printed sizes are if anything slightly larger.)
* No title inside the image: the manuscript caption already names the figure.
* Panel (c) of Figures 1-4 omits the persistence benchmark, whose R2OS lies
  between -0.88 and -1.23 and would compress every other bar to a sliver.
* Two type scales: dense bar panels use 7.5-8.5 pt because seven categories have
  to share 3 inches of width; the line panels of Figure 6 use 8.5-10 pt.

Inputs (archived results, nothing is retrained)
-----------------------------------------------
    results/final/analysis/TABLE4_MAIN_RESULTS.csv
    results/final/analysis/{ASSET}_ROLLING_ROS2_252.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.environ.get(
    "ANALYSIS_DIR", r"D:\stock_emh_project\results\final\analysis")
OUTDIR = os.path.join(HERE, "figs")
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ["LSTM", "Transformer", "Autoformer", "PatchTST", "iTransformer"]
COLORS = {
    "HM": "#8c8c8c",
    "PER": "#c9c9c9",
    "LSTM": "#1f77b4",
    "Transformer": "#ff7f0e",
    "Autoformer": "#2ca02c",
    "PatchTST": "#d62728",
    "iTransformer": "#9467bd",
}
ASSET_LABEL = {"CSI300": "CSI 300", "SP500": "S&P 500",
               "CSI1000": "CSI 1000", "BTCUSD": "BTC/USD"}
ASSETS = ["CSI300", "SP500", "CSI1000", "BTCUSD"]

STYLES = {
    "bars": dict(base=8.5, title=9.5, label=8.5, xtick=7.5, ytick=7.8,
                 legend=8.0, annot=7.2),
    "lines": dict(base=9.5, title=10.0, label=9.5, xtick=8.5, ytick=8.8,
                  legend=9.5, annot=8.5),
}

df = pd.read_csv(os.path.join(ANALYSIS, "TABLE4_MAIN_RESULTS.csv"))


def apply_style(name):
    s = STYLES[name]
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": s["base"],
        "axes.titlesize": s["title"],
        "axes.labelsize": s["label"],
        "xtick.labelsize": s["xtick"],
        "ytick.labelsize": s["ytick"],
        "legend.fontsize": s["legend"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "savefig.dpi": 400,
        "figure.dpi": 120,
    })


def save(fig, name):
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=400)
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Store a dpi that makes the image exactly 16.5 cm wide, so that Word
    # inserts it at the text width without any rescaling (no surprise shrinking).
    from PIL import Image
    im = Image.open(png)
    w_px = im.size[0]
    target_inch = 16.5 / 2.54
    im.save(png, dpi=(w_px / target_inch, w_px / target_inch))
    print(f"saved: {name}  ({w_px}px -> {16.5:.1f} cm)")


def style_axes(ax):
    ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bar_panel(ax, names, values, title, ylabel, fmt, pad=0.16, rotate=45):
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, width=0.66,
                  edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_title(title, pad=3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=rotate, ha="right",
                       rotation_mode="anchor")
    style_axes(ax)
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - 0.12 * span, hi + pad * span)
    lo, hi = ax.get_ylim()
    span = hi - lo
    for b, v in zip(bars, values):
        if v >= 0:
            y, va = v + 0.015 * span, "bottom"
        else:
            y, va = v - 0.015 * span, "top"
        ax.text(b.get_x() + b.get_width() / 2, y, format(v, fmt),
                ha="center", va=va, fontsize=STYLES["bars"]["annot"], zorder=4)


# ============================================================
# Figures 1-4
# ============================================================
apply_style("bars")

PANELS = [("MSE", 1e4, r"MSE ($\times 10^{-4}$)", "(a) Mean squared error", ".2f"),
          ("MAE", 1e3, r"MAE ($\times 10^{-3}$)", "(b) Mean absolute error", ".2f"),
          ("R_OS2", 1.0, r"$R^2_{OS}$", "(c) Out-of-sample $R^2_{OS}$ vs HM", ".3f"),
          ("DA", 100.0, "DA (%)", "(d) Directional accuracy", ".1f")]

for asset, name in [("CSI300", "Figure_1_CSI300"), ("SP500", "Figure_2_SP500"),
                    ("CSI1000", "Figure_3_CSI1000"), ("BTCUSD", "Figure_4_BTCUSD")]:
    sub = df[df["Asset"] == asset].set_index("Model")
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.9))
    fig.subplots_adjust(wspace=0.32, hspace=0.95)

    for ax, (col, scale, ylabel, title, fmt) in zip(axes.flat, PANELS):
        names = ["HM"] + MODELS if col == "R_OS2" else ["HM", "PER"] + MODELS
        values = np.array([sub.loc[n, col] for n in names]) * scale
        bar_panel(ax, names, values, title, ylabel, fmt)

    save(fig, name)

# ============================================================
# Figure 5: cross-asset comparison
# ============================================================
apply_style("bars")

fig, ax = plt.subplots(figsize=(6.5, 3.1))
width = 0.19
asset_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for k, asset in enumerate(ASSETS):
    sub = df[df["Asset"] == asset].set_index("Model")
    vals = [sub.loc[m, "R_OS2"] for m in MODELS]
    xpos = np.arange(len(MODELS)) + (k - 1.5) * width
    bars = ax.bar(xpos, vals, width=width, label=ASSET_LABEL[asset],
                  color=asset_colors[k], edgecolor="black", linewidth=0.35,
                  zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.006, f"{v:.3f}",
                ha="center", va="top", fontsize=7.0, rotation=90, zorder=4)

ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)
ax.set_xticks(np.arange(len(MODELS)))
ax.set_xticklabels(MODELS)
ax.set_ylabel(r"$R^2_{OS}$ (vs HM)")
style_axes(ax)
lo, hi = ax.get_ylim()
ax.set_ylim(lo * 1.16, max(hi, 0) * 1.0 + 0.010)
ax.legend(ncol=4, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, 1.13), handlelength=1.1, columnspacing=1.6)
save(fig, "Figure_5_CrossAsset")

# ============================================================
# Figure 6: rolling R2OS, 252-observation windows
# ============================================================
apply_style("lines")

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
fig.subplots_adjust(wspace=0.22, hspace=0.34, bottom=0.15)

for ax, asset in zip(axes.flat, ASSETS):
    d = pd.read_csv(os.path.join(ANALYSIS, f"{asset}_ROLLING_ROS2_252.csv"),
                    parse_dates=["Date"])
    for m in MODELS:
        ax.plot(d["Date"], d[m], linewidth=0.8, color=COLORS[m], label=m)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_title(ASSET_LABEL[asset], pad=3)
    ax.set_ylabel(r"Rolling $R^2_{OS}$")
    ax.grid(color="0.90", linewidth=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=STYLES["lines"]["xtick"])
    handles, labels_ = ax.get_legend_handles_labels()

fig.legend(handles, labels_, ncol=5, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=1.4)
save(fig, "Figure_6_Rolling")

print("\nall figures written to", OUTDIR)
