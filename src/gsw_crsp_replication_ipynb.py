# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GSW (2007) Figure Replication with CRSP Treasury Data
#
# This notebook reproduces Figures 1, 2, and 6 from Gurkaynak, Sack, and Wright
# (2007) using per-security CRSP Treasury data from WRDS.
#
# **The CRSP data is an optional dependency of this pipeline.** The core pipeline
# needs no credentials, and this notebook always builds — but the figure
# reproductions below are skipped unless the CRSP data has been pulled. To enable
# them, copy `.env.example` to `.env`, set `WRDS_USERNAME` to your WRDS account,
# and rerun `doit` (see the "Optional: WRDS / CRSP Treasury data" section of the
# README).
#
# Figures 4 and 5 of the paper need only the published Fed data and are reproduced
# in the companion notebook `summary_fed_yield_curve_ipynb`.
#
# Reference:
#
# - Gurkaynak, Refet S., Brian Sack, and Jonathan H. Wright. "The US Treasury yield
#   curve: 1961 to the present." *Journal of Monetary Economics* 54.8 (2007): 2291-2304.

# %%
import warnings

import chartbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

import gsw_pricing as gp

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

warnings.filterwarnings("ignore")

# %%
# CRSP Treasury data is optional. When it is absent, this notebook still builds:
# each figure cell below prints instructions instead of failing.
CRSP_INFO_PATH = DATA_DIR / "CRSP_TFZ_INFO.parquet"
CRSP_DAILY_PATH = DATA_DIR / "CRSP_TFZ_DAILY.parquet"
HAS_CRSP = CRSP_INFO_PATH.exists() and CRSP_DAILY_PATH.exists()

MISSING_NOTE = (
    "CRSP Treasury data not found -- figure skipped.\n"
    "To enable this reproduction, set WRDS_USERNAME in .env (copy .env.example)\n"
    "and rerun `doit`. See the README section 'Optional: WRDS / CRSP Treasury data'."
)

df_all = pd.read_parquet(DATA_DIR / "fed_yield_curve_all.parquet")
if HAS_CRSP:
    df_info = pd.read_parquet(CRSP_INFO_PATH)
    df_daily = pd.read_parquet(CRSP_DAILY_PATH)
    print("CRSP Treasury data found -- reproducing GSW Figures 1, 2, and 6.")
else:
    print(MISSING_NOTE)

# %% [markdown]
# ## Figure 1

# %% [markdown]
# For reference, figure_1.png:
#
# ![Figure 1](../assets/figure_1.png)

# %% [markdown]
# We reproduce this figure from the CRSP Treasury issue table
# (`CRSP_TFZ_INFO.parquet`, pulled via `doit pull:crsp_treasury`). Each noncallable
# note/bond (`itype` 1 or 2) is drawn as a line from its issue date `tdatdt` (at its
# original maturity in years) down to its maturity date `tmatdt` (at 0). The diagonal
# streaks trace each security aging toward maturity; the stepped upper envelope is the
# longest maturity issued over time (7 → 15 → 20 → 30 years). The x-axis starts at
# 1961 to match the paper; our data extends to the present.

# %%
# Reproduce GSW Figure 1: outstanding Treasury securities (years to maturity over time)
if HAS_CRSP:
    info = df_info.dropna(subset=["tdatdt", "tmatdt"])
    issue = mdates.date2num(info["tdatdt"])
    mat = mdates.date2num(info["tmatdt"])
    ytm_at_issue = (info["tmatdt"] - info["tdatdt"]).dt.days / 365.25
    segments = [[(i, y), (m, 0)] for i, y, m in zip(issue, ytm_at_issue, mat)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(LineCollection(segments, colors="blue", linewidths=0.5))
    ax.xaxis_date()
    ax.set_xlim(
        mdates.date2num(pd.Timestamp("1961-01-01")),
        mdates.date2num(info["tdatdt"].max()),
    )
    ax.set_ylim(0, 35)
    ax.set_title("Outstanding Treasury Securities", fontsize=14)
    ax.set_ylabel("Years to Maturity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(MISSING_NOTE)

# %% [markdown]
# ## Figure 2

# %% [markdown]
# For reference, figure_2.png:
#
# ![Figure 2](../assets/figure_2.png)

# %% [markdown]
# We reproduce all three series from the paper on May 9, 2006:
#
# - **Par Yield** (line) — the fitted par curve from the published `SVENPY01`-`SVENPY30`.
# - **Predicted Yield** (×) — for each coupon security trading that day, the
#   bond-equivalent yield implied by pricing its cash flows off the Svensson curve
#   (`BETA`/`TAU` params).
# - **Actual Yield** (○) — the bond-equivalent yield of each security's market price
#   (mid of CRSP bid/ask plus accrued interest).
#
# The pricing uses `src/gsw_pricing.py`, with per-security data from
# `CRSP_TFZ_INFO.parquet` (coupon/maturity) and `CRSP_TFZ_DAILY.parquet` (quotes).

# %%
# Reproduce GSW Figure 2: par-yield curve with actual and predicted security yields
if HAS_CRSP:
    fig2_date = pd.Timestamp("2006-05-09")
    maturities = list(range(1, 31))
    par_yields = [df_all.loc[fig2_date, f"SVENPY{m:02d}"] for m in maturities]
    scatter = gp.actual_vs_predicted_yields(
        fig2_date, df_daily, df_info, df_all.loc[fig2_date]
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scatter["maturity_years"], scatter["actual_yield"],
               facecolors="none", edgecolors="b", s=45, label="Actual Yield")
    ax.scatter(scatter["maturity_years"], scatter["predicted_yield"],
               marker="x", c="b", s=30, label="Predicted Yield")
    ax.plot(maturities, par_yields, "b-", linewidth=1.5, label="Par Yield")
    ax.set_title(f"Par Yield Curve on {fig2_date:%B %d, %Y}", fontsize=14)
    ax.set_xlabel("Maturity in Years")
    ax.set_ylabel("Yield")
    ax.set_xlim(0, 30)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(MISSING_NOTE)

# %% [markdown]
# ## Figure 6

# %% [markdown]
# For reference, figure_6.png:
#
# ![Figure 6](../assets/figure_6.png)

# %% [markdown]
# Same construction as Figure 2, on May 24, 2005 (par line + per-security actual and
# predicted yields). In the paper this date illustrates a cheapest-to-deliver premium:
# securities near ~7 years trade rich (actual yield sitting below the fitted curve).
# We plot the actual/predicted scatter; we do not single out the specific Aug-2012
# note, which would require matching its CUSIP.

# %%
# Reproduce GSW Figure 6: par-yield curve with actual and predicted security yields
if HAS_CRSP:
    fig6_date = pd.Timestamp("2005-05-24")
    maturities = list(range(1, 31))
    par_yields = [df_all.loc[fig6_date, f"SVENPY{m:02d}"] for m in maturities]
    scatter = gp.actual_vs_predicted_yields(
        fig6_date, df_daily, df_info, df_all.loc[fig6_date]
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scatter["maturity_years"], scatter["actual_yield"],
               facecolors="none", edgecolors="b", s=45, label="Actual Yield")
    ax.scatter(scatter["maturity_years"], scatter["predicted_yield"],
               marker="x", c="b", s=30, label="Predicted Yield")
    ax.plot(maturities, par_yields, "b-", linewidth=1.5, label="Par Yield")
    ax.set_title(f"Par Yield Curve on {fig6_date:%B %d, %Y}", fontsize=14)
    ax.set_xlabel("Maturity in Years")
    ax.set_ylabel("Yield")
    ax.set_xlim(0, 30)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(MISSING_NOTE)
