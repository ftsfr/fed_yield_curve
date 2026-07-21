# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: finm-32900-venv-p31211 (3.12.11.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Federal Reserve Treasury Yield Curve

# %%
import warnings

import chartbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Data Overview
#
# This pipeline estimates the Federal Reserve's Treasury yield curve with the
# Gurkaynak, Sack, and Wright (2007) / Svensson model and writes three parquet
# files to `_data/`, each documented below:
# - `fed_yield_curve_all.parquet` — the full fitted output (parameters, zero-coupon,
#   par, and forward series)
# - `fed_yield_curve.parquet` — the 1-30 year zero-coupon yields in wide format
# - `ftsfr_treas_yield_curve_zero_coupon.parquet` — the zero-coupon yields reshaped
#   into the standardized FTSFR long format (drives the analysis below)
#
# Data source:
#
# - [Federal Reserve nominal yield curve](https://www.federalreserve.gov/data/nominal-yield-curve.htm) (`feds200628.csv`)
#
# Reference:
#
# - Gurkaynak, Refet S., Brian Sack, and Jonathan H. Wright. "The US Treasury yield
#   curve: 1961 to the present." *Journal of Monetary Economics* 54.8 (2007): 2291-2304.

# %% [markdown]
# ### Full Fitted Output (`fed_yield_curve_all.parquet`)
#
# Every series published in `feds200628.csv`: the fitted Svensson parameters plus
# the zero-coupon, par, and forward-rate curves. Indexed by observation date.
#
# | Column(s) | Description |
# |-----------|-------------|
# | Date (index) | Observation date (daily) |
# | BETA0-BETA3 | Fitted Svensson (1994) yield-curve parameters |
# | TAU1, TAU2 | Fitted Svensson decay / scale parameters |
# | SVENY01-SVENY30 | Continuously-compounded zero-coupon yields, 1-30yr (percent) |
# | SVENPY01-SVENPY30 | Par yields, 1-30yr (percent) |
# | SVENF01-SVENF30 | Instantaneous forward rates, 1-30yr ahead (percent) |
# | SVEN1F01, SVEN1F04, SVEN1F09 | One-year forward rate beginning 1, 4, and 9 years ahead (percent) |

# %%
df_all = pd.read_parquet(DATA_DIR / "fed_yield_curve_all.parquet")
print(f"Shape: {df_all.shape}")
print(f"Number of columns: {df_all.shape[1]}")
print(f"\nDate range: {df_all.index.min()} to {df_all.index.max()}")
display(df_all)

# %% [markdown]
# ### Zero-Coupon Yields, Wide (`fed_yield_curve.parquet`)
#
# The `SVENY01`-`SVENY30` subset of the full output: continuously-compounded
# zero-coupon Treasury yields at 1-30 year maturities, one column per tenor,
# indexed by observation date.
#
# | Column(s) | Description |
# |-----------|-------------|
# | Date (index) | Observation date (daily) |
# | SVENY01-SVENY30 | Continuously-compounded zero-coupon Treasury yield, 1-30yr maturity (percent) |

# %%
df_zc = pd.read_parquet(DATA_DIR / "fed_yield_curve.parquet")
print(f"Shape: {df_zc.shape}")
print(f"Number of tenors: {df_zc.shape[1]}")
print(f"\nDate range: {df_zc.index.min()} to {df_zc.index.max()}")
display(df_zc)

# %% [markdown]
# ### Zero-Coupon Yields, FTSFR Long Format (`ftsfr_treas_yield_curve_zero_coupon.parquet`)
#
# The same zero-coupon yields reshaped into the standardized FTSFR schema: one row
# per (tenor, date) with the yield stacked into a single `y` column. This is the
# dataset used for the analysis in the rest of the notebook.
#
# | Variable | Description |
# |----------|-------------|
# | unique_id | Tenor / maturity label (e.g. `SVENY10` = 10-year zero-coupon yield) |
# | ds | Observation date (daily) |
# | y | Zero-coupon Treasury yield (percent) |

# %%
df = pd.read_parquet(DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDate range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Number of tenors: {df['unique_id'].nunique()}")
display(df)

# %%
# Pivot to wide format for analysis
df_wide = df.pivot(index="ds", columns="unique_id", values="y")
print(f"Wide format shape: {df_wide.shape}")
print(f"Tenors: {df_wide.columns.tolist()}")

# %% [markdown]
# ## Summary Statistics

# %%
# Summary statistics for each tenor
summary_stats = df_wide.describe().T
display(summary_stats)

# %% [markdown]
# ## Data Coverage

# %%
print(f"Date range: {df_wide.index.min()} to {df_wide.index.max()}")
print(f"Number of observations: {len(df_wide)}")
print(f"Missing values per tenor:")
print(df_wide.isnull().sum())

# %% [markdown]
# ## GSW Paper Comparisons

# %% [markdown]
# For reference, we compare the estimated Federal Reserve Treasury yield curve with
# the results reported in the Gurkaynak, Sack, and Wright (2007) paper.
#
# Figures 4 and 5 use only the published Fed data and are reproduced below. Figures
# 1, 2, and 6 additionally require CRSP Treasury data from WRDS and are reproduced
# in the companion notebook `gsw_crsp_replication_ipynb` — set `WRDS_USERNAME` in
# `.env` to enable them (see the README).

# %% [markdown]
# ### Figure 4

# %% [markdown]
# For reference, figure_4.png:
#
# ![Figure 4](../assets/figure_4.png)

# %% [markdown]
# We reproduce this figure directly from our published data by plotting the
# zero-coupon yields (`SVENY01`-`SVENY30`) and instantaneous forward rates
# (`SVENF01`-`SVENF30`) on May 9, 2006 — the same date used in the paper. A match
# confirms our fitted-curve data agrees with GSW (2007). Our series cover 1-30 year
# maturities; the paper's continuous curves also extend down to maturity 0.

# %%
# Reproduce GSW Figure 4: zero-coupon yield curve and instantaneous forward rates
fig4_date = pd.Timestamp("2006-05-09")
maturities = list(range(1, 31))
zero_yields = [df_all.loc[fig4_date, f"SVENY{m:02d}"] for m in maturities]
fwd_rates = [df_all.loc[fig4_date, f"SVENF{m:02d}"] for m in maturities]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(maturities, fwd_rates, "b-", linewidth=3, label="Instantaneous Forward Rate")
ax.plot(maturities, zero_yields, "b-", linewidth=1, label="Zero-Coupon Yield")
ax.set_title(f"Zero-Coupon Yield Curve and Forward Rates on {fig4_date:%B %d, %Y}", fontsize=14)
ax.set_xlabel("Maturity in Years")
ax.set_ylabel("Yield")
ax.set_xlim(0, 30)
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 5

# %% [markdown]
# For reference, figure_5.png:
#
# ![Figure 5](../assets/figure_5.png)

# %% [markdown]
# We reproduce this figure from the fitted Svensson parameters (`BETA0`-`BETA3`,
# `TAU1`, `TAU2`) on May 9, 2006. The zero-coupon yield is the sum of a constant
# level term plus three shape terms:
#
# $$y(n) = \beta_0
#   + \beta_1 \frac{1 - e^{-n/\tau_1}}{n/\tau_1}
#   + \beta_2 \left(\frac{1 - e^{-n/\tau_1}}{n/\tau_1} - e^{-n/\tau_1}\right)
#   + \beta_3 \left(\frac{1 - e^{-n/\tau_2}}{n/\tau_2} - e^{-n/\tau_2}\right)$$
#
# Plotting the four terms separately decomposes the curve. As a check, summing them
# reproduces the published `SVENY` yields exactly (to numerical precision).

# %%
# Reproduce GSW Figure 5: decompose the fitted yield curve into its Svensson terms
fig5_date = pd.Timestamp("2006-05-09")
p = df_all.loc[fig5_date]
b0, b1, b2, b3 = p["BETA0"], p["BETA1"], p["BETA2"], p["BETA3"]
tau1, tau2 = p["TAU1"], p["TAU2"]

n = np.linspace(0.01, 100, 1000)  # maturity in years (start >0 to avoid 0/0)
a, b = n / tau1, n / tau2
term0 = np.full_like(n, b0)
term1 = b1 * (1 - np.exp(-a)) / a
term2 = b2 * ((1 - np.exp(-a)) / a - np.exp(-a))
term3 = b3 * ((1 - np.exp(-b)) / b - np.exp(-b))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n, term0, "b-", linewidth=3, label=r"$\beta_0$")
ax.plot(n, term1, "b--", linewidth=3, label=r"$\beta_1$ term")
ax.plot(n, term2, "b-", linewidth=1, label=r"$\beta_2$ term")
ax.plot(n, term3, "b-.", linewidth=1, label=r"$\beta_3$ term")
ax.set_title(f"Decomposition of the Yield Curve on {fig5_date:%B %d, %Y}", fontsize=14)
ax.set_xlabel("Maturity in Years")
ax.set_ylabel("Yield")
ax.set_xlim(0, 100)
ax.set_ylim(0, 6)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Verify the four terms sum to the published zero-coupon yields
check_mats = [1, 5, 10, 20, 30]
ac, bc = np.array(check_mats) / tau1, np.array(check_mats) / tau2
recon = (
    b0
    + b1 * (1 - np.exp(-ac)) / ac
    + b2 * ((1 - np.exp(-ac)) / ac - np.exp(-ac))
    + b3 * ((1 - np.exp(-bc)) / bc - np.exp(-bc))
)
print("Maturity | reconstructed | published SVENY | diff")
for m, rc in zip(check_mats, recon):
    print(f"{m:8d} | {rc:13.4f} | {p[f'SVENY{m:02d}']:15.4f} | {rc - p[f'SVENY{m:02d}']:+.6f}")

# %% [markdown]
# ## Current Yield Curve
#
# The most recent yield curve snapshot.

# %%
# Get the most recent yield curve
latest_date = df_wide.index.max()
latest_curve = df_wide.loc[latest_date].dropna()

# Extract maturity in years from tenor names (e.g., SVENY01 -> 1)
maturities = [int(col.replace("SVENY", "")) for col in latest_curve.index]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(maturities, latest_curve.values, "b-o", linewidth=2, markersize=6)
ax.set_title(f'Treasury Yield Curve as of {latest_date.strftime("%Y-%m-%d")}', fontsize=14)
ax.set_xlabel("Maturity (Years)")
ax.set_ylabel("Yield (%)")
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 31, 5))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Historical Yield Curve Evolution
#
# Time series of selected tenors.

# %%
# Select key tenors for visualization
key_tenors = ["SVENY01", "SVENY02", "SVENY05", "SVENY10", "SVENY30"]
available_tenors = [t for t in key_tenors if t in df_wide.columns]

fig, ax = plt.subplots(figsize=(14, 6))
for tenor in available_tenors:
    label = tenor.replace("SVENY", "") + "Y"
    ax.plot(df_wide.index, df_wide[tenor], label=label, linewidth=0.8)

ax.set_title("Historical Treasury Yields by Maturity", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Yield (%)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Yield Curve Slope (10Y - 2Y Spread)

# %%
if "SVENY10" in df_wide.columns and "SVENY02" in df_wide.columns:
    spread_10_2 = df_wide["SVENY10"] - df_wide["SVENY02"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(spread_10_2.index, spread_10_2.values, linewidth=0.8, color="green")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.fill_between(spread_10_2.index, 0, spread_10_2.values,
                    where=(spread_10_2 < 0), color="red", alpha=0.3, label="Inverted")
    ax.fill_between(spread_10_2.index, 0, spread_10_2.values,
                    where=(spread_10_2 >= 0), color="green", alpha=0.3, label="Normal")
    ax.set_title("Treasury 10Y-2Y Spread (Yield Curve Slope)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nSpread Statistics:")
    print(f"  Mean: {spread_10_2.mean():.2f}%")
    print(f"  Std: {spread_10_2.std():.2f}%")
    print(f"  Min: {spread_10_2.min():.2f}%")
    print(f"  Max: {spread_10_2.max():.2f}%")
    print(f"  Current: {spread_10_2.iloc[-1]:.2f}%")

# %% [markdown]
# ## Correlation Matrix
#
# Correlation between yields at different maturities.

# %%
# Select subset of tenors for correlation
corr_tenors = ["SVENY01", "SVENY02", "SVENY05", "SVENY07", "SVENY10", "SVENY20", "SVENY30"]
available_corr = [t for t in corr_tenors if t in df_wide.columns]
corr_matrix = df_wide[available_corr].corr()

# Rename for display
rename_dict = {t: t.replace("SVENY", "") + "Y" for t in available_corr}
corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0.9,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix: Treasury Yields Across Maturities", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This dataset provides daily zero-coupon Treasury yields estimated using the
# Gurkaynak, Sack, and Wright (2007) model. The yields cover maturities from 1 to 30
# years and are widely used in fixed income research and practice.
#
# Key applications:
#
# - Yield curve analysis and term structure modeling
# - Fixed income derivatives pricing
# - Monetary policy analysis
# - Risk-free rate estimation

# %% [markdown]
#
