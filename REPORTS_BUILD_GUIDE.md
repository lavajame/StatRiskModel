# Recreating the Research Viewer & Factor Explorer (offline)

Both interactive HTML reports are **self-contained** — all CSS/JS is inlined and no
network calls happen when the page is opened. Everything they render is derived
from a set of CSV/JSON inputs in `reports/` and, ultimately, raw market data in
`data/raw/`. This guide lists exactly what you need to reproduce them in a private
GitHub repo, and how to pull the live data into offline CSVs first.

---

## 1. The two reports and how they are built

| Report | HTML file | Build script | Reads these CSVs |
|---|---|---|---|
| **Research Viewer** (guided audit: universe → frozen factors → funds → strategies) | `reports/research_viewer.html` | `scripts/build_research_viewer_polished.py` | `backtest_stats.csv`, `alpha_backtest_stats.csv`, `alpha_backtest_equity.csv`, `oos_latent_factor_moves.csv`, `strategy_oos_fitted.csv`, `strategy_oos_residuals.csv`, `frozen_factor_loadings.csv`, `strategy_train_exposures.csv`, `benchmark_cluster_order.csv`, `equal_weight_raw_fund_weights.csv`, `trend_raw_fund_weights.csv`, `trend_residual_fund_weights.csv`, `trend_residual_benchmark_hedge_exposures.csv`, + raw returns |
| **Factor Model Explorer** (3-D turntable: exposure cloud + residual MDS scatter) | `reports/factor_model_explorer.html` | `scripts/build_explorer.py` | `benchmark_train_exposures.csv`, `strategy_train_exposures.csv`, `benchmark_oos_diagnostics.csv`, `strategy_oos_diagnostics.csv`, `benchmark_factor_variance_decomposition.csv`, `strategy_factor_variance_decomposition.csv` |

The `research_viewer.py`, `build_research_viewer_v2.py`, and `build_research_viewer_v3.py`
scripts are older drafts — **use the polished version** (`build_research_viewer_polished.py`)
as the canonical builder.

---

## 2. Data needed — layered from bottom up

### Layer A — Raw market data (the source of truth)

**`data/raw/daily_returns.csv`** — daily adjusted-close returns (percent change),
index-labelled rows, missing cells left blank.

- **Tickers (36):** `ACWI, AGG, ARKK, BTAL, CTA, DBC, DBMF, DIA, DIVO, EEM, EFA, GLD, HYG, IEF, IWM, JEPI, JEPQ, KMLM, LQD, MNA, MTUM, PRPFX, QAI, QQQ, QUAL, RPAR, SHY, SPY, TIP, TLT, USMV, UUP, VLUE, VNQ, ^PUT`
- **Type:** ETFs + indexes (equities, bonds, credit, REIT, gold, commodities, dollar) plus the Cboe `^PUT` PutWrite index.
- **Range:** ~2012-01-04 → 2026-08-15 (matches `data/raw/universe.json` `start`/`end`).
- **Source:** yfinance `auto_adjust=False` adjusted close → `.pct_change()`.
- **Definition:** each row is the *daily simple return*; `null` where a ticker had no trading day.

**`data/raw/put_spx_daily_returns.csv`** and **`.../put_spx_performance.csv`**
- **Tickers:** `^PUT`, `^SPX` (returns + indexed performance, base 100).
- **Range:** ~1996-08-05 → 2026-08-26 (7,550 common observations).
- **Source:** yfinance on the Cboe S&P 500 **PutWrite Index** (`^PUT`) and the
  **S&P 500 Index** (`^SPX`).
- **Metadata:** `data/raw/put_spx_metadata.json`.

**`data/raw/universe.json`** — the ticker list and human labels. Not strictly needed
for the two HTML reports, but useful for downloading the right tickers and for labels.
It lists `downloaded_tickers` and a `model_universe` description map.

### Layer B — Model reports (intermediate CSVs)

These are produced by the analysis pipeline (`scripts/run_experiments.py`,
`scripts/build_correlation_report.py`, `scripts/backtest.py`, and
`src/riskmodel/**`) — **not** downloaded from anywhere. They are the direct inputs
to the HTML builders. Key ones:

| CSV | Contents used by the viewer/explorer |
|---|---|
| `benchmark_train_exposures.csv` | training factor betas (`beta_Factor_1..8`), gamma/delta/downside, `r_squared` |
| `strategy_train_exposures.csv` | fund factor betas + `alpha` |
| `benchmark_oos_diagnostics.csv` | `realized_volatility`, `systematic_volatility`, `idiosyncratic_volatility`, `oos_r_squared` |
| `strategy_oos_diagnostics.csv` | same diagnostics + `residual_return` |
| `frozen_factor_loadings.csv` | 8-PC loading matrix per benchmark (the "frozen loading space") |
| `oos_latent_factor_moves.csv` | daily latent factor innovations (8 factors), OOS window ~2022 → present, standardized scores |
| `strategy_oos_fitted.csv` / `strategy_oos_residuals.csv` | per-fund fitted and residual returns on the OOS window |
| `benchmark_factor_variance_decomposition.csv` / `strategy_...` | per-asset variance share across the 8 factors |
| `benchmark_cluster_order.csv` | hierarchical-cluster order of benchmarks |
| `backtest_stats.csv` / `alpha_backtest_stats.csv` | realized/target vol, CAGR, max drawdown, Sharpe, turnover |
| `equal_weight_raw_fund_weights.csv`, `trend_raw_fund_weights.csv`, `trend_residual_fund_weights.csv`, `trend_residual_benchmark_hedge_exposures.csv` | portfolio weight sets for the strategy charts |

### Layer C — No extra data needed for the HTML

- The **3-D residual embedding** (`residual_embedding`) and **correlation edges**
  (`residual_edges`) shown in the Factor Explorer are **computed at build time**
  from `strategy_oos_residuals.csv` (classic MDS on the residual correlation matrix,
  `|ρ| ≥ 0.20` edges). You do not need to store them.
- The **scatter points** in the Research Viewer come from `strategy_oos_residuals.csv`
  vs. raw-benchmark correlations, computed inline.

---

## 3. Tools needed

| Tool | Purpose | How it's used |
|---|---|---|
| **Python 3.9+** | Runs the build scripts | `pandas`, `numpy` (stdlib math otherwise) |
| **yfinance ≥ 0.2** *(optional, only for live download)* | Fetch raw prices | `riskmodel.data_loader.fetch_asset_data()` |
| **scipy / sklearn** | Factor PCA + clustering (pipeline only) | `build_correlation_report.py`, `run_experiments.py` |
| **pandas** | All CSV reading/writing | every build script |
| **web browser (headless ok)** | Preview the HTML | open the file directly — no server needed |

No runtime dependencies for the HTML itself — it is pure inline HTML/CSS/JS.

---

## 4. Recommended offline workflow (GitHub-ready)

### Step 1 — Pull live data into offline CSVs (one-time)

```bash
# 1. Bulk ETF/index daily returns -> data/raw/daily_returns.csv
python -c "from riskmodel.data_loader import fetch_asset_data; fetch_asset_data(
    ['ACWI','AGG','ARKK','BTAL','CTA','DBC','DBMF','DIA','DIVO','EEM','EFA','GLD',
     'HYG','IEF','IWM','JEPI','JEPQ','KMLM','LQD','MNA','MTUM','PRPFX','QAI','QQQ',
     'QUAL','RPAR','SHY','SPY','TIP','TLT','USMV','UUP','VLUE','VNQ','^PUT'],
    '2012-01-01','2026-09-01').to_csv('data/raw/daily_returns.csv', index_label='Date')"

# 2. Cboe PutWrite index vs S&P 500 -> data/raw/put_spx_*.csv + *.json
python scripts/download_put_spx_comparison.py --start 1996-01-01 --end 2026-09-01
```

> The ETF tickers and `^PUT`/`^SPX` match the symbols in `data/raw/universe.json`.
> Run this **once**; commit the resulting CSVs so the repo is fully offline.

### Step 2 — Run the analysis pipeline → `reports/*.csv`

```bash
python scripts/run_experiments.py        # factor model, OOS scoring, betas, diagnostics
python scripts/build_correlation_report.py
python scripts/backtest.py               # strategy + alpha backtests, weights
```

### Step 3 — Build the two HTML reports (fully offline)

```bash
python scripts/build_research_viewer_polished.py   # -> reports/research_viewer.html
python scripts/build_explorer.py                    # -> reports/factor_model_explorer.html
```

Open the HTML files in any browser. No internet, no packages, no build server.

---

## 5. Minimal file map for a shareable repo

```
data/raw/
  daily_returns.csv                 # Layer A — the core input
  put_spx_daily_returns.csv
  put_spx_performance.csv
  put_spx_metadata.json
  universe.json                     # ticker list + labels
scripts/
  data_loader.py                    # yfinance adapter + EWMA standardization
  download_put_spx_comparison.py    # ^PUT/^SPX downloader
  run_experiments.py                # factor model + OOS scoring
  build_correlation_report.py       # loadings, diagnostics, variance decomposition
  backtest.py                       # strategy + alpha backtests
  build_research_viewer_polished.py # -> research_viewer.html
  build_explorer.py                 # -> factor_model_explorer.html
src/riskmodel/                      # pipeline package (universes, data_loader, ...)
reports/                            # intermediate CSVs + final HTML
tests/
```

---

## 6. Key facts to remember

- **Both HTML files are static** — they embed their whole dataset as inline JSON
  and render with vanilla JS (SVG charts, no libraries). You can edit them by hand
  if needed, but the build scripts are the source of truth.
- **`^PUT`** is the Cboe S&P 500 **VIX PutWrite Index** — treated as a *benchmark*
  in the model, shown in gold in both viewers.
- **8 latent factors** from a chunked/aligned PCA of the benchmark return blocks.
- The **OOS window** for factor moves and fund scoring is roughly **2022 → present**;
  the training window for betas/loadings spans the full history (2012 → 2026).
- The reports use a **recursive EWMA (α = 0.99)** standardization of returns.
