# Statistical Factor Risk Model

This project implements the first executable slice of the specification in `statistical_factor_risk_model_spec.md`:

- recursive open-ended EWMA standardization
- global and 252-observation blockwise PCA
- orthogonal Procrustes alignment
- continuous regime scores and low/high stress loading baselines
- daily WLS factor innovations
- nonlinear delta, gamma, and downside asset exposures

## Setup

```powershell
python -m pip install -e ".[test]"
```

That installation is sufficient to rebuild the reports from the committed data;
the production build uses only the project dependencies and does not contact the
network.

Install data retrieval and visualization dependencies only when downloading fresh data or building figures:

```powershell
python -m pip install -e ".[data,visualization]"
```

## Usage

```python
from riskmodel.pipeline import run_factor_pipeline

result = run_factor_pipeline(asset_returns, benchmark_ewma_volatility, n_factors=8)
factor_returns = result.factor_returns
```

The pipeline accepts prepared return data so it can be tested and rebuilt without network access. `riskmodel.data_loader.fetch_asset_data` provides the optional `yfinance` adapter for refreshing the committed CSV.

## PUT versus S&P 500 comparison

Download the long-history Cboe PutWrite strategy index alongside the S&P 500 index:

```powershell
python scripts/download_put_spx_comparison.py
```

This writes aligned daily returns to `data/raw/put_spx_daily_returns.csv` and base-100 indexed performance to `data/raw/put_spx_performance.csv`. The download metadata is recorded in `data/raw/put_spx_metadata.json`.

## Interactive explorer

After generating the report CSVs, build the standalone factor-model cockpit:

```powershell
python scripts/build_explorer.py
```

Open [reports/factor_model_explorer.html](reports/factor_model_explorer.html) locally. It is a full-viewport cockpit: searchable asset rail, linked turntable views of exposure and residual MDS, and an inspector for holdout diagnostics and factor variance shares.

## Model selection

The current production configuration uses eight statistical factors. Residual PCA was tested as an extension but is retained as a diagnostic experiment rather than included in the production factor set: adding two or three residual factors improved mean holdout OOS $R^2$ only marginally and did not remove the largest shared residual correlations.

### Benchmark and target universes

Factor construction and target decomposition use separate universes. The benchmark universe is a broad, liquid cross-section intended to define stable statistical factors: global and US equities (`ACWI`, `SPY`, `DIA`, `QQQ`, `IWM`, `EFA`, `EEM`), Treasury duration (`SHY`, `IEF`, `TLT`), aggregate and inflation-linked bonds, credit, real estate, gold, commodities, and the dollar. The target universe contains alternative strategies and funds whose raw and residual relationships are studied.

The factor model standardizes the benchmark panel, fits global and rolling 252-observation PCA decompositions on the training period, aligns the block loadings, and averages them into low- and high-regime loading baselines. Target exposures are then fitted against those factors without allowing target assets to define the factors. `ACWI` remains the benchmark volatility anchor.

The target universe also includes US style sleeves (`MTUM`, `QUAL`, `USMV`, `VLUE`), option-income funds (`JEPI`, `JEPQ`, `DIVO`), thematic growth (`ARKK`), and hedge-fund replication (`QAI`). These are intentionally analysis assets: they broaden the raw-versus-residual study without contaminating the factor definitions. Provider availability and launch dates are recorded in `data/raw/universe.json`; missing or short-history targets should be reported rather than used to shorten the benchmark panel.

The benchmark panel should be downloaded with a long common history before reports are regenerated. The current target history is shorter for several newer products; keeping those assets out of factor construction prevents them from truncating the benchmark training sample.

```powershell
python scripts/regenerate_reports.py
```

For a fresh GitHub checkout, the complete restart is:

```powershell
git clone <repository-url>
cd RiskModel
python -m pip install -e ".[test]"
python scripts/regenerate_reports.py
python -m pytest -q
```

The committed source inputs are under `data/raw/`. The model CSVs, figures, and
HTML reports are build products: the first command above recreates them under
`reports/` from those inputs. The rebuild script resolves paths from the checkout,
so it can also be invoked with an absolute script path from another directory.

This reads the committed offline CSV, writes intermediate model CSVs under
`reports/`, and regenerates both HTML reports. Use
`python scripts/regenerate_reports.py --download` only when you intentionally want
to refresh the raw market data through yfinance. Generated CSVs, diagnostic
figures, and the PDF correlation report are ignored by git.

Experiment outputs are written to `reports/experiments/`, including `residual_pca_experiment.csv` for reproducibility.

## Validation

```powershell
python -m pytest -q
```
