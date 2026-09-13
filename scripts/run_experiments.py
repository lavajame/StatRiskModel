"""Run out-of-sample experiments for improving benchmark factor capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from riskmodel.asset_mapper import fit_nonlinear_asset_exposure, predict_nonlinear_asset_exposure
from riskmodel.data_loader import standardize_returns_ewma
from riskmodel.factor_engine import extract_block_loadings, fit_global_pca, solve_daily_factors
from riskmodel.regime import compute_regime_scores, construct_tilted_loadings
from riskmodel.universes import BENCHMARK_UNIVERSE


BENCHMARK_ANCHOR = "ACWI"
BLOCK_SIZE = 252
HOLDOUT_FRACTION = 0.30


OUTPUT = Path("reports/experiments")


def fit_factor_split(
    returns: pd.DataFrame, benchmark_volatility: pd.Series, split: int, n_factors: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standardized, _ = standardize_returns_ewma(returns)
    train_std, test_std = standardized.iloc[:split], standardized.iloc[split:]
    train_vol = benchmark_volatility.loc[train_std.index]
    test_vol = benchmark_volatility.loc[test_std.index]
    reference = fit_global_pca(train_std, n_factors)
    blocks = extract_block_loadings(train_std, reference, n_factors, BLOCK_SIZE)
    block_regimes = np.array([
        train_vol.iloc[index * BLOCK_SIZE : (index + 1) * BLOCK_SIZE].mean()
        for index in range(len(blocks))
    ])
    low, high = construct_tilted_loadings(blocks, block_regimes)
    train_scores = compute_regime_scores(train_vol)
    test_scores = compute_regime_scores(
        test_vol,
        calibration_mean=float(train_vol.mean()),
        calibration_std=float(train_vol.std()),
    )
    train_factors = solve_daily_factors(train_std, low, high, train_scores)
    test_factors = solve_daily_factors(test_std, low, high, test_scores)
    return train_std, test_std, train_factors, test_factors


def score_mapping(
    returns: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_factors: pd.DataFrame,
    test_factors: pd.DataFrame,
    basis: str = "nonlinear",
) -> pd.DataFrame:
    records = []
    for ticker in returns:
        train_asset = train_returns[ticker]
        common_train = train_asset.index.intersection(train_factors.index)
        y_train = train_asset.loc[common_train].to_numpy(float)
        f_train = train_factors.loc[common_train].to_numpy(float)
        valid_train = np.isfinite(y_train) & np.isfinite(f_train).all(axis=1)
        y_train, f_train = y_train[valid_train], f_train[valid_train]
        if basis == "linear":
            x_train = f_train
        elif basis == "gamma":
            x_train = np.hstack((f_train, f_train**2 - np.mean(f_train**2, axis=0)))
        else:
            x_train = np.hstack((
                f_train,
                f_train**2 - np.mean(f_train**2, axis=0),
                np.minimum(f_train, 0.0),
            ))
        coefficients = linalg.solve(
            x_train.T @ x_train + 1e-3 * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
            assume_a="pos",
        )

        common_test = returns[ticker].index.intersection(test_factors.index)
        y_test = returns.loc[common_test, ticker].to_numpy(float)
        f_test = test_factors.loc[common_test].to_numpy(float)
        valid_test = np.isfinite(y_test) & np.isfinite(f_test).all(axis=1)
        y_test, f_test = y_test[valid_test], f_test[valid_test]
        if basis == "linear":
            x_test = f_test
        elif basis == "gamma":
            x_test = np.hstack((f_test, f_test**2 - np.mean(f_train**2, axis=0)))
        else:
            x_test = np.hstack((
                f_test,
                f_test**2 - np.mean(f_train**2, axis=0),
                np.minimum(f_test, 0.0),
            ))
        fitted = x_test @ coefficients
        residual = y_test - fitted
        denominator = np.sum((y_test - y_train.mean()) ** 2)
        records.append({
            "ticker": ticker,
            "basis": basis,
            "oos_r_squared": 1.0 - np.sum(residual**2) / denominator if denominator else 0.0,
            "residual_volatility": residual.std() * np.sqrt(252),
        })
    return pd.DataFrame(records)


def residual_pca_experiment(
    train_std: pd.DataFrame,
    test_std: pd.DataFrame,
    train_factors: pd.DataFrame,
    test_factors: pd.DataFrame,
    returns: pd.DataFrame,
    train_returns: pd.DataFrame,
    n_residual_factors: int = 2,
) -> pd.DataFrame:
    """Add PCA factors extracted only from training-period baseline residuals."""
    train_residual = train_std.copy()
    for ticker in train_std:
        common = train_std.index.intersection(train_factors.index)
        x = train_factors.loc[common].to_numpy()
        y = train_std.loc[common, ticker].to_numpy()
        coefficients = linalg.lstsq(x, y, cond=None)[0]
        train_residual.loc[common, ticker] = y - x @ coefficients
    # Residual PCA is intentionally evaluated as a diagnostic extension: the
    # first residual factors are learned from training residual cross-section.
    correlation = train_residual.corr().to_numpy()
    _, vectors = linalg.eigh(correlation)
    residual_vectors = vectors[:, -n_residual_factors:]
    train_residual_factors = train_residual @ residual_vectors
    test_residual = test_std.copy()
    for ticker in test_std:
        common = train_factors.index
        x = train_factors.loc[common].to_numpy()
        y = train_std.loc[common, ticker].to_numpy()
        coefficients = linalg.lstsq(x, y, cond=None)[0]
        test_residual.loc[:, ticker] = test_std[ticker].to_numpy() - test_factors.to_numpy() @ coefficients
    test_residual_factors = test_residual @ residual_vectors
    augmented_train = pd.concat([train_factors, train_residual_factors.add_prefix("Residual_")], axis=1)
    augmented_test = pd.concat([test_factors, test_residual_factors.add_prefix("Residual_")], axis=1)
    result = score_mapping(returns, train_returns, augmented_train, augmented_test, "linear")
    return result.assign(
        experiment="base_plus_residual_pca",
        base_factors=train_factors.shape[1],
        residual_factors=n_residual_factors,
    )


def economic_factor_experiment(
    returns: pd.DataFrame, train_returns: pd.DataFrame, test_returns: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate a transparent proxy-factor model without using each asset itself."""
    factor_definitions = {
        "Global equity": (returns[BENCHMARK_ANCHOR], {BENCHMARK_ANCHOR}),
        "Nominal duration": (returns["AGG"], {"AGG"}),
        "Inflation": (returns["TIP"] - returns["AGG"], {"TIP", "AGG"}),
        "Credit spread": (returns["HYG"] - returns["LQD"], {"HYG", "LQD"}),
        "Commodities": (returns["DBC"], {"DBC"}),
        "Gold": (returns["GLD"], {"GLD"}),
        "Dollar": (returns["UUP"], {"UUP"}),
    }
    rows = []
    for ticker in returns:
        usable = {
            name: series
            for name, (series, dependencies) in factor_definitions.items()
            if ticker not in dependencies
        }
        train_factors = pd.DataFrame(usable).loc[train_returns.index]
        test_factors = pd.DataFrame(usable).loc[test_returns.index]
        rows.append(
            score_mapping(
                returns[[ticker]],
                train_returns[[ticker]],
                train_factors,
                test_factors,
                "linear",
            )
        )
    return pd.concat(rows, ignore_index=True).assign(experiment="economic_proxy_factors")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv("data/raw/daily_returns.csv", index_col=0, parse_dates=True)
    tickers = [ticker for ticker in BENCHMARK_UNIVERSE if ticker in raw]
    returns = raw[tickers].dropna()
    benchmark_volatility = returns[BENCHMARK_ANCHOR].ewm(alpha=0.01, adjust=False).std().fillna(0.0)
    split = int(len(returns) * (1.0 - HOLDOUT_FRACTION))
    train_returns, test_returns = returns.iloc[:split], returns.iloc[split:]

    factor_rows = []
    nested_rows = []
    for n_factors in range(3, min(8, len(tickers)) + 1):
        train_std, test_std, train_factors, test_factors = fit_factor_split(
            returns, benchmark_volatility, split, n_factors
        )
        scored = score_mapping(returns, train_returns, train_factors, test_factors)
        scored["n_factors"] = n_factors
        factor_rows.append(scored)
        for basis in ("linear", "gamma", "nonlinear"):
            nested = score_mapping(returns, train_returns, train_factors, test_factors, basis)
            nested["n_factors"] = n_factors
            nested_rows.append(nested)

    factor_results = pd.concat(factor_rows, ignore_index=True)
    nested_results = pd.concat(nested_rows, ignore_index=True)
    factor_results.to_csv(OUTPUT / "factor_count_sweep.csv", index=False)
    nested_results.to_csv(OUTPUT / "nested_basis_comparison.csv", index=False)

    train_std, test_std, train_factors, test_factors = fit_factor_split(
        returns, benchmark_volatility, split, 8
    )
    residual_results = pd.concat(
        [
            residual_pca_experiment(
                train_std,
                test_std,
                train_factors,
                test_factors,
                returns,
                train_returns,
                n_residual_factors,
            )
            for n_residual_factors in (1, 2, 3)
        ],
        ignore_index=True,
    )
    residual_results.to_csv(OUTPUT / "residual_pca_experiment.csv", index=False)
    economic_results = economic_factor_experiment(returns, train_returns, test_returns)
    economic_results.to_csv(OUTPUT / "economic_factor_experiment.csv", index=False)

    walk_rows = []
    fold_size = 252
    for fold, test_start in enumerate(range(2 * fold_size, len(returns) - fold_size + 1, fold_size), start=1):
        train_end = test_start
        test_end = min(test_start + fold_size, len(returns))
        fold_train = returns.iloc[:train_end]
        fold_test = returns.iloc[test_start:test_end]
        if len(fold_train) < 2 * fold_size or len(fold_test) < 50:
            continue
        vol = returns[BENCHMARK_ANCHOR].ewm(alpha=0.01, adjust=False).std().fillna(0.0)
        _, _, train_factors, test_factors = fit_factor_split(returns, vol, train_end, 8)
        scored = score_mapping(returns, fold_train, train_factors, test_factors)
        scored["fold"] = fold
        scored["test_start"] = fold_test.index.min()
        scored["test_end"] = fold_test.index.max()
        walk_rows.append(scored)
    walkforward = pd.concat(walk_rows, ignore_index=True)
    walkforward.to_csv(OUTPUT / "walkforward_validation.csv", index=False)

    summary = pd.DataFrame([
        {"experiment": "factor_count", "setting": str(n), "mean_oos_r2": factor_results.loc[factor_results.n_factors == n, "oos_r_squared"].mean(), "worst_oos_r2": factor_results.loc[factor_results.n_factors == n, "oos_r_squared"].min()}
        for n in sorted(factor_results.n_factors.unique())
    ])
    summary.to_csv(OUTPUT / "experiment_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote experiment outputs to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
