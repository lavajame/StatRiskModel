"""Download research data, fit the factor model, and build HTML visuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from riskmodel.asset_mapper import (
    fit_nonlinear_asset_exposure,
    predict_nonlinear_asset_exposure,
)
from riskmodel.data_loader import fetch_asset_data, standardize_returns_ewma
from riskmodel.factor_engine import extract_block_loadings, fit_global_pca
from riskmodel.regime import compute_regime_scores, construct_tilted_loadings
from riskmodel.universes import BENCHMARK_UNIVERSE, EXPLANATORY_UNIVERSE, display_labels


BENCHMARK_ANCHOR = "ACWI"
N_FACTORS = 8
BLOCK_SIZE = 252
HOLDOUT_FRACTION = 0.30
from riskmodel.visuals import (
    plot_cumulative_returns,
    plot_correlation_heatmap,
    plot_capture_diagnostics,
    plot_factor_decomposition,
    plot_loadings,
    plot_realized_vs_fitted,
    plot_systematic_vs_idio,
    plot_backtest_comparison,
    plot_backtest_attribution,
    plot_factor_moves,
    plot_fund_factor_attribution,
)
from backtest import BacktestResult, factor_attribution, run_trend_following, run_neutral_following, weekly_rebalanced_weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-08-15")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/daily_returns.csv"),
        help="Offline daily-returns CSV; download from yfinance only when absent",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Ignore --input and download fresh market data with yfinance",
    )
    parser.add_argument("--output", type=Path, default=Path("reports"))
    return parser.parse_args()


def _fit_holdout_model(
    model_returns: pd.DataFrame,
    benchmark_volatility: pd.Series,
    split: int,
    n_factors: int = 5,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Fit static factor loadings on the training period and track both periods."""
    standardized, _ = standardize_returns_ewma(model_returns)
    train_std = standardized.iloc[:split]
    test_std = standardized.iloc[split:]
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
    # Freeze the historical loading space: average the aligned benchmark
    # eigenvectors, then estimate daily factor moves by linear regression.
    frozen = np.mean(blocks, axis=0)
    # Eigenvectors have arbitrary signs. Use a positive cross-sectional
    # orientation where possible so the loading heatmap is readable; the same
    # orientation is then used for every factor-move estimate.
    signs = np.where(frozen.sum(axis=0) < 0.0, -1.0, 1.0)
    frozen = frozen * signs
    train_factors = _solve_frozen_factors(train_std, frozen)
    test_factors = _solve_frozen_factors(test_std, frozen)
    return frozen, train_factors, test_factors


def _solve_frozen_factors(std_returns: pd.DataFrame, frozen_loadings: np.ndarray) -> pd.DataFrame:
    """Estimate latent daily factors from benchmark returns and frozen loadings."""
    factors = np.linalg.lstsq(frozen_loadings, std_returns.to_numpy(dtype=float).T, rcond=None)[0].T
    return pd.DataFrame(
        factors,
        index=std_returns.index,
        columns=[f"Factor_{index + 1}" for index in range(factors.shape[1])],
    )


def _linear_holdout_attribution(
    returns: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_factors: pd.DataFrame,
    test_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit fund betas in-sample, then score fixed betas on OOS factor moves."""
    records, exposures = [], []
    fitted_frame = pd.DataFrame(index=test_factors.index)
    residual_frame = pd.DataFrame(index=test_factors.index)
    for ticker in returns:
        common_train = train_returns[ticker].index.intersection(train_factors.index)
        train_frame = pd.concat([train_returns.loc[common_train, ticker], train_factors.loc[common_train]], axis=1).dropna()
        y = train_frame.iloc[:, 0].to_numpy(dtype=float)
        x = train_frame.iloc[:, 1:].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        common_test = returns[ticker].index.intersection(test_factors.index)
        scored = pd.concat([returns.loc[common_test, ticker], test_factors.loc[common_test]], axis=1).dropna()
        test_design = scored.iloc[:, 1:].to_numpy(dtype=float)
        fitted = pd.Series(test_design @ coefficients, index=scored.index)
        realized = scored.iloc[:, 0]
        residual = realized - fitted
        fitted_frame.loc[fitted.index, ticker] = fitted
        residual_frame.loc[residual.index, ticker] = residual
        total_var = realized.var(ddof=1)
        records.append({
            "ticker": ticker,
            "realized_volatility": realized.std(ddof=1) * np.sqrt(252),
            "systematic_volatility": fitted.std(ddof=1) * np.sqrt(252),
            "idiosyncratic_volatility": residual.std(ddof=1) * np.sqrt(252),
            "oos_r_squared": 1.0 - residual.var(ddof=1) / total_var if total_var > 0 else 0.0,
            "residual_return": residual.mean() * 252,
        })
        exposures.append({"ticker": ticker, "alpha": 0.0, **{
            f"beta_{name}": value for name, value in zip(train_factors.columns, coefficients)
        }})
    return pd.DataFrame(records).set_index("ticker"), pd.DataFrame(exposures).set_index("ticker"), fitted_frame, residual_frame


def _holdout_attribution(
    returns: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_factors: pd.DataFrame,
    test_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit exposures in-sample and score fixed exposures on the holdout period."""
    records = []
    exposures = []
    fitted_frame = pd.DataFrame(index=test_factors.index)
    residual_frame = pd.DataFrame(index=test_factors.index)
    test_returns = returns.loc[test_factors.index]
    for ticker in returns:
        exposure = fit_nonlinear_asset_exposure(train_returns[ticker], train_factors)
        common = test_returns[ticker].index.intersection(test_factors.index)
        raw_asset = test_returns.loc[common, ticker].to_numpy(dtype=float)
        factors = test_factors.loc[common]
        valid = np.isfinite(raw_asset) & np.isfinite(factors.to_numpy()).all(axis=1)
        raw_asset = raw_asset[valid]
        factors = factors.loc[valid]
        fitted = predict_nonlinear_asset_exposure(factors, exposure)
        realized = pd.Series(raw_asset, index=factors.index)
        fitted_frame.loc[factors.index, ticker] = fitted
        residual_frame.loc[factors.index, ticker] = realized - fitted
        train_mean = train_returns[ticker].mean()
        residual = realized.to_numpy() - fitted.to_numpy()
        systematic = fitted.to_numpy() - fitted.mean()
        total_sum = np.sum((realized.to_numpy() - train_mean) ** 2)
        residual_sum = np.sum(residual**2)
        oos_r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
        records.append(
            {
                "ticker": ticker,
                "systematic_volatility": systematic.std() * np.sqrt(252),
                "idiosyncratic_volatility": residual.std() * np.sqrt(252),
                "realized_volatility": realized.std() * np.sqrt(252),
                "oos_r_squared": oos_r_squared,
                "systematic_risk_budget_fraction": (
                    systematic.var() / (systematic.var() + residual.var())
                    if systematic.var() + residual.var() > 0
                    else 0.0
                ),
                "fitted_residual_correlation": (
                    np.corrcoef(systematic, residual)[0, 1]
                    if systematic.std() > 0 and residual.std() > 0
                    else 0.0
                ),
            }
        )
        exposures.append({"ticker": ticker, **exposure})
    return (
        pd.DataFrame(records).set_index("ticker"),
        pd.DataFrame(exposures).set_index("ticker"),
        fitted_frame,
        residual_frame,
    )


def _factor_variance_decomposition(
    returns: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_factors: pd.DataFrame,
    test_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate fitted systematic variance across full nonlinear factor components."""
    allocations = {}
    exposures = {}
    for ticker in returns:
        exposure = fit_nonlinear_asset_exposure(train_returns[ticker], train_factors)
        common = returns[ticker].index.intersection(test_factors.index)
        asset = returns.loc[common, ticker].to_numpy(dtype=float)
        factors = test_factors.loc[common]
        valid = np.isfinite(asset) & np.isfinite(factors.to_numpy()).all(axis=1)
        factors = factors.loc[valid]
        factor_values = factors.to_numpy()
        gamma = factor_values**2 - np.asarray(exposure["factor_squared_mean"])
        downside = np.minimum(factor_values, 0.0)
        components = np.stack(
            [
                factor_values[:, index] * exposure["beta_delta"][index]
                + gamma[:, index] * exposure["beta_gamma"][index]
                + downside[:, index] * exposure["beta_downside"][index]
                for index in range(factor_values.shape[1])
            ],
            axis=1,
        )
        fitted = components.sum(axis=1)
        variance = np.var(fitted)
        allocations[ticker] = (
            np.cov(components, fitted, rowvar=False)[:-1, -1] / variance
            if variance > 0
            else np.zeros(components.shape[1])
        )
        exposures[ticker] = exposure
    factor_names = [f"Factor {index + 1}" for index in range(train_factors.shape[1])]
    return pd.DataFrame.from_dict(allocations, orient="index", columns=factor_names), pd.DataFrame.from_dict(exposures, orient="index")


def _run_trend_following(returns: pd.DataFrame) -> object:
    """Run the rolling EWMA trend follower on the benchmark universe.

    The raw trend follower is compared against the factor-neutralised version
    (beta stripped against the anchor benchmark), so the residual strategy is
    long-only but market-neutral, exposing the idiosyncratic alpha.
    """
    anchor = BENCHMARK_ANCHOR
    if anchor not in returns:
        raise ValueError(f"The required {anchor} benchmark is absent from the data")
    model_df = returns.dropna()
    factor_returns = model_df[anchor]
    raw = run_trend_following(model_df, factor_returns, target_vol=0.10, max_leverage=1.25)
    neutral = run_neutral_following(model_df, factor_returns, target_vol=0.10, max_leverage=2.5)
    return raw, neutral


def main() -> None:
    args = _parse_args()
    data_dir = args.output.parent / "data" / "raw"
    figures_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_tickers = list(BENCHMARK_UNIVERSE)
    strategy_tickers = list(EXPLANATORY_UNIVERSE)
    all_tickers = model_tickers + strategy_tickers
    if not args.download and args.input.exists():
        returns = pd.read_csv(args.input, index_col=0, parse_dates=True)
        returns = returns.loc[args.start:args.end]
    else:
        returns = fetch_asset_data(all_tickers, args.start, args.end)
    output_data = (data_dir / "daily_returns.csv").resolve()
    if args.input.resolve() != output_data:
        returns.to_csv(output_data)

    metadata = {
        "start": args.start,
        "end": args.end,
        "downloaded_tickers": list(returns.columns),
        "model_universe": BENCHMARK_UNIVERSE,
        "explanatory_universe": EXPLANATORY_UNIVERSE,
        "note": "Tickers absent from the provider response are omitted from daily_returns.csv.",
    }
    (data_dir / "universe.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    available_model_tickers = [
        ticker for ticker in model_tickers
        if ticker in returns and returns[ticker].notna().any()
    ]
    labels = display_labels()
    if BENCHMARK_ANCHOR not in available_model_tickers:
        raise ValueError(f"The data provider did not return the required {BENCHMARK_ANCHOR} benchmark")
    model_returns = returns[available_model_tickers].dropna()
    benchmark_volatility = (
        model_returns[BENCHMARK_ANCHOR].ewm(alpha=0.01, adjust=False).std().fillna(0.0)
    )
    split = int(len(model_returns) * (1.0 - HOLDOUT_FRACTION))
    fitted_space, train_factors, test_factors = _fit_holdout_model(
        model_returns, benchmark_volatility, split, n_factors=N_FACTORS
    )
    train_returns = model_returns.iloc[:split]
    test_returns = model_returns.iloc[split:]
    available_strategy_tickers = [
        ticker for ticker in strategy_tickers
        if (
            ticker in returns
            and returns.loc[train_factors.index, ticker].notna().sum() > 0
            and returns.loc[test_factors.index, ticker].notna().sum() > 0
        )
    ]
    benchmark_diag, benchmark_exposures, benchmark_fitted, benchmark_residuals = _linear_holdout_attribution(
        model_returns, train_returns, train_factors, test_factors
    )
    factor_decomposition, _ = _factor_variance_decomposition(
        model_returns, train_returns, train_factors, test_factors
    )
    strategy_returns = returns[available_strategy_tickers]
    strategy_train = strategy_returns.loc[train_factors.index]
    strategy_diag, strategy_exposures, strategy_fitted, strategy_residuals = _linear_holdout_attribution(
        strategy_returns, strategy_train, train_factors, test_factors
    )
    strategy_factor_decomposition, _ = _factor_variance_decomposition(
        strategy_returns, strategy_train, train_factors, test_factors
    )
    benchmark_diag.to_csv(args.output / "benchmark_oos_diagnostics.csv")
    benchmark_exposures.to_csv(args.output / "benchmark_train_exposures.csv")
    benchmark_residuals.to_csv(args.output / "benchmark_oos_residuals.csv")
    benchmark_residuals.corr().to_csv(args.output / "benchmark_oos_residual_correlation.csv")
    factor_decomposition.to_csv(args.output / "benchmark_factor_variance_decomposition.csv")
    strategy_diag.to_csv(args.output / "strategy_oos_diagnostics.csv")
    strategy_exposures.to_csv(args.output / "strategy_train_exposures.csv")
    strategy_residuals.to_csv(args.output / "strategy_oos_residuals.csv")
    strategy_fitted.to_csv(args.output / "strategy_oos_fitted.csv")
    strategy_factor_decomposition.to_csv(
        args.output / "strategy_factor_variance_decomposition.csv"
    )
    test_factors.to_csv(args.output / "oos_latent_factor_moves.csv")
    pd.DataFrame(fitted_space, index=model_returns.columns, columns=train_factors.columns).to_csv(
        args.output / "frozen_factor_loadings.csv"
    )
    benchmark_corr = train_returns.corr().fillna(0.0)
    cluster_order = leaves_list(linkage(1.0 - benchmark_corr.to_numpy(), method="average"))
    pd.DataFrame({"ticker": benchmark_corr.columns[cluster_order], "cluster_order": np.arange(len(cluster_order))}).to_csv(
        args.output / "benchmark_cluster_order.csv", index=False
    )

    plot_cumulative_returns(
        test_returns,
        "Benchmark assets: untouched holdout performance",
        figures_dir / "model_cumulative_returns.html",
        labels=labels,
    )
    plot_cumulative_returns(
        strategy_returns.loc[test_factors.index],
        "Other assets: untouched holdout performance",
        figures_dir / "strategy_cumulative_returns.html",
        labels=labels,
    )
    plot_correlation_heatmap(
        model_returns.loc[test_factors.index],
        "Benchmark assets: holdout correlations",
        figures_dir / "model_correlation.html",
        labels=labels,
    )
    plot_correlation_heatmap(
        benchmark_residuals,
        "Benchmark assets: holdout residual correlations",
        figures_dir / "benchmark_residual_correlation.html",
        labels=labels,
    )
    plot_systematic_vs_idio(
        benchmark_diag.sort_values("idiosyncratic_volatility"),
        figures_dir / "benchmark_residual_volatility.html",
        title="Benchmark assets: holdout residual volatility",
        labels=labels,
    )
    plot_loadings(
        fitted_space,
        model_returns.columns.tolist(),
        "Training-only PCA reference loadings",
        figures_dir / "reference_loadings.html",
        labels=labels,
    )
    plot_capture_diagnostics(
        benchmark_diag,
        "Benchmark assets: out-of-sample factor capture",
        figures_dir / "benchmark_oos_capture.html",
        labels=labels,
    )
    plot_factor_decomposition(
        factor_decomposition,
        "EEM",
        figures_dir / "benchmark_factor_decomposition.html",
        labels=labels,
    )
    strategy_highlight = next(
        (ticker for ticker in ["DBMF", "RPAR", "PRPFX", "MNA"] if ticker in strategy_factor_decomposition),
        strategy_factor_decomposition.index[0],
    )
    plot_factor_decomposition(
        strategy_factor_decomposition,
        strategy_highlight,
        figures_dir / "strategy_factor_decomposition.html",
        labels=labels,
    )
    plot_realized_vs_fitted(
        test_returns[[ticker for ticker in ["ACWI", "AGG", "GLD"] if ticker in test_returns]],
        benchmark_fitted[[ticker for ticker in ["ACWI", "AGG", "GLD"] if ticker in benchmark_fitted]],
        "Benchmark assets: realized versus fixed-factor fitted returns",
        figures_dir / "benchmark_realized_vs_fitted.html",
        labels=labels,
    )
    plot_factor_moves(test_factors, figures_dir / "oos_latent_factor_moves.html")
    plot_fund_factor_attribution(
        strategy_returns.loc[test_factors.index], strategy_fitted, strategy_residuals,
        test_factors, strategy_exposures, figures_dir / "fund_factor_attribution.html",
        labels=labels,
    )
    plot_capture_diagnostics(
        strategy_diag,
        "Other assets: out-of-sample systematic versus idiosyncratic risk",
        figures_dir / "other_oos_decomposition.html",
        labels=labels,
    )
    plot_realized_vs_fitted(
        strategy_returns.loc[test_factors.index][[ticker for ticker in ["PRPFX", "DBMF", "MNA"] if ticker in strategy_returns]],
        strategy_fitted[[ticker for ticker in ["PRPFX", "DBMF", "MNA"] if ticker in strategy_fitted]],
        "Other assets: realized versus fixed-factor fitted returns",
        figures_dir / "other_realized_vs_fitted.html",
        labels=labels,
    )

    # Rolling EWMA trend-following backtest: compare the raw benchmark assets
    # against the factor-neutralised version to show the cleaner risk profile.
    raw_backtest, neutral_backtest = _run_trend_following(model_returns)
    stats = pd.DataFrame([
        {
            "strategy": result.strategy,
            "target_vol": result.target_vol,
            "max_leverage": result.max_leverage,
            "annual_vol": result.annual_vol,
            "cagr": result.cagr,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "final_return": result.final_return,
            "annual_turnover": result.annual_turnover,
            "portfolio_age": result.portfolio_age,
        }
        for result in (raw_backtest, neutral_backtest)
    ]).set_index("strategy")
    stats.to_csv(args.output / "backtest_stats.csv")
    pd.DataFrame({
        "raw": raw_backtest.equity_curve,
        "neutral": neutral_backtest.equity_curve,
    }).to_csv(args.output / "backtest_equity.csv")
    fund_returns = strategy_returns.dropna()
    fund_raw_backtest = run_trend_following(
        fund_returns,
        model_returns.loc[fund_returns.index, BENCHMARK_ANCHOR],
        target_vol=0.10,
        max_leverage=1.25,
    )
    equal_weight = pd.DataFrame(
        1.0 / len(fund_returns.columns), index=fund_returns.index, columns=fund_returns.columns
    )
    equal_weight = weekly_rebalanced_weights(equal_weight, fund_returns).fillna(0.0)
    equal_weight.to_csv(args.output / "equal_weight_raw_fund_weights.csv")
    fund_raw_backtest.weights.fillna(0.0).to_csv(args.output / "trend_raw_fund_weights.csv")
    all_factors = pd.concat([train_factors, test_factors]).loc[model_returns.index]
    attribution = pd.concat({
        "raw": factor_attribution(raw_backtest.daily_returns, all_factors, model_returns["SPY"]),
        "neutral": factor_attribution(neutral_backtest.daily_returns, all_factors, model_returns["SPY"]),
    }, names=["strategy", "source"])
    attribution.to_csv(args.output / "backtest_factor_attribution.csv")
    plot_backtest_comparison(raw_backtest, neutral_backtest, figures_dir / "trend_following_comparison.html")
    plot_backtest_attribution(attribution, figures_dir / "trend_following_factor_attribution.html")
    alpha_backtest = run_trend_following(
        strategy_residuals.dropna(), model_returns.loc[strategy_residuals.dropna().index, BENCHMARK_ANCHOR],
        target_vol=0.10, max_leverage=2.5,
    )
    pd.DataFrame([{
        "strategy": "residual_alpha",
        "target_vol": alpha_backtest.target_vol,
        "max_leverage": alpha_backtest.max_leverage,
        "annual_vol": alpha_backtest.annual_vol,
        "cagr": alpha_backtest.cagr,
        "max_drawdown": alpha_backtest.max_drawdown,
        "sharpe": alpha_backtest.sharpe,
        "final_return": alpha_backtest.final_return,
        "annual_turnover": alpha_backtest.annual_turnover,
        "portfolio_age": alpha_backtest.portfolio_age,
    }]).set_index("strategy").to_csv(args.output / "alpha_backtest_stats.csv")
    alpha_backtest.equity_curve.ffill().fillna(1.0).to_csv(
        args.output / "alpha_backtest_equity.csv", header=["residual_alpha"]
    )
    alpha_backtest.weights.fillna(0.0).to_csv(args.output / "trend_residual_fund_weights.csv")
    hedge_loadings = pd.read_csv(args.output / "frozen_factor_loadings.csv", index_col=0)
    _, benchmark_volatility = standardize_returns_ewma(model_returns)
    benchmark_mean = model_returns.ewm(alpha=0.01, adjust=False).mean().reindex(alpha_backtest.weights.index)
    benchmark_volatility = benchmark_volatility.reindex(alpha_backtest.weights.index)
    factor_betas = strategy_exposures.drop(columns="alpha").rename(
        columns=lambda column: column.removeprefix("beta_")
    ).reindex(columns=hedge_loadings.columns, fill_value=0.0)
    fund_weights = alpha_backtest.weights.fillna(0.0).reindex(columns=strategy_exposures.index, fill_value=0.0)
    hedge_beta = fund_weights.to_numpy() @ factor_betas.to_numpy()
    hedge_beta = pd.DataFrame(hedge_beta, index=alpha_backtest.weights.index, columns=factor_betas.columns)
    factor_to_standardized_benchmark = np.linalg.pinv(hedge_loadings.to_numpy())
    benchmark_hedge_values = (
        -hedge_beta.to_numpy() @ factor_to_standardized_benchmark
        / benchmark_volatility.to_numpy()
    )
    benchmark_hedges = pd.DataFrame(
        benchmark_hedge_values,
        index=hedge_beta.index,
        columns=hedge_loadings.index,
    )
    benchmark_hedges = benchmark_hedges.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cash_hedge = pd.Series(
        (
            hedge_beta.to_numpy() @ factor_to_standardized_benchmark
            * benchmark_mean.to_numpy() / benchmark_volatility.to_numpy()
        ).sum(axis=1),
        index=hedge_beta.index,
        name="Cash",
    )
    benchmark_hedges["Cash"] = cash_hedge.fillna(0.0)
    hedge_returns = model_returns.reindex(index=benchmark_hedges.index, columns=benchmark_hedges.columns).copy()
    hedge_returns["Cash"] = 0.0
    benchmark_hedges = weekly_rebalanced_weights(benchmark_hedges, hedge_returns)
    combined_weights = pd.concat([fund_weights, benchmark_hedges], axis=1)
    combined_returns = pd.concat([strategy_residuals.reindex(combined_weights.index), hedge_returns.reindex(combined_weights.index)], axis=1)
    combined_returns = combined_returns.loc[:, ~combined_returns.columns.duplicated()].fillna(0.0)
    combined_daily = (combined_weights * combined_returns.reindex(columns=combined_weights.columns)).sum(axis=1)
    combined_volatility = combined_daily.ewm(alpha=0.06, adjust=False).std(bias=False).mul(np.sqrt(252.0)).shift(1)
    portfolio_scale = (0.10 / combined_volatility.replace(0.0, np.nan)).clip(upper=2.5).fillna(1.0)
    gross_scale = (2.5 / combined_weights.abs().sum(axis=1).replace(0.0, np.nan)).fillna(1.0)
    portfolio_scale = pd.concat([portfolio_scale, gross_scale], axis=1).min(axis=1)
    combined_weights = combined_weights.mul(portfolio_scale, axis=0)
    fund_weights = combined_weights.loc[:, fund_weights.columns]
    benchmark_hedges = combined_weights.loc[:, benchmark_hedges.columns]
    combined_daily = (combined_weights * combined_returns.reindex(columns=combined_weights.columns)).sum(axis=1)
    combined_equity = (1.0 + combined_daily).cumprod()
    alpha_backtest = BacktestResult("residual_alpha", combined_equity, combined_daily, 0.10, 2.5, combined_weights)
    pd.DataFrame([{
        "strategy": "residual_alpha",
        "target_vol": alpha_backtest.target_vol,
        "max_leverage": alpha_backtest.max_leverage,
        "annual_vol": alpha_backtest.annual_vol,
        "cagr": alpha_backtest.cagr,
        "max_drawdown": alpha_backtest.max_drawdown,
        "sharpe": alpha_backtest.sharpe,
        "final_return": alpha_backtest.final_return,
        "annual_turnover": alpha_backtest.annual_turnover,
        "portfolio_age": alpha_backtest.portfolio_age,
    }]).set_index("strategy").to_csv(args.output / "alpha_backtest_stats.csv")
    alpha_backtest.equity_curve.ffill().fillna(1.0).to_csv(args.output / "alpha_backtest_equity.csv", header=["residual_alpha"])
    combined_daily.to_csv(args.output / "residual_alpha_portfolio_returns.csv", header=["residual_alpha_portfolio"])
    fund_weights.to_csv(args.output / "trend_residual_fund_weights.csv")
    benchmark_hedges.to_csv(args.output / "trend_residual_benchmark_hedge_exposures.csv")
    plot_backtest_comparison(alpha_backtest, alpha_backtest, figures_dir / "residual_alpha_trend_follower.html", labels={"Raw trend follower": "Residual alpha", "Neutral trend follower": "Residual alpha"})

    source = "Downloaded" if args.download else "Loaded"
    print(f"{source} columns: {', '.join(returns.columns)}")
    print(f"Model observations after complete-case filtering: {len(model_returns)}")
    print(f"Training observations: {len(train_returns)}; holdout observations: {len(test_returns)}")
    print("Holdout split: 70% training / 30% evaluation; no future observations used for fitting")
    print(f"Wrote figures to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
