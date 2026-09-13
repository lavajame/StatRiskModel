"""End-to-end orchestration for the statistical factor risk model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import standardize_returns_ewma
from .factor_engine import (
    extract_block_loadings,
    fit_global_pca,
    solve_daily_factors,
)
from .regime import compute_regime_scores, construct_tilted_loadings


@dataclass(frozen=True)
class FactorRiskModelResult:
    """Outputs produced by a fitted historical factor pipeline."""

    standardized_returns: pd.DataFrame
    ewma_volatility: pd.DataFrame
    reference_loadings: np.ndarray
    aligned_block_loadings: list[np.ndarray]
    low_stress_loadings: np.ndarray
    high_stress_loadings: np.ndarray
    regime_scores: pd.Series
    factor_returns: pd.DataFrame


def run_factor_pipeline(
    returns: pd.DataFrame,
    benchmark_volatility: pd.Series,
    n_factors: int = 8,
    block_size: int = 252,
    decay_factor: float = 0.99,
    regime_gamma: float = 10.0,
) -> FactorRiskModelResult:
    """Fit the full model from historical returns and a benchmark volatility series."""
    if not returns.index.equals(benchmark_volatility.index):
        raise ValueError("benchmark_volatility must have the same index as returns")
    if n_factors <= 0:
        raise ValueError("n_factors must be positive")

    standardized, volatility = standardize_returns_ewma(returns, decay_factor)
    benchmark = benchmark_volatility.loc[standardized.index]
    reference = fit_global_pca(standardized, n_factors)
    blocks = extract_block_loadings(standardized, reference, n_factors, block_size)

    block_regimes = np.array([
        benchmark.iloc[index * block_size : (index + 1) * block_size].mean()
        for index in range(len(blocks))
    ])
    low, high = construct_tilted_loadings(blocks, block_regimes)
    scores = compute_regime_scores(benchmark, regime_gamma)
    factors = solve_daily_factors(standardized, low, high, scores)
    return FactorRiskModelResult(
        standardized_returns=standardized,
        ewma_volatility=volatility,
        reference_loadings=reference,
        aligned_block_loadings=blocks,
        low_stress_loadings=low,
        high_stress_loadings=high,
        regime_scores=scores,
        factor_returns=factors,
    )
