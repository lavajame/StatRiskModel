"""Continuous market regime scores and regime-specific loading baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_scores(
    volatility: pd.Series,
    gamma: float = 10.0,
    calibration_mean: float | None = None,
    calibration_std: float | None = None,
) -> pd.Series:
    """Map volatility into [0, 1], optionally using frozen calibration values."""
    if volatility.empty:
        raise ValueError("volatility must contain observations")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    center = volatility.mean() if calibration_mean is None else calibration_mean
    scale = volatility.std() if calibration_std is None else calibration_std
    normalized = (volatility - center) / (scale if scale > 0 else 1.0)
    return pd.Series(
        1.0 / (1.0 + np.exp(-gamma * normalized)), index=volatility.index
    )


def construct_tilted_loadings(
    aligned_blocks: list[np.ndarray], block_regimes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average aligned blocks below and above the median regime."""
    if len(aligned_blocks) == 0 or len(aligned_blocks) != len(block_regimes):
        raise ValueError("aligned_blocks and block_regimes must be non-empty and paired")
    median = float(np.median(block_regimes))
    low = np.asarray(block_regimes) <= median
    high = ~low
    if not low.any() or not high.any():
        raise ValueError("block_regimes must contain both low and high states")
    return np.mean(np.asarray(aligned_blocks)[low], axis=0), np.mean(
        np.asarray(aligned_blocks)[high], axis=0
    )
