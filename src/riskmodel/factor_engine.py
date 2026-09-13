"""PCA factor extraction, Procrustes alignment, and daily factor tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


def _top_eigenvectors(correlation: np.ndarray, n_factors: int) -> np.ndarray:
    eigenvalues, eigenvectors = linalg.eigh(correlation)
    indices = np.argsort(eigenvalues)[::-1][:n_factors]
    return eigenvectors[:, indices]


def fit_global_pca(std_returns: pd.DataFrame, n_factors: int = 5) -> np.ndarray:
    """Fit the reference loading space from the full standardized sample."""
    if std_returns.shape[1] < n_factors:
        raise ValueError("n_factors cannot exceed the number of assets")
    return _top_eigenvectors(std_returns.corr().to_numpy(), n_factors)


def align_procrustes(V_block: np.ndarray, V_star: np.ndarray) -> np.ndarray:
    """Align block loadings to the reference with an orthogonal rotation."""
    if V_block.shape != V_star.shape:
        raise ValueError("V_block and V_star must have the same shape")
    U, _, Vt = linalg.svd(V_block.T @ V_star)
    return V_block @ (U @ Vt)


def extract_block_loadings(
    std_returns: pd.DataFrame,
    V_star: np.ndarray,
    n_factors: int = 5,
    block_size: int = 252,
) -> list[np.ndarray]:
    """Extract and align loadings for each complete block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_blocks = len(std_returns) // block_size
    if n_blocks == 0:
        raise ValueError("std_returns does not contain a complete block")

    aligned_blocks = []
    for block_index in range(n_blocks):
        start = block_index * block_size
        block = std_returns.iloc[start : start + block_size]
        block_loadings = _top_eigenvectors(block.corr().to_numpy(), n_factors)
        aligned_blocks.append(align_procrustes(block_loadings, V_star))
    return aligned_blocks


def solve_daily_factors(
    std_returns: pd.DataFrame,
    V_low: np.ndarray,
    V_high: np.ndarray,
    scores: pd.Series,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Extract daily factor innovations with iteratively updated WLS weights."""
    if not std_returns.index.equals(scores.index):
        raise ValueError("scores must have the same index as std_returns")
    if V_low.shape != V_high.shape or V_low.shape[0] != std_returns.shape[1]:
        raise ValueError("loading matrices must match the asset cross-section")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    n_factors = V_low.shape[1]
    weights = np.ones(std_returns.shape[1])
    factor_values = np.empty((len(std_returns), n_factors))
    for row, score in enumerate(scores.to_numpy(dtype=float)):
        loading = (1.0 - score) * V_low + score * V_high
        observations = std_returns.iloc[row].to_numpy()
        weighted_loading = loading * weights[:, None]
        system = loading.T @ weighted_loading + ridge * np.eye(n_factors)
        factor = linalg.solve(system, weighted_loading.T @ observations, assume_a="pos")
        residual = observations - loading @ factor
        weights = 1.0 / (np.abs(residual) + 1e-4)
        factor_values[row] = factor

    columns = [f"Factor_{index + 1}" for index in range(n_factors)]
    return pd.DataFrame(factor_values, index=std_returns.index, columns=columns)
