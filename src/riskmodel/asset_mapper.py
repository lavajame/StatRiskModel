"""Nonlinear asset exposure estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


def _design_matrix(
    factors: np.ndarray, factor_squared_mean: np.ndarray | None = None
) -> np.ndarray:
    if factor_squared_mean is None:
        factor_squared_mean = np.mean(factors**2, axis=0)
    gamma_basis = factors**2 - factor_squared_mean
    return np.hstack((factors, gamma_basis, np.minimum(factors, 0.0)))


def fit_nonlinear_asset_exposure(
    asset_returns: pd.Series, factor_returns: pd.DataFrame, alpha: float = 1e-3
) -> dict[str, object]:
    """Fit delta, gamma, and downside exposures with a ridge-regularized basis."""
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    common_index = asset_returns.index.intersection(factor_returns.index)
    if len(common_index) == 0:
        raise ValueError("asset_returns and factor_returns have no common dates")

    y = asset_returns.loc[common_index].to_numpy(dtype=float)
    factors = factor_returns.loc[common_index].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(factors).all(axis=1)
    if not valid.any():
        raise ValueError("asset_returns and factor_returns have no finite common observations")
    y = y[valid]
    factors = factors[valid]
    design = _design_matrix(factors)
    coefficients = linalg.solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ y,
        assume_a="pos",
    )
    predictions = design @ coefficients
    residual_sum = np.sum((y - predictions) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    n_factors = factors.shape[1]
    return {
        "alpha": 0.0,
        "beta_delta": coefficients[:n_factors],
        "beta_gamma": coefficients[n_factors : 2 * n_factors],
        "beta_downside": coefficients[2 * n_factors :],
        "factor_squared_mean": np.mean(factors**2, axis=0),
        "r_squared": float(r_squared),
    }


def predict_nonlinear_asset_exposure(
    factor_returns: pd.DataFrame, exposure: dict[str, object]
) -> pd.Series:
    """Predict asset returns from fixed nonlinear exposures on new factor data."""
    factors = factor_returns.to_numpy(dtype=float)
    if not np.isfinite(factors).all():
        raise ValueError("factor_returns must contain only finite values")
    design = _design_matrix(
        factors, np.asarray(exposure["factor_squared_mean"], dtype=float)
    )
    coefficients = np.concatenate(
        [exposure["beta_delta"], exposure["beta_gamma"], exposure["beta_downside"]]
    )
    predictions = design @ coefficients
    return pd.Series(predictions, index=factor_returns.index, name="fitted")
