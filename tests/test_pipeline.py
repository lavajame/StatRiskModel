import numpy as np
import pandas as pd

from riskmodel import (
    align_procrustes,
    compute_regime_scores,
    construct_tilted_loadings,
    extract_block_loadings,
    fit_global_pca,
    fit_nonlinear_asset_exposure,
    run_factor_pipeline,
    solve_daily_factors,
    standardize_returns_ewma,
)


def test_end_to_end_synthetic_pipeline():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=504, freq="B")
    latent = rng.normal(size=(504, 3))
    exposures = rng.normal(size=(6, 3))
    returns = latent @ exposures.T + 0.05 * rng.normal(size=(504, 6))
    raw = pd.DataFrame(returns / 100, index=dates, columns=[f"Asset_{i}" for i in range(6)])

    standardized, volatility = standardize_returns_ewma(raw)
    reference = fit_global_pca(standardized, n_factors=3)
    blocks = extract_block_loadings(standardized, reference, n_factors=3, block_size=252)
    assert np.allclose(blocks[0].T @ blocks[0], np.eye(3), atol=1e-6)
    assert np.allclose(align_procrustes(blocks[0], reference).T @ align_procrustes(blocks[0], reference), np.eye(3), atol=1e-6)

    block_scores = np.array([0.2, 0.8])
    low, high = construct_tilted_loadings(blocks, block_scores)
    scores = compute_regime_scores(volatility.mean(axis=1))
    factors = solve_daily_factors(standardized, low, high, scores)
    exposure = fit_nonlinear_asset_exposure(raw.iloc[:, 0], factors)
    result = run_factor_pipeline(raw, volatility.mean(axis=1), n_factors=3)

    assert factors.shape == (len(standardized), 3)
    assert np.isfinite(factors.to_numpy()).all()
    assert set(exposure) == {
        "alpha",
        "beta_delta",
        "beta_gamma",
        "beta_downside",
        "factor_squared_mean",
        "r_squared",
    }
    assert result.factor_returns.shape == factors.shape
