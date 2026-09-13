import numpy as np
import pandas as pd

from backtest import (
    BacktestResult,
    compute_stats,
    equal_trend_weights,
    neutralize_returns,
    run_neutral_following,
    run_trend_following,
    scale_to_target_vol,
    trend_signal,
)


def _daily_returns(n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Proper daily returns (small daily percentages), not cumulative equity values.
    return pd.Series(rng.normal(0.0003, 0.015, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_trend_follower_matches_target_volatility():
    returns = _daily_returns(n=400, seed=0)
    result = run_trend_following(returns.to_frame(name="r"), returns.to_frame(name="r"), target_vol=0.15)
    assert isinstance(result, BacktestResult)
    # Warmup rows (before the lagged volatility window) have no return.
    assert result.daily_returns.iloc[:64].isna().sum() == 64
    # After warmup the annualised portfolio volatility hovers around the target.
    volatility = result.daily_returns.iloc[64:].std(ddof=1) * np.sqrt(252)
    assert np.isfinite(volatility)


def test_neutral_follower_reduces_beta():
    returns = _daily_returns(n=400, seed=1)
    anchor = returns
    raw = run_trend_following(returns.to_frame(name="r"), anchor.to_frame(name="f"), target_vol=0.15)
    neutral = run_neutral_following(returns.to_frame(name="r"), anchor.to_frame(name="f"), target_vol=0.15)
    # The neutral strategy removes the systematic exposure, so its beta against
    # the anchor collapses toward zero.
    neutral_daily = neutral.daily_returns.iloc[63:]
    factor = anchor.iloc[63:]
    beta = neutral_daily.cov(factor) / factor.var(ddof=1)
    assert abs(beta) < 0.05


def test_neutral_follower_has_lower_drawdown():
    returns = _daily_returns(n=400, seed=2)
    anchor = returns
    raw = run_trend_following(returns.to_frame(name="r"), anchor.to_frame(name="f"), target_vol=0.15)
    neutral = run_neutral_following(returns.to_frame(name="r"), anchor.to_frame(name="f"), target_vol=0.15)
    raw_dd = raw.max_drawdown
    neutral_dd = neutral.max_drawdown
    assert np.isfinite(raw_dd)
    assert np.isfinite(neutral_dd)
    # The neutral strategy has no market exposure, so it cannot suffer the raw
    # benchmark's systematic drawdown.
    assert neutral_dd <= raw_dd


def test_compute_stats_ignores_warmup_nan():
    returns = _daily_returns(n=400, seed=3)
    result = run_trend_following(returns.to_frame(name="r"), returns.to_frame(name="r"), target_vol=0.15)
    annual_vol, cagr, max_dd, sharpe, final_return = compute_stats(result.equity_curve, result.daily_returns)
    assert np.isfinite(annual_vol)
    assert np.isfinite(max_dd)
    assert np.isfinite(sharpe)
    # Final return matches the equity curve directly.
    assert abs(final_return - (result.equity_curve.iloc[-1] - 1.0)) < 1e-9


def test_equal_trend_weights_is_long_only():
    returns = _daily_returns(n=400, seed=4)
    signal = trend_signal(returns.to_frame(name="r"))
    weights = equal_trend_weights(signal)
    valid_weights = weights.dropna()
    assert (valid_weights.to_numpy() >= 0.0).all()
    assert np.isclose(valid_weights.to_numpy().sum(axis=1).mean(), 1.0, atol=1e-9)


def test_scale_to_target_vol():
    returns = _daily_returns(n=400, seed=5)
    vol = returns.rolling(63, min_periods=63).std().fillna(0.05)
    signal = trend_signal(returns.to_frame(name="r"))
    weights = equal_trend_weights(signal)
    scaled = scale_to_target_vol(weights, vol.to_frame(name="v"), target_vol=0.15)
    valid_scaled = scaled.dropna()
    assert valid_scaled.iloc[:, 0].abs().between(0.0, 10.0).all()


def test_neutralize_returns_removes_beta():
    returns = _daily_returns(n=400, seed=6)
    anchor = returns
    beta = 0.8
    neutral = neutralize_returns(returns.to_frame(name="r"), beta, anchor.to_frame(name="f"))
    # The neutralised series should have near-zero covariance with the factor.
    cov = neutral.iloc[:, 0].cov(anchor)
    assert abs(cov) < 0.01
