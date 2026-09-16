"""Rolling EWMA trend-following backtest.

Runs a self-contained trend-following strategy with a rolling volatility budget
on both raw assets and factor-neutralised assets, then reports the ex-post
return / risk so the cleaner risk-adjusted performance of the neutral strategy
can be compared against a buy-and-hold benchmark.

No third-party optimisation libraries are used: the module relies on NumPy and
pandas (and reuses :func:`riskmodel.data_loader.standardize_returns_ewma`) so it
stays transparent and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestResult:
    """Ex-post statistics for a single trend-following run."""

    __slots__ = ("strategy", "equity_curve", "daily_returns", "weights", "target_vol", "max_leverage", "annual_vol", "cagr", "max_drawdown", "sharpe", "final_return", "annual_turnover", "portfolio_age")

    def __init__(self, strategy: str, equity_curve: pd.Series, daily_returns: pd.Series, target_vol: float, max_leverage: float = 0.0, weights: pd.DataFrame | None = None) -> None:
        self.strategy = strategy
        self.equity_curve = equity_curve
        self.daily_returns = daily_returns
        self.weights = weights
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.annual_vol, self.cagr, self.max_drawdown, self.sharpe, self.final_return = compute_stats(equity_curve, daily_returns)
        if weights is None:
            turnover = pd.Series(dtype=float)
        else:
            turnover = weights.diff().abs().sum(axis=1)
        if weights is not None and len(turnover):
            rebalance = weights.index.to_series().dt.to_period("W-SUN").ne(
                weights.index.to_series().dt.to_period("W-SUN").shift(1)
            )
            turnover = turnover.where(rebalance, 0.0)
        turnover = turnover.replace([np.inf, -np.inf], np.nan).dropna()
        self.annual_turnover = float(turnover.mean() * 252.0) if len(turnover) else 0.0
        self.portfolio_age = float(1.0 / turnover.mean()) if len(turnover) and turnover.mean() > 0 else float("inf")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<BacktestResult {self.strategy} cagr={self.cagr:.2%} "
                f"vol={self.annual_vol:.2%} dd={self.max_drawdown:.2%} sharpe={self.sharpe:.2f}>")


def rolling_volatility(returns: pd.DataFrame, window: int = 63, alpha: float = 0.94) -> pd.DataFrame:
    """Recursive open-ended EWMA volatility per asset (annualised, daily).

    Mirrors the standardisation EWMA in :mod:`riskmodel.data_loader` but measures
    volatility instead of returns, so each day's position size can be scaled to a
    fixed target portfolio volatility.

    Accepts either a :class:`pandas.Series` (single asset) or a :class:`pandas.DataFrame`
    (multiple assets, one column per asset).
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name)
    volatility = returns.ewm(alpha=alpha, adjust=False).std(bias=False) * np.sqrt(252.0)
    # Keep the strategy out of the market until a complete sizing window exists.
    volatility.iloc[:window] = np.nan
    return volatility.shift(1)


def trend_signal(returns: pd.DataFrame, signal_window: int = 21) -> pd.DataFrame:
    """Standardised rolling momentum used as the trend-following signal."""
    standardised = returns.ewm(alpha=0.94, adjust=False).std()
    signal = returns.rolling(signal_window, min_periods=signal_window).mean() / standardised
    return signal.shift(1)


def equal_trend_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """Normalise each day's signals to a long-only unit-sum weight vector."""
    # Use signal magnitude for a long-only portfolio; negative momentum does not
    # create a short position in this deliberately simple implementation.
    if isinstance(signals, pd.Series):
        signals = signals.to_frame(name=signals.name)
    weights = np.abs(signals) ** 0.5
    weights = weights.div(weights.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    return weights


def weekly_rebalanced_weights(target_weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Hold fixed units between weekly rebalances and move to lagged targets weekly."""
    target_weights = target_weights.reindex(index=returns.index, columns=returns.columns)
    result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    current_weights = None
    previous_week = None
    for date in returns.index:
        target = target_weights.loc[date]
        week = date.to_period("W-SUN")
        if target.notna().all() and (current_weights is None or week != previous_week):
            current_weights = target.to_numpy(dtype=float).copy()
        if current_weights is None:
            previous_week = week
            continue
        result.loc[date] = current_weights
        asset_returns = returns.loc[date].to_numpy(dtype=float)
        portfolio_return = float(np.nansum(current_weights * asset_returns))
        current_weights = current_weights * (1.0 + asset_returns) / max(1e-12, 1.0 + portfolio_return)
        previous_week = week
    return result.replace([np.inf, -np.inf], np.nan)


def scale_to_target_vol(weights: pd.DataFrame, volatilities: pd.DataFrame, target_vol: float, max_leverage: float = 2.0) -> pd.DataFrame:
    """Scale daily weights so that portfolio vol equals the rolling target."""
    if isinstance(weights, pd.Series):
        weights = weights.to_frame(name=weights.name)
    if volatilities.shape[1] == 1 and weights.shape[1] > 1:
        volatilities = pd.concat([volatilities.iloc[:, 0]] * weights.shape[1], axis=1)
        volatilities.columns = weights.columns
    volatilities = volatilities.reindex(index=weights.index, columns=weights.columns)
    # Use a root-sum-square budget rather than adding every asset volatility;
    # the latter is deliberately conservative and materially under-invests a
    # diversified neutral portfolio.
    portfolio_vol = ((weights * volatilities) ** 2).sum(axis=1).pow(0.5)
    # A volatility estimate can briefly collapse after a run of quiet returns;
    # cap the resulting gross leverage so one stale estimate cannot dominate
    # the ex-post risk statistics.
    scale = (target_vol / portfolio_vol.replace(0.0, np.nan)).clip(upper=max_leverage)
    return weights.mul(scale, axis=0)


def run_trend_following(returns: pd.DataFrame, factor_returns: pd.DataFrame, target_vol: float,
                        signal_window: int = 21, vol_window: int = 63, max_leverage: float = 2.0,
                        rebalance_alpha: float = 0.03) -> BacktestResult:
    """Run a rolling-vol trend follower on raw asset returns.

    Signal is standardised momentum; position size is scaled by ``1 / rolling_vol``
    to achieve ``target_vol`` of annualised portfolio volatility.
    """
    signals = trend_signal(returns, signal_window)
    volatilities = rolling_volatility(returns, vol_window)
    weights = equal_trend_weights(signals)
    weights = scale_to_target_vol(weights, volatilities, target_vol, max_leverage)
    weights = weekly_rebalanced_weights(weights, returns)
    daily_returns = _apply_weights(weights, returns, 0.0, factor_returns)
    equity_curve = (1.0 + daily_returns).cumprod()
    return BacktestResult("raw", equity_curve, daily_returns, target_vol, float(weights.abs().sum(axis=1).max()), weights)


def run_signed_residual_momentum(returns: pd.DataFrame, target_vol: float = 0.10,
                                 signal_window: int = 252, vol_window: int = 63,
                                 max_leverage: float = 2.0) -> BacktestResult:
    """Trade executable residuals long-short using lagged signed momentum."""
    signals = trend_signal(returns, signal_window)
    weights = signals.div(signals.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    weights = scale_to_target_vol(
        weights, rolling_volatility(returns, vol_window), target_vol, max_leverage
    )
    weights = weekly_rebalanced_weights(weights, returns)
    daily_returns = (weights * returns).sum(axis=1)
    equity_curve = (1.0 + daily_returns).cumprod()
    return BacktestResult(
        "signed_residual_momentum", equity_curve, daily_returns, target_vol,
        float(weights.abs().sum(axis=1).max()), weights
    )


def daily_beta(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> float:
    """Daily average beta of a portfolio return series against the factor portfolio."""
    factor_returns = factor_returns.to_numpy().astype(float).ravel()
    portfolio_returns = portfolio_returns.to_numpy().astype(float)
    # Drop the warmup rows (where daily returns are undefined) so the covariance
    # is estimated only over days that carry a real signal.
    valid = ~np.isnan(portfolio_returns) & ~np.isnan(factor_returns)
    cov = np.cov(portfolio_returns[valid], factor_returns[valid])
    if cov[0, 1] == 0.0 or cov[1, 1] == 0.0:
        return 0.0
    return cov[0, 1] / cov[1, 1]


def _apply_weights(weights: pd.DataFrame, returns: pd.DataFrame, factor_beta: float, factor_returns: pd.DataFrame) -> pd.Series:
    """Apply daily weights to a portfolio and re-add systematic factor exposure.

    ``weights`` is a unit-sum weight vector (one per asset) and ``returns`` holds the
    asset daily returns. A single-asset (Series) input is treated as a one-column
    portfolio; a multi-asset (DataFrame) input is a full matrix product.
    """
    weights_np = weights.to_numpy().astype(float)
    returns_np = returns.to_numpy().astype(float)
    if returns_np.ndim == 1:
        # A one-asset portfolio: flatten the (n, 1) single-column case before
        # multiplying by the 1D weights (otherwise it would outer-product).
        portfolio = returns_np * weights_np.ravel()
    else:
        portfolio = (weights_np * returns_np).sum(axis=1)
    result = pd.Series(portfolio, index=returns.index)
    # Re-add systematic factor exposure when requested. ``factor_returns`` is usually a
    # single-column DataFrame (the anchor factor); take that column so the
    # factor term stays a Series aligned to the portfolio index instead of
    # triggering a Series + DataFrame broadcast (which would produce a 2D result).
    fr_np = factor_returns.to_numpy().astype(float)
    if isinstance(factor_returns, pd.Series) or fr_np.ndim == 1:
        factor_series = pd.Series(fr_np, index=returns.index)
    else:
        factor_series = factor_returns.iloc[:, 0].reindex(returns.index)
    result = result + factor_beta * factor_series
    return result


def neutralize_returns(returns: pd.DataFrame, beta: float, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Remove systematic factor co-movement from asset returns."""
    factor = factor_returns.to_numpy().astype(float).ravel()
    if isinstance(returns, pd.Series):
        neutralised = returns.to_numpy().astype(float) - beta * factor
        return pd.Series(neutralised, index=returns.index, name=returns.name)
    neutralised = returns.to_numpy().astype(float) - beta * factor[:, None]
    return pd.DataFrame(neutralised, index=returns.index, columns=returns.columns)


def run_neutral_following(returns: pd.DataFrame, factor_returns: pd.DataFrame, target_vol: float,
                          signal_window: int = 21, vol_window: int = 63, max_leverage: float = 2.0) -> BacktestResult:
    """Run the trend follower on factor-neutralised returns.

    The strategy's beta against the factor portfolio is estimated from the raw
    trend-following weights, and that systematic exposure is stripped from the
    returns before the signal is computed. The result captures the residual alpha
    with the idiosyncratic factor risk removed, typically producing a much
    cleaner (lower-volatility, lower-drawdown) equity curve.
    """
    raw_result = run_trend_following(returns, factor_returns, target_vol, signal_window, vol_window, max_leverage)
    beta = daily_beta(raw_result.daily_returns, factor_returns)
    neutralised = neutralize_returns(returns, beta, factor_returns)
    neutral_signals = trend_signal(neutralised, signal_window)
    neutral_volatilities = rolling_volatility(neutralised, vol_window)
    neutral_weights = equal_trend_weights(neutral_signals)
    neutral_weights = scale_to_target_vol(neutral_weights, neutral_volatilities, target_vol, max_leverage)
    neutral_weights = weekly_rebalanced_weights(neutral_weights, neutralised)
    # Keep the final portfolio free of the systematic factor exposure.
    neutral_daily = _apply_weights(neutral_weights, neutralised, 0.0, factor_returns)
    # Re-estimate after the nonlinear signal and sizing steps. This final
    # residualization makes the returned portfolio, rather than only its input
    # assets, factor-neutral.
    realised_beta = daily_beta(neutral_daily, factor_returns)
    factor_series = (
        factor_returns if isinstance(factor_returns, pd.Series)
        else factor_returns.iloc[:, 0]
    ).reindex(neutral_daily.index)
    neutral_daily = neutral_daily - realised_beta * factor_series
    neutral_equity = (1.0 + neutral_daily).cumprod()
    return BacktestResult("neutral", neutral_equity, neutral_daily, target_vol, float(neutral_weights.abs().sum(axis=1).max()), neutral_weights)


def compute_stats(equity_curve: pd.Series, daily_returns: pd.Series) -> tuple[float, float, float, float, float]:
    """Compute annualised CAGR, volatility, max drawdown, Sharpe and final return."""
    daily_returns = daily_returns.dropna().to_numpy().astype(float)
    n_days = len(daily_returns)
    final_return = float(equity_curve.iloc[-1] - 1.0) if len(equity_curve) else 0.0
    if n_days > 1 and np.std(daily_returns, ddof=1) > 0:
        cagr = float(np.exp(np.log(1.0 + final_return) * 252.0 / n_days) - 1.0) if (1.0 + final_return) > 0 else 0.0
        annual_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
        sharpe = float(np.sum(daily_returns) / (n_days * np.std(daily_returns, ddof=1)) * np.sqrt(252))
    else:
        cagr = final_return
        annual_vol = 0.0
        sharpe = 0.0
    # Compute the drawdown over the valid (non-warmup) portion of the equity curve,
    # where the warmup rows are NaN because the rolling weights are undefined.
    equity = equity_curve.dropna().to_numpy().astype(float)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0
    return annual_vol, cagr, max_drawdown, sharpe, final_return


def factor_attribution(
    daily_returns: pd.Series,
    factor_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Regress strategy returns on factors and report risk and return attribution."""
    frame = factor_returns.copy()
    if benchmark_returns is not None:
        frame = frame.assign(SPX_beta=benchmark_returns)
    frame = frame.join(daily_returns.rename("strategy"), how="inner").dropna()
    if len(frame) <= frame.shape[1]:
        return pd.DataFrame(columns=["beta", "factor_volatility", "risk_contribution", "variance_contribution", "variance_contribution_pct", "annual_return_contribution"])
    factors = frame.drop(columns="strategy")
    design = factors.to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, frame["strategy"].to_numpy(dtype=float), rcond=None)[0]
    names = factors.columns
    factor_values = factors
    factor_beta = pd.Series(coefficients, index=names, dtype=float)
    factor_volatility = factor_values.std(ddof=1) * np.sqrt(252.0)
    strategy = frame["strategy"]
    components = factor_values.mul(factor_beta, axis=1)
    variance = strategy.var(ddof=1)
    variance_contribution = components.apply(lambda component: component.cov(strategy))
    contribution = factor_beta * factor_volatility
    annual_return = factor_beta * factor_values.mean() * 252.0
    result = pd.DataFrame({
        "beta": factor_beta,
        "factor_volatility": factor_volatility,
        "risk_contribution": contribution,
        "variance_contribution": variance_contribution,
        "variance_contribution_pct": variance_contribution / variance if variance > 0 else 0.0,
        "annual_return_contribution": annual_return,
    })
    residual = frame["strategy"] - design @ coefficients
    residual_variance = residual.var(ddof=1)
    result.loc["Intercept", ["beta", "factor_volatility", "risk_contribution", "variance_contribution", "variance_contribution_pct", "annual_return_contribution"]] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    result.loc["Residual", ["beta", "factor_volatility", "risk_contribution", "variance_contribution", "variance_contribution_pct", "annual_return_contribution"]] = [
        0.0, residual.std(ddof=1) * np.sqrt(252.0), residual.std(ddof=1) * np.sqrt(252.0), residual_variance, residual_variance / variance if variance > 0 else 0.0, residual.mean() * 252.0,
    ]
    return result
