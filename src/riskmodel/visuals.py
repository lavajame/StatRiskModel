"""Dependency-free interactive HTML visuals for the factor model."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestResult


_STYLE = """
:root { color-scheme: light; --ink:#0f1b24; --ink-soft:#3b4a57; --muted:#8895a5; --grid:#e6edf3; --border:#e2e8ef; --accent:#176b87; --accent-2:#0e8a9b; --warm:#d95f59; --good:#2f855a; --shadow:0 14px 40px #0f1b2424; }
* { box-sizing:border-box; }
html { -webkit-font-smoothing:antialiased; }
body { margin:0; background:radial-gradient(1200px 600px at 80% -10%, #eef5f8 0%, #f5f7f8 55%); color:var(--ink); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width:1120px; margin:34px auto; padding:0 24px 60px; }
h1 { font-size:30px; margin:0 0 6px; letter-spacing:-.02em; font-weight:700; }
.subtitle { color:var(--muted); margin:0 0 22px; font-size:14px; }
h2 { font-size:13px; margin:30px 0 12px; color:var(--muted); font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.panel { background:linear-gradient(180deg,#ffffff,#fcfdff); border:1px solid var(--border); border-radius:14px; padding:18px 20px; box-shadow:var(--shadow); }
.legend { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 16px; }
.legend button, .toolbar button { border:1px solid var(--border); background:rgba(255,255,255,.7); backdrop-filter:blur(4px); border-radius:999px; padding:7px 13px; color:var(--ink); cursor:pointer; font-size:13px; font-weight:500; transition:all .15s ease; }
.legend button:hover, .toolbar button:hover { border-color:var(--accent); background:#ffffff; transform:translateY(-1px); box-shadow:0 4px 12px #0f1b2418; }
.legend button.off { opacity:.42; text-decoration:line-through; background:#eef1f4; }
svg { width:100%; min-width:640px; height:auto; display:block; }
.grid { stroke:var(--grid); stroke-width:1; }
.axis { fill:var(--muted); font-size:11px; }
.axis-label { fill:var(--ink-soft); font-size:11px; font-weight:600; }
.grid-value { fill:var(--muted); font-size:10px; }
.path { fill:none; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }
.dot { stroke:#ffffff; stroke-width:1.5; }
.band { stroke:#ffffff; stroke-width:1; opacity:.6; }
.area { stroke:none; }
.note { color:var(--muted); margin:0 0 14px; font-size:13px; }
.table-wrap { overflow:auto; border-radius:10px; border:1px solid var(--border); }
table { border-collapse:collapse; width:100%; min-width:640px; font-size:13px; }
th, td { border-bottom:1px solid #eef1f4; padding:9px 12px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; background:#fafbfc; position:sticky; top:0; }
.heat td { min-width:96px; text-align:center; transition:transform .12s ease, box-shadow .12s ease; cursor:pointer; }
.heat td:hover { outline:2.5px solid var(--accent); transform:scale(1.05); box-shadow:0 6px 20px #0f1b241e; z-index:2; position:relative; }
.bar { height:14px; border-radius:3px; min-width:3px; transition:width .3s ease; }
.bar.warm { background:var(--warm); }
.bar.good { background:var(--good); }
.bar.band { background:#eef1f4; }
.metric { display:grid; grid-template-columns:minmax(160px,1.1fr) 1fr 120px; gap:12px; align-items:center; margin:8px 0; }
.metric .name { overflow:hidden; text-overflow:ellipsis; font-weight:500; }
.metric .rank { color:var(--accent); font-size:11px; font-weight:700; margin-right:8px; }
.small { color:var(--muted); font-size:12px; }
.kpi { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:0 0 24px; }
.kpi .card { background:linear-gradient(180deg,#ffffff,#f6f9fb); border:1px solid var(--border); border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); text-align:center; }
.kpi .kpi-value { font-size:24px; font-weight:700; letter-spacing:-.02em; }
.kpi .kpi-label { color:var(--muted); font-size:12px; margin-top:2px; }
@media (max-width:760px) { main { margin:16px auto; padding:0 12px 28px; } h1 { font-size:23px; } .panel { padding:12px; } }
"""

_SCRIPT = """
function toggleSeries(button) {
  const key = button.dataset.series;
  button.classList.toggle('off');
  document.querySelectorAll('[data-line="' + CSS.escape(key) + '"]').forEach(el => el.classList.toggle('hidden'));
}
function sortMetrics() {
  const box = document.querySelector('[data-metrics]');
  if (!box) return;
  box.querySelectorAll('.metric').sort((a, b) => {
    const ta = parseFloat(a.querySelector('.small').dataset.value || '0');
    const tb = parseFloat(b.querySelector('.small').dataset.value || '0');
    return tb - ta;
  });
}
function sortAsc() {
  document.querySelector('[data-sort]')?.classList.toggle('off');
  const box = document.querySelector('[data-metrics]');
  if (!box) return;
  box.querySelectorAll('.metric').sort((a, b) => {
    const ta = parseFloat(a.querySelector('.small').dataset.value || '0');
    const tb = parseFloat(b.querySelector('.small').dataset.value || '0');
    return ta - tb;
  });
  box.querySelectorAll('.metric').forEach(row => box.appendChild(row));
}
function animateBars() {
  document.querySelectorAll('[data-bar]').forEach(bar => {
    const target = parseFloat(bar.dataset.width);
    bar.style.width = '0%';
    requestAnimationFrame(() => { bar.style.transition = 'width .8s cubic-bezier(.2,.8,.2,1)'; bar.style.width = target + '%'; });
  });
}
"""


def _label(value: str, labels: dict[str, str] | None) -> str:
    return (labels or {}).get(value, value)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _document(title: str, body: str) -> str:
    return f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{_esc(title)}</title><style>{_STYLE}</style></head><body><main><h1>{_esc(title)}</h1>{body}</main><script>{_SCRIPT}</script></body></html>"


def _write(title: str, body: str, output_path: Path) -> None:
    output_path.write_text(_document(title, body), encoding="utf-8")


def _sample(values: np.ndarray, maximum: int = 480) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).astype(int)
    return values[indices]


def _line_chart(series: dict[str, np.ndarray], title: str, labels: dict[str, str] | None = None, ytitle: str = "Value") -> str:
    colors = ["#176b87", "#d95f59", "#5c6bc0", "#c28e0e", "#2f855a", "#8b5cf6", "#c2410c", "#475569", "#0f766e", "#be185d"]
    sampled = {key: _sample(np.asarray(value, dtype=float)) for key, value in series.items()}
    finite = np.concatenate([value[np.isfinite(value)] for value in sampled.values() if np.isfinite(value).any()])
    low, high = float(np.min(finite)), float(np.max(finite))
    if high == low:
        high = low + 1.0
    width, height, left, top, plot_w, plot_h = 1000, 470, 62, 20, 900, 385
    parts = [f'<div class="legend">']
    for index, key in enumerate(sampled):
        parts.append(f'<button data-series="{_esc(key)}" onclick="toggleSeries(this)"><span style="color:{colors[index % len(colors)]}">●</span> {_esc(_label(key, labels))}</button>')
    parts.append('</div><div class="panel"><svg viewBox="0 0 1000 470" role="img" aria-label="line chart">')
    parts.append(f'<text class="axis-label" x="{left + plot_w / 2}" y="8" text-anchor="middle">{_esc(title)}</text>')
    for tick in range(5):
        y = top + plot_h * tick / 4
        value = high - (high - low) * tick / 4
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/><text class="grid-value" x="4" y="{y + 3:.1f}">{value:.3g}</text>')
    parts.append(f'<text class="axis-label" transform="translate(15 {top + plot_h / 2}) rotate(-90)">{_esc(ytitle)}</text>')
    for index, (key, values) in enumerate(sampled.items()):
        points = []
        for x_index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = left + plot_w * x_index / max(1, len(values) - 1)
            y = top + plot_h * (high - value) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        points_text = " ".join(points)
        parts.append(f'<polyline class="path" data-line="{_esc(key)}" stroke="{color}" points="{points_text}" fill="none"><title>{_esc(_label(key, labels))}</title></polyline>')
        parts.append(f'<circle class="dot" data-line="{_esc(key)}" cx="{left + plot_w * (len(values) - 1) / max(1, len(values) - 1):.1f}" cy="{top + plot_h * (high - values[-1]) / (high - low):.1f}" r="4.5" fill="{color}"><title>{_esc(_label(key, labels))} final</title></circle>')
    parts.append('</svg></div>')
    return ''.join(parts)


def plot_cumulative_returns(returns: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a lightweight interactive normalized cumulative performance chart."""
    cumulative = (1.0 + returns).cumprod()
    _write(title, _line_chart({column: cumulative[column].to_numpy() for column in cumulative}, title, labels, "Growth of $1"), output_path)


def plot_realized_vs_fitted(realized: pd.DataFrame, fitted: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a lightweight realized-versus-fitted chart."""
    series = {}
    for asset in realized.columns:
        series[f"{asset} realized"] = ((1.0 + realized[asset].fillna(0.0)).cumprod()).to_numpy()
        series[f"{asset} fitted"] = ((1.0 + fitted[asset].fillna(0.0)).cumprod()).to_numpy()
    display = {key: f"{_label(key.rsplit(' ', 1)[0], labels)} {key.rsplit(' ', 1)[1]}" for key in series}
    _write(title, _line_chart(series, title, display, "Growth of $1"), output_path)


def plot_factor_moves(factors: pd.DataFrame, output_path: Path) -> None:
    """Write the estimated OOS latent factor innovations."""
    _write("OOS latent factor moves from frozen benchmark loadings", _line_chart(
        {column: factors[column].cumsum().to_numpy() for column in factors},
        "OOS latent factor moves from frozen benchmark loadings", ytitle="Cumulative factor move",
    ), output_path)


def plot_fund_factor_attribution(
    raw: pd.DataFrame,
    fitted: pd.DataFrame,
    residual: pd.DataFrame,
    factors: pd.DataFrame,
    exposures: pd.DataFrame,
    output_path: Path,
    labels: dict[str, str] | None = None,
) -> None:
    """Write raw, fitted, residual, and factor contribution views for every fund."""
    parts = ['<div class="panel"><p class="note">Each fund is scored OOS using no-intercept betas fitted only on the training sample. The fitted line is the sum of its daily factor contributions; the residual is the unexplained return.</p>']
    for ticker in raw.columns:
        series = {
            f"{ticker} raw": (1.0 + raw[ticker].fillna(0.0)).cumprod().to_numpy(),
            f"{ticker} fitted": (1.0 + fitted[ticker].fillna(0.0)).cumprod().to_numpy(),
            f"{ticker} residual": (1.0 + residual[ticker].fillna(0.0)).cumprod().to_numpy(),
        }
        parts.append(f'<h2>{_esc(_label(ticker, labels))}</h2>')
        parts.append(_line_chart(series, f"{ticker}: raw versus factor-fitted and residual", labels, "Growth of $1"))
        exposure = exposures.loc[ticker]
        parts.append('<div class="table-wrap"><table><thead><tr><th>Factor</th><th>Beta</th><th>Annual return contribution</th></tr></thead><tbody>')
        for factor in factors.columns:
            beta = exposure.get(f"beta_{factor}", 0.0)
            parts.append(f'<tr><td>{_esc(factor)}</td><td>{beta:.4f}</td><td>{beta * factors[factor].mean() * 252:.2%}</td></tr>')
        parts.append(f'<tr><td>Intercept (fixed at zero)</td><td></td><td>{exposure.get("alpha", 0.0) * 252:.2%}</td></tr>')
        parts.append(f'<tr><td>Residual</td><td></td><td>{residual[ticker].mean() * 252:.2%}</td></tr></tbody></table></div>')
    parts.append('</div>')
    _write("Fund factor attribution: raw, fitted, and residual performance", ''.join(parts), output_path)


def plot_backtest_comparison(raw: "BacktestResult", neutral: "BacktestResult", output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a side-by-side trend-following comparison: raw versus factor-neutralised."""
    raw_stats = {"annual_vol": raw.annual_vol, "max_drawdown": raw.max_drawdown, "sharpe": raw.sharpe, "cagr": raw.cagr}
    neutral_stats = {"annual_vol": neutral.annual_vol, "max_drawdown": neutral.max_drawdown, "sharpe": neutral.sharpe, "cagr": neutral.cagr}
    parts = ['<div class="kpi">', _kpi_cards(raw_stats, {"annual_vol": "Annual vol", "max_drawdown": "Max drawdown", "sharpe": "Sharpe", "cagr": "CAGR"}), '</div>']
    parts.append('<div class="kpi">')
    parts.append(_kpi_cards(neutral_stats, {"annual_vol": "Annual vol", "max_drawdown": "Max drawdown", "sharpe": "Sharpe", "cagr": "CAGR"}))
    parts.append('</div>')
    parts.append(f'<p class="note">The factor-neutralised strategy strips the systematic beta against the benchmark, leaving a long-only, market-neutral position. Lower volatility and drawdown with sustained capture indicate the residual idiosyncratic alpha.</p>')
    raw_equity = raw.equity_curve.to_numpy()
    neutral_equity = neutral.equity_curve.to_numpy()
    first_live = np.flatnonzero(np.isfinite(raw_equity) & np.isfinite(neutral_equity))
    if len(first_live):
        start = first_live[0]
        raw_equity = raw_equity / raw_equity[start]
        neutral_equity = neutral_equity / neutral_equity[start]
    chart_values = np.concatenate([raw_equity[np.isfinite(raw_equity)], neutral_equity[np.isfinite(neutral_equity)]])
    chart_low, chart_high = float(np.min(chart_values)), float(np.max(chart_values))
    if chart_high == chart_low:
        chart_high = chart_low + 1.0
    raw_daily = raw.daily_returns.to_numpy()
    days = np.arange(1.0, len(raw_daily) + 1.0)
    parts.append('<div class="panel"><svg viewBox="0 0 1000 470" role="img" aria-label="Trend-following equity comparison">')
    parts.append(f'<text class="axis-label" x="500" y="8" text-anchor="middle">Rolling EWMA trend follower: raw benchmark vs. factor-neutralised</text>')
    parts.append(f'<line class="grid" x1="62" x2="938" y1="20" y2="20"/><text class="axis-label" transform="translate(15 244)" rotate="-90">Cumulative return</text>')
    for key, values in (("Raw trend follower", raw_equity), ("Neutral trend follower", neutral_equity)):
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        points = []
        for x_index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = 62 + 876 * x_index / max(1, len(values) - 1)
            y = 42 + 388 * (chart_high - value) / (chart_high - chart_low)
            points.append(f"{x:.1f},{y:.1f}")
        color = "#176b87" if "Raw" in key else "#2f855a"
        result = raw if "Raw" in key else neutral
        parts.append(f'<polyline class="path" data-line="{_esc(key)}" stroke="{color}" points="{" ".join(points)}" fill="none"><title>{_esc(key)}: final {values[np.isfinite(values)][-1]:.1%}, vol {result.annual_vol:.1%}</title></polyline>')
    parts.append('</svg></div>')
    parts.append('<div class="panel"><p class="note">Daily return paths. The neutral strategy compresses the volatility while preserving the trend-following exposure.</p>')
    parts.append('<div class="table-wrap"><table><thead><tr><th></th><th>Annual vol</th><th>Max drawdown</th><th>Sharpe</th><th>Final return</th><th>Max leverage</th></tr></thead><tbody>')
    for key, result in (("Raw trend follower", raw), ("Neutral trend follower", neutral)):
        parts.append(f'<tr><td>{_esc(key)}</td><td>{result.annual_vol:.2%}</td><td>{result.max_drawdown:.2%}</td><td>{result.sharpe:.2f}</td><td>{result.final_return:.2%}</td><td>{result.max_leverage:.2f}x</td></tr>')
    parts.append('</tbody></table></div></div>')
    _write("Rolling EWMA trend follower comparison", ''.join(parts), output_path)


def plot_backtest_attribution(attribution: pd.DataFrame, output_path: Path) -> None:
    """Write portfolio beta, variance, and return attribution for both sleeves."""
    columns = ["beta", "factor_volatility", "variance_contribution_pct", "annual_return_contribution"]
    headers = ["Source", "Beta", "Factor vol", "Variance share", "Annual return"]
    parts = ['<div class="panel"><p class="note">Variance shares are covariance allocations to total portfolio variance; return contributions are beta times mean factor return. SPX_beta uses SPY as its proxy.</p><div class="table-wrap"><table><thead><tr><th>Strategy</th>']
    parts.extend(f"<th>{header}</th>" for header in headers)
    parts.append('</tr></thead><tbody>')
    for strategy, rows in attribution.groupby(level=0):
        for source, values in rows.droplevel(0).iterrows():
            parts.append(
                f'<tr><td>{_esc(strategy)} / {_esc(source)}</td>'
                f'<td>{values[columns[0]]:.3f}</td><td>{values[columns[1]]:.2%}</td>'
                f'<td>{values[columns[2]]:.2%}</td><td>{values[columns[3]]:.2%}</td></tr>'
            )
    parts.append('</tbody></table></div></div>')
    _write("Portfolio factor decomposition and attribution", ''.join(parts), output_path)


def plot_correlation_heatmap(returns: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a compact hoverable correlation heatmap."""
    correlation = returns.corr()
    values = correlation.to_numpy()
    headers = [_label(value, labels) for value in correlation.columns]
    cells = []
    for row_index, row in enumerate(values):
        cells.append(f"<tr><th>{_esc(headers[row_index])}</th>")
        for value in row:
            red = int(255 - max(0, value) * 100)
            blue = int(255 - max(0, -value) * 100)
            cells.append(f'<td title="Correlation: {value:.3f}" style="background:rgb(255,{blue},{red})">{value:.2f}</td>')
        cells.append('</tr>')
    header = ''.join(f'<th>{_esc(value)}</th>' for value in headers)
    body = f'<div class="panel"><p class="note">Hover a cell for the exact correlation.</p><div class="table-wrap"><table class="heat"><tr><th></th>{header}</tr>{"".join(cells)}</table></div></div>'
    _write(title, body, output_path)


def plot_loadings(loadings: np.ndarray, assets: list[str], title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a compact hoverable loading heatmap."""
    headers = [f"Factor {index + 1}" for index in range(loadings.shape[1])]
    rows = []
    for asset, values in zip(assets, loadings):
        cells = ''.join(f'<td title="Loading: {value:.6f}">{value:.3f}</td>' for value in values)
        rows.append(f'<tr><th>{_esc(_label(asset, labels))}</th>{cells}</tr>')
    body = f'<div class="panel"><p class="note">Hover a cell for the full loading.</p><div class="table-wrap"><table class="heat"><tr><th></th>{"".join(f"<th>{_esc(value)}</th>" for value in headers)}</tr>{"".join(rows)}</table></div></div>'
    _write(title, body, output_path)


def plot_systematic_vs_idio(attribution: pd.DataFrame, output_path: Path, title: str = "Strategy risk decomposition", labels: dict[str, str] | None = None) -> None:
    """Write interactive systematic and idiosyncratic volatility bars."""
    metrics = []
    for asset, row in attribution.iterrows():
        label = _label(asset, labels)
        metrics.append(f'<div class="metric"><span class="name">{_esc(label)}</span><div><div class="bar" style="width:{max(2, min(100, row.systematic_volatility * 300)):.1f}%" title="Systematic: {row.systematic_volatility:.2%}"></div><div class="bar warm" style="width:{max(2, min(100, row.idiosyncratic_volatility * 300)):.1f}%" title="Idiosyncratic: {row.idiosyncratic_volatility:.2%}"></div></div><span class="small">{row.idiosyncratic_volatility:.1%} residual</span></div>')
    body = '<div class="panel"><p class="note">Blue: systematic volatility. Red: residual volatility. Hover bars for exact values.</p><div data-metrics>' + ''.join(metrics) + '</div></div>'
    _write(title, body, output_path)


def plot_capture_diagnostics(diagnostics: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write compact interactive capture and residual-risk diagnostics."""
    rows = []
    for asset, row in diagnostics.iterrows():
        rows.append(f'<div class="metric"><span class="name">{_esc(_label(asset, labels))}</span><div><div class="bar" style="width:{max(2, min(100, row.oos_r_squared * 100)):.1f}%" title="OOS R2: {row.oos_r_squared:.3f}"></div></div><span class="small">OOS R2 {row.oos_r_squared:.1%}</span></div>')
    body = '<div class="panel"><p class="note">Out-of-sample capture. Negative values are shown as zero-width bars but remain in the CSV diagnostics.</p><div data-metrics>' + ''.join(rows) + '</div></div>'
    _write(title, body, output_path)


def plot_factor_decomposition(decomposition: pd.DataFrame, asset: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a factor allocation table with a highlighted asset panel."""
    headers = list(decomposition.columns)
    rows = []
    for name, values in decomposition.iterrows():
        cells = ''.join(f'<td title="Contribution: {value:.4%}">{value:.1%}</td>' for value in values)
        rows.append(f'<tr><th>{_esc(_label(name, labels))}</th>{cells}</tr>')
    selected = decomposition.loc[asset]
    selected_rows = ''.join(f'<div class="metric"><span class="name">{_esc(name)}</span><div><div class="bar" style="width:{max(2, min(100, abs(value) * 100)):.1f}%" title="Contribution: {value:.4%}"></div></div><span class="small">{value:.1%}</span></div>' for name, value in selected.items())
    body = f'<div class="panel"><p class="note">Variance contributions can be negative because factors are correlated. Hover cells for exact values.</p><div class="table-wrap"><table class="heat"><tr><th></th>{"".join(f"<th>{_esc(value)}</th>" for value in headers)}</tr>{"".join(rows)}</table></div><h2>{_esc(_label(asset, labels))}: modeled risk allocation</h2>{selected_rows}</div>'
    _write("Nonlinear factor variance allocation", body, output_path)


def _kpi_cards(metrics: dict[str, float], labels: dict[str, str] | None = None) -> str:
    """Write a row of summary metric cards."""
    parts = ['<div class="kpi">']

    for rank, (key, value) in enumerate(metrics.items()):
        label = _label(key, labels)
        parts.append(
            f'<div class="card"><div class="kpi-value">{_esc(f"{value:.2%}" if -1.0 <= value <= 1.0 else f"{value:.3g}")}</div><div class="kpi-label">{_esc(label)}</div></div>')
    parts.append('</div>')
    return ''.join(parts)


def plot_cumulative_returns_with_kpis(returns: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a growth curve with a summary KPI band, like a polished performance sheet."""
    cumulative = (1.0 + returns).cumprod()
    stats = {key: float(cumulative[key].iloc[-1] - 1.0) for key in cumulative.columns}
    body = _kpi_cards(stats, labels)
    body += _line_chart({column: cumulative[column].to_numpy() for column in cumulative}, title, labels, "Growth of $1")
    _write(title, body, output_path)


def plot_drawdown(equity_curve: np.ndarray, title: str, output_path: Path, label: str = "Drawdown") -> None:
    """Write a rolling drawdown chart with a gradient shaded area."""
    equity = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100.0
    width, height, left, top, plot_w, plot_h = 1000, 470, 62, 20, 900, 385
    n = len(drawdown)
    xs = np.linspace(left, left + plot_w, n)
    ys = top + plot_h * drawdown
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, np.full(n, top)))
    body = f'''<div class="panel"><p class="note">The depth of each trough is the peak-to-trough drawdown at that time. Cleaner strategies show shallower, wider troughs.</p><svg viewBox="0 0 1000 470" role="img" aria-label="drawdown chart">
  <text class="axis-label" x="{left + plot_w / 2}" y="8" text-anchor="middle">{_esc(title)}</text>
  <line class="grid" x1="{left}" x2="{left + plot_w}" y1="{top}" y2="{top}"/>
  <polyline class="area" points="{area}" fill="rgba(217,95,89,.28)"/>
  <polyline class="path" stroke="#d95f59" points="{path}" fill="none"/>
</svg></div>'''
    _write(title, body, output_path)


def plot_equity_comparison(equities: dict[str, np.ndarray], title: str, output_path: Path, labels: dict[str, str] | None = None) -> None:
    """Write a backtest equity-curve comparison with stats cards."""
    stats = {}
    for key, values in equities.items():
        values = np.asarray(values, dtype=float)
        final = values[-1]
        stats[key] = {
            "cagr": float(np.exp(np.log(final) / max(1, len(values) - 1)) - 1.0),
            "vol": float(np.std(values[1:] - values[:-1], ddof=1) * np.sqrt(252)),
            "sharpe": float(np.sum(np.diff(values)) / (len(values) * np.std(values[1:] - values[:-1], ddof=1)) * np.sqrt(252)) if np.std(values[1:] - values[:-1], ddof=1) > 0 else 0.0,
        }
    cards = []
    for rank, (key, value) in enumerate(stats.items()):
        label = _label(key, labels)
        cagr = value["cagr"]
        cards.append(f'<div class="card"><div class="kpi-value">{_esc(f"{cagr:.2%}")}</div><div class="kpi-label">{_esc(label)} CAGR</div></div>')
    parts = f'<div class="panel"><h2 style="margin-top:0">Performance</h2>{"".join(cards)}'
    parts += _line_chart({key: _sample(np.asarray(values, dtype=float)) for key, values in equities.items()}, title, labels, "Growth of $1")
    parts += '</div>'
    _write(title, parts, output_path)
