"""Ticker universes used by the research data and visualisation workflow."""

from __future__ import annotations

BENCHMARK_UNIVERSE: dict[str, str] = {
    "ACWI": "Global equities",
    "SPY": "US large-cap equities",
    "DIA": "US industrial large-cap equities",
    "QQQ": "US large-cap growth equities",
    "IWM": "US small-cap equities",
    "EFA": "Developed ex-US equities",
    "EEM": "Emerging-market equities",
    "AGG": "US aggregate bonds",
    "SHY": "US 1-3 year Treasury bonds",
    "IEF": "US 7-10 year Treasury bonds",
    "TLT": "US 20+ year Treasury bonds",
    "TIP": "US inflation-linked bonds",
    "LQD": "Investment-grade credit",
    "HYG": "High-yield credit",
    "VNQ": "US real estate",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "UUP": "US dollar",
}

MODEL_UNIVERSE = BENCHMARK_UNIVERSE

TARGET_UNIVERSE: dict[str, str] = {
    "PRPFX": "Permanent Portfolio mutual fund",
    "QMN": "Alternative risk premia ETF",
    "DBMF": "Managed futures ETF",
    "KMLM": "Trend-following ETF",
    "BTAL": "Long defensive / short beta ETF",
    "RPAR": "Risk-parity ETF",
    "MNA": "Merger-arbitrage ETF",
    "CTA": "Managed futures ETF",
    "MTUM": "US momentum equities",
    "QUAL": "US quality equities",
    "USMV": "US minimum-volatility equities",
    "VLUE": "US value equities",
    "JEPI": "US equity option-income ETF",
    "JEPQ": "US technology option-income ETF",
    "DIVO": "US dividend and option-income ETF",
    "ARKK": "US innovation and thematic growth ETF",
    "QAI": "Hedge-fund strategy replication ETF",
    "^PUT": "Cboe S&P 500 PutWrite Index (benchmark)",
}

EXPLANATORY_UNIVERSE = TARGET_UNIVERSE


ALL_UNIVERSES = {**MODEL_UNIVERSE, **EXPLANATORY_UNIVERSE}


def display_labels(universe: dict[str, str] | None = None) -> dict[str, str]:
    """Return chart labels that preserve tickers and add readable descriptions."""
    descriptions = ALL_UNIVERSES if universe is None else universe
    return {ticker: f"{ticker} - {description}" for ticker, description in descriptions.items()}
