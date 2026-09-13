"""Download the Cboe PUT strategy index alongside the S&P 500 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from riskmodel.data_loader import fetch_asset_data
from riskmodel.visuals import plot_cumulative_returns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1996-01-01")
    parser.add_argument("--end", default="2026-08-27")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prices_returns = fetch_asset_data(["^PUT", "^SPX"], args.start, args.end)
    returns = prices_returns.dropna()
    if not {"^PUT", "^SPX"}.issubset(returns.columns):
        missing = sorted({"^PUT", "^SPX"} - set(returns.columns))
        raise ValueError(f"The data provider did not return: {', '.join(missing)}")

    performance = (1.0 + returns).cumprod()
    performance = performance.div(performance.iloc[0]).mul(100.0)
    returns.to_csv(args.output / "put_spx_daily_returns.csv", index_label="Date")
    performance.to_csv(args.output / "put_spx_performance.csv", index_label="Date")
    figures_dir = args.output.parent.parent / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_cumulative_returns(
        returns,
        "Cboe PUT versus S&P 500: cumulative performance",
        figures_dir / "put_vs_spx_performance.html",
        labels={
            "^PUT": "^PUT - Cboe S&P 500 PutWrite Index",
            "^SPX": "^SPX - S&P 500 Index",
        },
    )

    metadata = {
        "start": args.start,
        "end": args.end,
        "tickers": ["^PUT", "^SPX"],
        "price_field": "Adj Close",
        "performance_base": 100.0,
        "common_start": returns.index[0].date().isoformat(),
        "common_end": returns.index[-1].date().isoformat(),
        "observations": len(returns),
        "note": "Returns and indexed performance are calculated from yfinance adjusted-close data on the common observation dates.",
    }
    (args.output / "put_spx_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Downloaded {len(returns)} common observations")
    print(f"Common range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Wrote returns and performance to {args.output.resolve()}")
    print(f"Wrote chart to {(figures_dir / 'put_vs_spx_performance.html').resolve()}")


if __name__ == "__main__":
    main()