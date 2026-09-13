"""Regenerate the offline HTML reports from committed market data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model outputs and regenerate both shareable HTML reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "daily_returns.csv",
        help="Offline daily returns CSV (default: data/raw/daily_returns.csv)",
    )
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-08-15")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fresh market data with yfinance instead of using --input",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    build = [
        sys.executable,
        "scripts/build_visuals.py",
        "--start",
        args.start,
        "--end",
        args.end,
        "--output",
        "reports",
    ]
    if args.download:
        build.append("--download")
    else:
        build.extend(["--input", str(args.input)])

    run(build)
    run([sys.executable, "scripts/build_research_viewer_polished.py"])
    run([sys.executable, "scripts/build_explorer.py"])
    print("\nReports regenerated:")
    print("  reports/research_viewer.html")
    print("  reports/factor_model_explorer.html")


if __name__ == "__main__":
    main()
