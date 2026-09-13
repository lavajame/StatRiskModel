"""Build one dependency-free interactive explorer from generated model reports."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskmodel.universes import ALL_UNIVERSES


REPORTS = Path("reports")
OUTPUT = REPORTS / "factor_model_explorer.html"
TEMPLATE = Path(__file__).with_name("templates") / "factor_explorer.html"


def parse_array(value: str) -> list[float]:
    """Parse the compact array strings written by pandas into JSON-safe values."""
    text = str(value).replace("\n", " ").replace("  ", " ")
    try:
        return [float(item) for item in ast.literal_eval(text)]
    except (SyntaxError, ValueError):
        stripped = text.strip().strip("[]")
        return [float(item) for item in stripped.split()]


def read_exposures(path: Path, group: str) -> list[dict[str, object]]:
    frame = pd.read_csv(path, index_col=0)
    factor_columns = sorted(
        (column for column in frame.columns if column.startswith("beta_Factor_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    records = []
    for ticker, row in frame.iterrows():
        if factor_columns:
            delta = [float(row[column]) for column in factor_columns]
            gamma = [0.0] * len(delta)
            downside = [0.0] * len(delta)
        else:
            delta = parse_array(row["beta_delta"])
            gamma = parse_array(row["beta_gamma"])
            downside = parse_array(row["beta_downside"])
        records.append({
            "ticker": ticker,
            "label": f"{ticker} - {ALL_UNIVERSES.get(ticker, ticker)}",
            "group": group,
            "delta": delta,
            "gamma": gamma,
            "downside": downside,
            "r2_train": float(row.get("r_squared", 0.0)),
        })
    return records


def build_payload() -> dict[str, object]:
    benchmark = read_exposures(REPORTS / "benchmark_train_exposures.csv", "Benchmark")
    other = read_exposures(REPORTS / "strategy_train_exposures.csv", "Other")
    assets = {item["ticker"]: item for item in benchmark + other}
    if "^PUT" in assets:
        assets["^PUT"]["group"] = "Benchmark"

    diagnostics = pd.concat([
        pd.read_csv(REPORTS / "benchmark_oos_diagnostics.csv"),
        pd.read_csv(REPORTS / "strategy_oos_diagnostics.csv"),
    ]).set_index("ticker")
    for ticker, item in assets.items():
        if ticker in diagnostics.index:
            row = diagnostics.loc[ticker]
            item["oos_r2"] = float(row["oos_r_squared"])
            item["residual_volatility"] = float(row["idiosyncratic_volatility"])
            item["realized_volatility"] = float(row["realized_volatility"])
            item["systematic_volatility"] = float(row["systematic_volatility"])

    decomposition = pd.concat([
        pd.read_csv(REPORTS / "benchmark_factor_variance_decomposition.csv", index_col=0),
        pd.read_csv(REPORTS / "strategy_factor_variance_decomposition.csv", index_col=0),
    ])
    for ticker, item in assets.items():
        item["variance_share"] = [float(value) for value in decomposition.loc[ticker].to_numpy()]

    residuals = pd.concat([
        pd.read_csv(REPORTS / "benchmark_oos_residuals.csv", index_col=0, parse_dates=True),
        pd.read_csv(REPORTS / "strategy_oos_residuals.csv", index_col=0, parse_dates=True),
    ], axis=1)
    correlation = residuals.corr().fillna(0.0)
    corr_values = correlation.to_numpy()
    centering = np.eye(len(correlation)) - np.ones(corr_values.shape) / len(correlation)
    distances_squared = 2.0 * (1.0 - corr_values)
    gram = -0.5 * centering @ distances_squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    coordinates = []
    for row in range(len(correlation)):
        point = []
        for index in order[:3]:
            point.append(float(eigenvectors[row, index] * np.sqrt(max(eigenvalues[index], 0.0))))
        coordinates.append((point + [0.0, 0.0, 0.0])[:3])
    for ticker, item in assets.items():
        realized = max(float(item.get("realized_volatility", 0.0)), 1e-12)
        residual = float(item.get("residual_volatility", 0.0))
        item["residual_variance_share"] = min(1.0, max(0.0, residual * residual / (realized * realized)))
        if ticker in correlation.index:
            neighbors = correlation.loc[ticker].drop(labels=ticker).sort_values(ascending=False).head(3)
            item["residual_neighbors"] = [{"ticker": name, "correlation": float(value)} for name, value in neighbors.items()]
    names = correlation.columns.tolist()
    edges = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            rho = float(correlation.iat[i, j])
            if abs(rho) >= 0.20:
                edges.append({"a": left, "b": names[j], "rho": rho})
    return {
        "assets": list(assets.values()),
        "labels": {ticker: item["label"] for ticker, item in assets.items()},
        "residual_embedding": coordinates,
        "residual_edges": edges,
        "correlation_assets": names,
        "factor_names": [f"Factor {index + 1}" for index in range(8)],
    }


def render(payload: dict[str, object]) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))


def main() -> None:
    OUTPUT.write_text(render(build_payload()), encoding="utf-8")
    print(f"Wrote {OUTPUT.resolve()} ({OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
