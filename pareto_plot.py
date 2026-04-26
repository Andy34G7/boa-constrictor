#!/usr/bin/env python3
"""Create a Pareto frontier plot from model performance metrics.

Expected input: CSV with one row per model/method. By default this script uses:
- x-axis: throughput_compress_MBps (higher is better)
- y-axis: compression_ratio (higher is better)
- label: model

You can override columns with CLI flags.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    v = str(value).strip()
    if not v:
        return float("nan")
    return float(v)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _compute_derived_metrics(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        # Derive compression_ratio when compressed and original sizes are provided.
        if "compression_ratio" not in row or not str(row.get("compression_ratio", "")).strip():
            comp = _to_float(row.get("compressed_size"))
            orig = _to_float(row.get("original_size"))
            if np.isfinite(comp) and np.isfinite(orig) and comp > 0:
                row["compression_ratio"] = str(orig / comp)

        # Derive throughput_compress_MBps when time and original size are provided.
        if "throughput_compress_MBps" not in row or not str(row.get("throughput_compress_MBps", "")).strip():
            t_comp = _to_float(row.get("time_compress_s"))
            orig = _to_float(row.get("original_size"))
            if np.isfinite(t_comp) and np.isfinite(orig) and t_comp > 0:
                row["throughput_compress_MBps"] = str((orig / 1e6) / t_comp)


def _pareto_mask(x: np.ndarray, y: np.ndarray, maximize_x: bool, maximize_y: bool) -> np.ndarray:
    """Return boolean mask where True means point is Pareto-optimal."""
    sx = x if maximize_x else -x
    sy = y if maximize_y else -y

    n = len(sx)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = ((sx >= sx[i]) & (sy >= sy[i]) & ((sx > sx[i]) | (sy > sy[i]))).any()
        if dominated:
            mask[i] = False
    return mask


def _filter_rows(
    rows: List[Dict[str, str]],
    label_col: str,
    x_col: str,
    y_col: str,
    include_models: set[str] | None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    labels: List[str] = []
    xs: List[float] = []
    ys: List[float] = []

    for row in rows:
        if label_col not in row:
            continue
        label = str(row[label_col]).strip()
        if not label:
            continue
        if include_models is not None and label not in include_models:
            continue

        x = _to_float(row.get(x_col))
        y = _to_float(row.get(y_col))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        labels.append(label)
        xs.append(x)
        ys.append(y)

    return labels, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot model Pareto frontier from a CSV file")
    p.add_argument("--input", type=Path, required=True, help="Input CSV file with model metrics")
    p.add_argument("--output", type=Path, default=Path("pareto_frontier.png"), help="Output image path")
    p.add_argument("--label-col", default="model", help="Column used for model labels")
    p.add_argument("--x-col", default="throughput_compress_MBps", help="X-axis metric column")
    p.add_argument("--y-col", default="compression_ratio", help="Y-axis metric column")
    p.add_argument("--x-minimize", action="store_true", help="Treat smaller X as better (default: maximize X)")
    p.add_argument("--y-minimize", action="store_true", help="Treat smaller Y as better (default: maximize Y)")
    p.add_argument("--title", default="Pareto Frontier", help="Plot title")
    p.add_argument("--xlabel", default=None, help="Custom x-axis label")
    p.add_argument("--ylabel", default=None, help="Custom y-axis label")
    p.add_argument("--models", default=None, help="Comma-separated labels to include")
    p.add_argument("--no-labels", action="store_true", help="Disable point text labels")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    maximize_x = not args.x_minimize
    maximize_y = not args.y_minimize

    rows = _read_csv(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    _compute_derived_metrics(rows)

    include_models = None
    if args.models:
        include_models = {x.strip() for x in args.models.split(",") if x.strip()}

    labels, x, y = _filter_rows(
        rows,
        label_col=args.label_col,
        x_col=args.x_col,
        y_col=args.y_col,
        include_models=include_models,
    )

    if len(labels) == 0:
        raise ValueError(
            "No valid points after filtering. Check columns or provide data for derived metrics: "
            "(original_size, compressed_size) and/or (original_size, time_compress_s)."
        )

    frontier_mask = _pareto_mask(x, y, maximize_x=maximize_x, maximize_y=maximize_y)

    xf = x[frontier_mask]
    yf = y[frontier_mask]
    lf = [labels[i] for i in np.where(frontier_mask)[0]]

    order = np.argsort(xf)
    xf = xf[order]
    yf = yf[order]
    lf = [lf[i] for i in order]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=45, alpha=0.55, label="All models")
    plt.scatter(xf, yf, s=70, alpha=0.95, label="Pareto frontier")
    plt.plot(xf, yf, linewidth=2, alpha=0.9)

    if not args.no_labels:
        for xi, yi, li in zip(x, y, labels):
            plt.annotate(li, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.title(args.title)
    plt.xlabel(args.xlabel or args.x_col)
    plt.ylabel(args.ylabel or args.y_col)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)

    print(f"Saved Pareto plot to: {args.output}")
    print("Pareto-optimal models:")
    for li in lf:
        print(f"  - {li}")


if __name__ == "__main__":
    main()
