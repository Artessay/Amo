#!/usr/bin/env python3
"""Plot response-level helpfulness--harmlessness joint distributions.

This figure intentionally does not draw a cross-prompt Pareto frontier or a
set hypervolume.  Each panel visualizes the empirical distribution of
individual responses, while the annotation reports the mean rooted singleton
hypervolume.  It uses the frozen PKU-SafeRLHF calibration shared by training
and paper evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amo-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


MODELS = (
    ("qwen2.5-3b", "Qwen2.5-3B-Instruct"),
)
METHODS = (
    ("grpo", "GRPO", "#4C78A8"),
    ("gdpo", "GDPO", "#F58518"),
    ("hvpo", "HVPO", "#54A24B"),
)
EXPECTED_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_response(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("row does not contain a generated response")
    response = value[0]
    if not isinstance(response, str):
        raise ValueError("generated response is not a string")
    return response


def read_score_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing score cache: {path}\n"
            "Run playground/visualization/record_score.py first."
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != EXPECTED_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number} uses a legacy cache schema; rescore "
                    "the current Parquet with record_score.py --overwrite"
                )
            records.append(record)
    if not records:
        raise ValueError(f"score cache is empty: {path}")
    return records


def validate_cache(
    records: list[dict[str, Any]],
    parquet_path: Path,
    experiment: str,
) -> int:
    source_sha256 = sha256_file(parquet_path)
    cache_sources = {record.get("source_sha256") for record in records}
    if cache_sources != {source_sha256}:
        raise ValueError(
            f"score cache for {experiment} does not match {parquet_path}; "
            "rescore with record_score.py --overwrite"
        )
    cache_experiments = {record.get("experiment") for record in records}
    if cache_experiments != {experiment}:
        raise ValueError(f"score cache contains unexpected experiments: {cache_experiments}")

    frame = pd.read_parquet(parquet_path, columns=["extra_info", "responses"])
    seen_rows: set[int] = set()
    for record in records:
        source_row = int(record["source_row"])
        if source_row in seen_rows:
            raise ValueError(f"duplicate source_row={source_row} in {experiment} cache")
        seen_rows.add(source_row)
        if not 0 <= source_row < len(frame):
            raise ValueError(f"source_row={source_row} is outside {parquet_path}")

        row = frame.iloc[source_row]
        extra_info = row["extra_info"]
        question = extra_info.get("question", "") if isinstance(extra_info, dict) else ""
        response = _first_response(row["responses"])
        if (
            sha256_text(question) != record.get("question_sha256")
            or sha256_text(response) != record.get("response_sha256")
        ):
            raise ValueError(
                f"score cache for {experiment} does not match source_row={source_row}"
            )
    return len(frame)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    lower = np.asarray(data["calib_lower"], dtype=float)
    upper = np.asarray(data["calib_upper"], dtype=float)
    ideal_raw = lower + np.asarray(data.get("ideal", [1.0, 1.0]), dtype=float) * (upper - lower)
    reference_raw = np.asarray(data["hv_reference"], dtype=float)
    for name, value in {
        "calib_lower": lower,
        "calib_upper": upper,
        "ideal": ideal_raw,
        "hv_reference": reference_raw,
    }.items():
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain two finite values")
    if np.any(upper <= lower):
        raise ValueError("calibration upper bounds must exceed lower bounds")
    return lower, upper, reference_raw, ideal_raw


def calibrate(raw_scores: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (raw_scores - lower) / (upper - lower)


def rooted_singleton_hv(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gains = points - reference
    positive = np.all(gains > 0.0, axis=1)
    values = np.zeros(len(points), dtype=float)
    values[positive] = np.sqrt(gains[positive, 0] * gains[positive, 1])
    return values


def _blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1.0 - strength) * np.ones(3) + strength * rgb)


def _mass_thresholds(density: np.ndarray, masses: tuple[float, ...]) -> list[float]:
    flattened = np.asarray(density, dtype=float).ravel()
    flattened = flattened[np.isfinite(flattened) & (flattened > 0)]
    if not len(flattened):
        return []
    ordered = np.sort(flattened)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = []
    for mass in masses:
        index = min(int(np.searchsorted(cumulative, mass)), len(ordered) - 1)
        thresholds.append(float(ordered[index]))
    return thresholds


def draw_joint_density(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bins: int,
    smooth_sigma: float,
) -> None:
    histogram, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=(xlim, ylim)
    )
    density = gaussian_filter(histogram.T, sigma=smooth_sigma)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    outer, middle, inner = _mass_thresholds(density, (0.95, 0.80, 0.50))
    levels = np.unique([outer, middle, inner, float(density.max()) * (1.0 + 1e-9)])

    if len(levels) >= 2:
        colors = [
            _blend_with_white(color, strength)
            for strength in np.linspace(0.25, 0.78, len(levels) - 1)
        ]
        ax.contourf(
            x_centers,
            y_centers,
            density,
            levels=levels,
            colors=colors,
            antialiased=True,
            zorder=1,
        )
        ax.contour(
            x_centers,
            y_centers,
            density,
            levels=levels[:-1],
            colors=[color],
            linewidths=0.65,
            alpha=0.72,
            zorder=2,
        )
    else:
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.15, color=color, zorder=1)


def _axis_limits(point_sets: list[np.ndarray], anchors: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use the observed support plus visible calibration anchors.

    The HV reference is intentionally not forced into view when every response
    lies far above it; doing so would compress the joint distribution into a
    small corner.  If any response approaches or crosses the reference, the
    data limits include it naturally and the reference guides are drawn.
    """
    all_points = np.vstack([*point_sets, anchors])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    # Extra headroom is useful because the reward models place substantial
    # mass at their upper score ceilings; without it the outer density contour
    # visually collides with the top and right spines.
    x_pad = max(0.05, 0.07 * (x_max - x_min))
    y_pad = max(0.05, 0.07 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_joint_distributions(
    result_dir: Path,
    score_dir: Path,
    calibration_path: Path,
    output_pdf: Path,
    output_png: Path,
    summary_path: Path,
    bins: int,
    smooth_sigma: float,
) -> dict[str, dict[str, float | int | str]]:
    lower, upper, reference_raw, ideal_raw = load_calibration(calibration_path)
    reference = calibrate(reference_raw[None, :], lower, upper)[0]
    ideal = calibrate(ideal_raw[None, :], lower, upper)[0]

    datasets: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float | int | str]] = {}
    for model_key, _ in MODELS:
        for method_key, method_label, _ in METHODS:
            experiment = f"{model_key}_{method_key}"
            parquet_path = result_dir / f"{experiment}.parquet"
            cache_path = score_dir / f"{experiment}.jsonl"
            records = read_score_cache(cache_path)
            source_n = validate_cache(records, parquet_path, experiment)

            raw_scores = np.asarray(
                [[record["helpful_score"], record["harmless_score"]] for record in records],
                dtype=float,
            )
            if raw_scores.ndim != 2 or raw_scores.shape[1] != 2 or not np.isfinite(raw_scores).all():
                raise ValueError(f"non-finite or malformed scores in {cache_path}")
            points = calibrate(raw_scores, lower, upper)
            datasets[experiment] = points
            hv = rooted_singleton_hv(points, reference)
            summaries[experiment] = {
                "model": model_key,
                "method": method_label,
                "n": int(len(points)),
                "source_n": int(source_n),
                "complete_source_coverage": bool(len(points) == source_n),
                "helpfulness_mean_raw": float(raw_scores[:, 0].mean()),
                "harmlessness_mean_raw": float(raw_scores[:, 1].mean()),
                "mean_rooted_singleton_hv": float(hv.mean()),
                "zero_hv_fraction": float(np.mean(hv == 0.0)),
                "source_sha256": records[0]["source_sha256"],
                "score_cache_sha256": sha256_file(cache_path),
            }

    xlim, ylim = _axis_limits(list(datasets.values()), ideal[None, :])
    reference_visible = (
        xlim[0] <= reference[0] <= xlim[1]
        and ylim[0] <= reference[1] <= ylim[1]
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, (model_key, model_label) in enumerate(MODELS):
        for column, (method_key, method_label, color) in enumerate(METHODS):
            experiment = f"{model_key}_{method_key}"
            points = datasets[experiment]
            stats = summaries[experiment]
            ax = axes[row, column]
            draw_joint_density(ax, points, color, xlim, ylim, bins, smooth_sigma)

            mean = points.mean(axis=0)
            ax.scatter(
                mean[0],
                mean[1],
                marker="X",
                s=42,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            if reference_visible:
                ax.axvline(reference[0], color="#333333", linestyle="--", linewidth=0.7, alpha=0.7)
                ax.axhline(reference[1], color="#333333", linestyle="--", linewidth=0.7, alpha=0.7)
                ax.scatter(
                    reference[0],
                    reference[1],
                    marker="D",
                    s=24,
                    facecolor="#222222",
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=6,
                )
            ax.scatter(
                ideal[0],
                ideal[1],
                marker="*",
                s=58,
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.8,
                zorder=6,
            )
            ax.text(
                0.025,
                0.975,
                f"$\\overline{{H}}$={stats['mean_rooted_singleton_hv']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
                zorder=7,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35, zorder=0)
            if row == 0:
                ax.set_title(method_label, fontweight="bold", color=color, pad=5)
            ax.set_xlabel("Calibrated helpfulness")

        axes[row, 0].set_ylabel("Calibrated harmlessness")

    fig.suptitle(MODELS[0][1], y=0.995, fontsize=9, fontstyle="italic")

    legend_handles = [
        Line2D([0], [0], marker="X", linestyle="none", markersize=6, markerfacecolor="#666666", markeredgecolor="white", label="Response mean"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=8, markerfacecolor="white", markeredgecolor="#222222", label="Calibration ideal"),
    ]
    if reference_visible:
        legend_handles.insert(
            1,
            Line2D([0], [0], marker="D", linestyle="--", color="#333333", markersize=4, markerfacecolor="#222222", label="HV reference"),
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.005),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=7.5,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.82, bottom=0.25, wspace=0.08)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen response-level helpfulness--harmlessness distributions."
    )
    parser.add_argument("--result-dir", type=Path, default=Path("results/PKU-SafeRLHF"))
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("playground/visualization/scored_responses"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/PKU-SafeRLHF/safe_calibration.json"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("playground/visualization/joint_reward_distribution_qwen.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("playground/visualization/joint_reward_distribution_qwen.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("playground/visualization/joint_reward_distribution_qwen.json"),
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--smooth-sigma", type=float, default=1.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = plot_joint_distributions(
        result_dir=args.result_dir,
        score_dir=args.score_dir,
        calibration_path=args.calibration,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        summary_path=args.summary,
        bins=args.bins,
        smooth_sigma=args.smooth_sigma,
    )
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")
    print(f"Saved: {args.summary}")
    for experiment, stats in summaries.items():
        print(
            f"{experiment}: n={stats['n']} "
            f"coverage={'full' if stats['complete_source_coverage'] else 'preview'} "
            f"Hbar={stats['mean_rooted_singleton_hv']:.4f} "
            f"zero-HV={100 * stats['zero_hv_fraction']:.2f}%"
        )


if __name__ == "__main__":
    main()
