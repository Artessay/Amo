import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 读取单个 jsonl 文件为 DataFrame（不合并）======
def read_single_jsonl(jsonl_path: str | Path, label: str | None = None) -> pd.DataFrame:
    p = Path(jsonl_path)
    if not p.exists():
        print(f"File not found: {p}")
        return pd.DataFrame(columns=["experiment", "helpful", "harmless"])

    exp_name = label if label is not None else p.stem
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(
                    {
                        "experiment": exp_name,
                        "helpful": float(data.get("helpful_score", 0)),
                        "harmless": float(data.get("harmless_score", 0)),
                    }
                )
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records)

# ====== Pareto / HV (2D maximize) ======
def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.size == 0:
        return points

    idx = np.argsort(points[:, 0])[::-1]
    pts = points[idx]

    best_y = -np.inf
    nd = []
    for x, y in pts:
        if y > best_y:
            nd.append((x, y))
            best_y = y

    nd = np.array(nd, dtype=float)
    nd = nd[np.argsort(nd[:, 0])]
    return nd

def hypervolume_2d_max(points: np.ndarray, ref: tuple[float, float]) -> float:
    front = pareto_front_2d(points)
    if front.size == 0:
        return 0.0
    rx, ry = ref
    hv = 0.0
    prev_x = rx
    for x, y in front:
        hv += max(0.0, x - prev_x) * max(0.0, y - ry)
        prev_x = x
    return float(hv)

def fill_hv_rects(ax, front: np.ndarray, ref: tuple[float, float], color, alpha=0.25, zorder=1):
    rx, ry = ref
    if front.size == 0:
        return
    front = front[np.argsort(front[:, 0])]
    prev_x = rx
    for x, y in front:
        if x > prev_x and y > ry:
            ax.fill_between(
                [prev_x, x],
                [ry, ry],
                [y, y],
                color=color,
                alpha=alpha,
                zorder=zorder,
            )
        prev_x = x

# ====== 3列 × 2行 绘图 ======
def plot_pareto_grid_3x2_from_files(
    jsonl_paths: list[str],
    output_path: str,
    ref_point: tuple[float, float],
    share_axes: bool = True,
):
    group_names = [Path(p).stem for p in jsonl_paths]

    group_points = []
    for p, name in zip(jsonl_paths, group_names):
        df = read_single_jsonl(p, label=name)
        pts = df[["helpful", "harmless"]].to_numpy(dtype=float) if not df.empty else np.zeros((0, 2))
        pts = pts[np.isfinite(pts).all(axis=1)]
        group_points.append(pts)

    all_pts = np.vstack([p for p in group_points if p.size > 0])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    x_pad = 0.06 * (x_max - x_min)
    y_pad = 0.06 * (y_max - y_min)
    xlim = (min(ref_point[0], x_min) - x_pad, x_max + x_pad)
    ylim = (min(ref_point[1], y_min) - y_pad, y_max + y_pad)

    palette = [
        {"edge": "#d62728", "fill": "#f4a6a6"},
        {"edge": "#1f77b4", "fill": "#9ecae1"},
        {"edge": "#2ca02c", "fill": "#a1d99b"},
        {"edge": "#ff7f0e", "fill": "#fdd0a2"},
        {"edge": "#9467bd", "fill": "#dadaeb"},
        {"edge": "#8c564b", "fill": "#e7cb94"},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=share_axes, sharey=share_axes)
    axes = axes.ravel()

    for i in range(6):
        ax = axes[i]
        pts = group_points[i]
        edge_c, fill_c = palette[i]["edge"], palette[i]["fill"]

        front = pareto_front_2d(pts)
        hv = hypervolume_2d_max(pts, ref_point)

        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.35, color=edge_c, zorder=2)

        if front.size > 0:
            fill_hv_rects(ax, front, ref_point, fill_c)
            ax.plot(front[:, 0], front[:, 1], "--", linewidth=4.0, color=edge_c, zorder=3)
            ax.scatter(front[:, 0], front[:, 1], s=80, color=edge_c, zorder=4)

        ax.scatter([ref_point[0]], [ref_point[1]], s=40, color="black", zorder=5)

        # title = f"{group_names[i]}\nHV={hv:.4f}" if "qwen2.5-1.5b" in group_names[i] else f"\n\n{group_names[i]}\nHV={hv:.4f}"
        title = f"{group_names[i]}" if "qwen2.5-1.5b" in group_names[i] else f"\n\n{group_names[i]}"
        ax.set_title(
            title,
            fontsize=20,
            fontweight="bold",
            pad=12,
        )

        ax.grid(True, linestyle=":", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        ax.tick_params(axis="both", labelsize=18, width=2.5, length=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for r in range(2):
        axes[r * 3].set_ylabel("harmless", fontsize=20, fontweight="bold")
    for c in range(3):
        axes[3 + c].set_xlabel("helpful", fontsize=20, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to: {output_path}")

# ====== main ======
if __name__ == "__main__":
    experiments = [
        "qwen2.5-1.5b_grpo",
        "qwen2.5-1.5b_gdpo",
        "qwen2.5-1.5b_hvpo",
        "qwen2.5-3b_grpo",
        "qwen2.5-3b_gdpo",
        "qwen2.5-3b_hvpo",
    ]

    paths = [f"playground/visualization/scores/{exp}.jsonl" for exp in experiments]

    plot_pareto_grid_3x2_from_files(
        jsonl_paths=paths,
        output_path="playground/visualization/pareto_grid_3x2.pdf",
        ref_point=(0, 0),
    )
