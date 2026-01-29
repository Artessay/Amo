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
    """Maximize both dims; return nondominated points sorted by helpful ascending."""
    if points.size == 0:
        return points
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.size == 0:
        return points

    # helpful 降序扫描，保留 harmless 严格递增点
    idx = np.argsort(points[:, 0])[::-1]
    pts = points[idx]

    best_y = -np.inf
    nd = []
    for x, y in pts:
        if y > best_y:
            nd.append((x, y))
            best_y = y

    nd = np.array(nd, dtype=float)
    nd = nd[np.argsort(nd[:, 0])]  # helpful 升序
    return nd

def hypervolume_2d_max(points: np.ndarray, ref: tuple[float, float]) -> float:
    """2D hypervolume (maximize) wrt ref."""
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

# def build_hv_polygon(front: np.ndarray, ref: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
#     """构造阶梯填充多边形，用于展示HV覆盖区域。"""
#     rx, ry = ref
#     if front.size == 0:
#         return np.array([rx]), np.array([ry])

#     front = front[np.argsort(front[:, 0])]  # helpful 升序
#     xs = [rx, front[0, 0]]
#     ys = [ry, ry]

#     cur_y = ry
#     for x, y in front:
#         xs.append(x); ys.append(cur_y)  # 横
#         xs.append(x); ys.append(y)      # 竖
#         cur_y = y

#     xs.append(front[-1, 0]); ys.append(ry)
#     xs.append(rx);           ys.append(ry)
#     return np.array(xs), np.array(ys)

def build_hv_polygon(front: np.ndarray, ref: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """构造 2D maximize 的HV覆盖区域阶梯多边形（ref在左下）。"""
    rx, ry = ref
    if front.size == 0:
        return np.array([rx, rx]), np.array([ry, ry])

    # helpful 升序
    front = front[np.argsort(front[:, 0])]

    xs = [rx]
    ys = [ry]

    prev_x = rx
    cur_y = ry

    for x, y in front:
        # 横：从 (prev_x, cur_y) -> (x, cur_y)
        xs += [prev_x, x]
        ys += [cur_y, cur_y]

        # 竖：从 (x, cur_y) -> (x, y)
        xs += [x]
        ys += [y]

        prev_x = x
        cur_y = y

    # 关到底边，再回到ref
    xs += [prev_x, rx]
    ys += [ry, ry]

    return np.array(xs), np.array(ys)

def fill_hv_rects(ax, front: np.ndarray, ref: tuple[float, float], color, alpha=0.25, zorder=1):
    rx, ry = ref
    if front.size == 0:
        return
    front = front[np.argsort(front[:, 0])]  # x升序
    prev_x = rx
    for x, y in front:
        if x > prev_x and y > ry:
            ax.fill_between([prev_x, x], [ry, ry], [y, y], color=color, alpha=alpha, zorder=zorder)
        prev_x = x



# ====== 3x3 绘图（输入9个jsonl路径） ======
def plot_pareto_grid_3x3_from_files(
    jsonl_paths: list[str],
    output_path: str = "pareto_grid_3x3.png",
    ref_point: tuple[float, float] | None = None,
    share_axes: bool = True,
    titles: list[str] | None = None,
):
    """
    jsonl_paths: 9个jsonl文件路径，每个文件=1组实验（不合并）
    titles: 可选，自定义每个子图标题（长度必须为9）；否则用文件名 stem
    """
    if len(jsonl_paths) != 9:
        raise ValueError(f"Expected 9 jsonl paths, got {len(jsonl_paths)}")

    # 读取每个文件
    group_names = titles if (titles is not None) else [Path(p).stem for p in jsonl_paths]
    group_points = []
    for p, name in zip(jsonl_paths, group_names):
        df = read_single_jsonl(p, label=name)
        pts = df[["helpful", "harmless"]].to_numpy(dtype=float) if not df.empty else np.zeros((0, 2), dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        group_points.append(pts)

    # 全局参考点：默认基于所有文件的全局最小值（保证可比）
    if ref_point is None:
        all_pts = np.vstack([p for p in group_points if p.size > 0]) if any(p.size > 0 for p in group_points) else np.zeros((0,2))
        if all_pts.size == 0:
            raise ValueError("All groups are empty.")
        min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
        max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
        pad_x = (max_x - min_x) * 0.05 if max_x > min_x else 1.0
        pad_y = (max_y - min_y) * 0.05 if max_y > min_y else 1.0
        ref_point = (float(min_x - pad_x), float(min_y - pad_y))

    # 统一坐标范围（便于跨组比较）
    if share_axes:
        all_pts = np.vstack([p for p in group_points if p.size > 0])
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        x_pad = 0.06 * (x_max - x_min + 1e-9)
        y_pad = 0.06 * (y_max - y_min + 1e-9)
        xlim = (min(ref_point[0], x_min) - x_pad, x_max + x_pad)
        ylim = (min(ref_point[1], y_min) - y_pad, y_max + y_pad)
    else:
        xlim = ylim = None

    # 颜色（示意图风格：虚线+半透明填充）
    palette = [
        {"edge": "#d62728", "fill": "#f4a6a6"},
        {"edge": "#1f77b4", "fill": "#9ecae1"},
        {"edge": "#2ca02c", "fill": "#a1d99b"},
        {"edge": "#ff7f0e", "fill": "#fdd0a2"},
        {"edge": "#9467bd", "fill": "#dadaeb"},
        {"edge": "#8c564b", "fill": "#e7cb94"},
        {"edge": "#17becf", "fill": "#9edae5"},
        {"edge": "#7f7f7f", "fill": "#d9d9d9"},
        {"edge": "#bcbd22", "fill": "#dbdb8d"},
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=share_axes, sharey=share_axes)
    axes = axes.ravel()

    for i in range(9):
        ax = axes[i]
        pts = group_points[i]
        style = palette[i]
        edge_c, fill_c = style["edge"], style["fill"]

        front = pareto_front_2d(pts)
        hv = hypervolume_2d_max(pts, ref_point)

        # 散点
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.35, color=edge_c, edgecolors="none", zorder=2)

        # HV 填充 + 前沿
        if front.size > 0:
            # xs, ys = build_hv_polygon(front, ref_point)
            # ax.fill(xs, ys, color=fill_c, alpha=0.25, zorder=1)
            fill_hv_rects(ax, front, ref_point, fill_c, alpha=0.25, zorder=1)
            ax.plot(front[:, 0], front[:, 1], linestyle="--", linewidth=2.0, color=edge_c, zorder=3)
            ax.scatter(front[:, 0], front[:, 1], s=26, color=edge_c, zorder=4)

        # 参考点
        ax.scatter([ref_point[0]], [ref_point[1]], s=18, color="black", zorder=5)

        title = f"{group_names[i]}\nHV={hv:.4f}" if "qwen2.5-1.5b" in group_names[i] else f"\n\n{group_names[i]}\nHV={hv:.4f}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # 外圈坐标轴标签
    for r in range(3):
        axes[r * 3].set_ylabel("harmless")
    for c in range(3):
        axes[6 + c].set_xlabel("helpful")

    # fig.suptitle("Pareto Fronts (3×3) + Hypervolume Coverage", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved to: {out.resolve()}")

if __name__ == "__main__":
    
    experiments = [
        "qwen2.5-1.5b_grpo", "qwen2.5-1.5b_gdpo", "qwen2.5-1.5b_hvpo",
        "qwen2.5-3b_grpo", "qwen2.5-3b_gdpo", "qwen2.5-3b_hvpo",
        "llama3.2-3b_grpo", "llama3.2-3b_gdpo", "llama3.2-3b_hvpo",
    ]

    paths = [f"playground/visualization/scores/{exp}.jsonl" for exp in experiments]
    print(paths)

    plot_pareto_grid_3x3_from_files(
        jsonl_paths=paths,
        output_path="playground/visualization/pareto_grid_3x3.png",  # 或 "pareto_grid_3x3.pdf"
        ref_point=(0, 0),                      # 或手动指定 (rx, ry)
        # ref_point=None,                      # 或手动指定 (rx, ry)
        share_axes=True,
        titles=None,                         # 可传入长度为9的自定义标题列表
    )
