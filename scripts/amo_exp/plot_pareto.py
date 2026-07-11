#!/usr/bin/env python
"""画 HVPO vs GRPO 的目标空间散点 + Pareto 前沿 (safe: helpful vs harmless)。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = "/home/rihongqiu/data/code/Amo/results/PKU-SafeRLHF"


def pareto_front(points):
    """max-max 非支配前沿, 返回按 x 升序的前沿点。"""
    order = points[np.argsort(-points[:, 0])]
    front, best_y = [], -np.inf
    for x, y in order:
        if y > best_y:
            front.append((x, y)); best_y = y
    return np.array(sorted(front, key=lambda p: p[0]))


fig, ax = plt.subplots(figsize=(7, 6))
colors = {"grpo": "#d62728", "hvpo": "#1f77b4"}
labels = {"grpo": "GRPO (weighted-sum)", "hvpo": "HVPO (hypervolume)"}

for m in ["grpo", "hvpo"]:
    v = np.load(f"{RESULT_DIR}/vecs_{m}.npy")
    ax.scatter(v[:, 0], v[:, 1], s=14, alpha=0.35, c=colors[m], label=f"{labels[m]}  (mean=({v[:,0].mean():.2f},{v[:,1].mean():.2f}))")
    pf = pareto_front(v)
    ax.plot(pf[:, 0], pf[:, 1], "-o", c=colors[m], lw=2, ms=5, alpha=0.9)

ax.set_xlabel("Helpfulness reward", fontsize=12)
ax.set_ylabel("Harmlessness reward", fontsize=12)
ax.set_title("HVPO vs GRPO on PKU-SafeRLHF (Qwen2.5-1.5B, 50 steps)\nscatter = per-prompt objectives, line = Pareto front", fontsize=11)
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = f"{RESULT_DIR}/pareto_1.5b.png"
fig.savefig(out, dpi=130)
print(f"[saved] {out}")
