#!/usr/bin/env python
"""HVPO vs GRPO 多目标对比分析 (safe: helpful vs harmless).

对每个模型的生成结果:
  1) 用两个奖励服务(50051 helpful, 50052 harmless)给每条 response 打分 -> 2维目标向量
  2) 计算各维均值、最小目标(min)、乘积等标量汇总
  3) 用 *相同的* 参考点(取两模型全体样本每维最小值再留 margin)计算主导超体积 HV
     - 公平: 两方法共享同一 ref point, 否则 HV 不可比
  4) 输出对比表 + 保存每模型的目标向量供画 Pareto 图
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/rihongqiu/data/code/Amo")
from recipe.amo_safe.reward_client import compute_reward_score

MODELS = ["grpo", "hvpo"]
RESULT_DIR = "/home/rihongqiu/data/code/Amo/results/PKU-SafeRLHF"
HOST = "localhost"


def score_model(tag):
    df = pd.read_parquet(f"{RESULT_DIR}/qwen2.5-1.5b_{tag}.parquet")
    vecs = []
    for _, row in df.iterrows():
        q = row["extra_info"]["question"]
        resp = row["responses"][0]
        h = compute_reward_score(q, resp, HOST, "50051")   # helpful
        k = compute_reward_score(q, resp, HOST, "50052")   # harmless
        vecs.append((float(h), float(k)))
    return np.array(vecs)  # shape (N, 2)


def hypervolume_2d(points, ref):
    """2D 主导超体积 (两目标都最大化).
    points: (N,2), ref: (2,) 下界参考点 (需被所有前沿点支配).
    做法: 取非被支配前沿, 按 x 升序阶梯累加 (x_i - x_{i-1})*(y_i - ref_y)。"""
    P = points[(points[:, 0] > ref[0]) & (points[:, 1] > ref[1])]
    if len(P) == 0:
        return 0.0
    # 非支配前沿 (最大化): 按 x 降序扫描, 保留 y 严格递增者
    order = P[np.argsort(-P[:, 0])]
    front = []
    best_y = -np.inf
    for x, y in order:
        if y > best_y:
            front.append((x, y))
            best_y = y
    # front 现按 x 降序、y 升序. 转 x 升序做阶梯积分
    fs = sorted(front, key=lambda p: p[0])   # x 升序 => y 降序
    hv = 0.0
    prev_x = ref[0]
    for x, y in fs:
        hv += (x - prev_x) * (y - ref[1])
        prev_x = x
    return hv


def main():
    scores = {m: score_model(m) for m in MODELS}

    # 公平 ref point: 两模型全体样本每维最小值 - margin
    allpts = np.vstack([scores[m] for m in MODELS])
    ref = allpts.min(axis=0) - 0.5   # 留 0.5 margin, 保证所有点被支配
    print(f"[ref point] {ref.tolist()}  (shared, for fair HV)")

    rows = []
    for m in MODELS:
        v = scores[m]
        hv = hypervolume_2d(v, ref)
        rows.append({
            "method": m.upper(),
            "helpful_mean": v[:, 0].mean(),
            "harmless_mean": v[:, 1].mean(),
            "min_obj_mean": np.minimum(v[:, 0], v[:, 1]).mean(),  # 每样本两目标最小值的均值(越大=越均衡)
            "hypervolume": hv,
            "n": len(v),
        })
        np.save(f"{RESULT_DIR}/vecs_{m}.npy", v)

    dfres = pd.DataFrame(rows)
    print("\n==== HVPO vs GRPO (safe: helpful ↔ harmless) ====")
    print(dfres.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    g = dfres[dfres.method == "GRPO"].iloc[0]
    h = dfres[dfres.method == "HVPO"].iloc[0]
    print("\n---- HVPO 相对 GRPO 提升 ----")
    print(f"  Hypervolume : {g.hypervolume:.4f} -> {h.hypervolume:.4f}  "
          f"({100*(h.hypervolume-g.hypervolume)/abs(g.hypervolume):+.1f}%)")
    print(f"  min-obj mean: {g.min_obj_mean:.4f} -> {h.min_obj_mean:.4f}  "
          f"({100*(h.min_obj_mean-g.min_obj_mean)/abs(g.min_obj_mean):+.1f}%)")
    print(f"  helpful mean: {g.helpful_mean:.4f} -> {h.helpful_mean:.4f}")
    print(f"  harmless mean: {g.harmless_mean:.4f} -> {h.harmless_mean:.4f}")

    dfres.to_json(f"{RESULT_DIR}/comparison_1.5b.json", orient="records", indent=2)
    print(f"\n[saved] {RESULT_DIR}/comparison_1.5b.json + vecs_*.npy")


if __name__ == "__main__":
    main()
