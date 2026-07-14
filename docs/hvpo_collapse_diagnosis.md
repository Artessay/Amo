# P0 诊断：HVPO 奖励塌缩的因果验证

**日期**：2026-07（塌缩对照实验）
**结论**：LLM 版 HVPO 表现不佳的根因是**膨胀的全局 Pareto 前沿导致 exclusive-HV 贡献塌缩为零**，而非 HVPO 算法思想本身有问题。修复方向（P0）是把 LLM 版的前沿管理**对齐到 moo_suite 已验证有效的形式**（组内 / 小滑动窗口，而非跨 batch 持久累积到 1024 的全局 cache）。

---

## 背景：两个 HVPO 实现并未对齐

| | moo_suite 的 HVPO（`HVPOSurvival`） | LLM 的 HVPO（`amo_hvpo.py`） |
|---|---|---|
| exclusive-HV 在什么集合内算 | **仅当前种群**，每代重算 bound | **当前 group ∪ 持久累积、膨胀到 1024 的全局 cache** |
| 参考点 / bound | 每代从种群 nadir 重算，尺度固定可比 | 静态 `[0,0]` + clamp，跨 step 漂移 |
| 实测表现 | 22 题赢 10（HV），碾压 vanilla | 信号塌缩：训练日志中 **94% 样本 HV 贡献 = 0** |

## reward-manager 层的直接证据

同一个 4-样本 rollout group、同一组分数，仅改变背景前沿：

| 场景 | 组内奖励 | std | 正贡献数 |
|---|---|---|---|
| 空前沿 / 组内（= suite 形式） | `[2.5, 5.0, 0.25, -1.5]` | **2.82** | 3/4 |
| 膨胀全局前沿（= LLM 病灶） | `[0, 0, 0, -1.5]` | **0.75** | **0/4** |

→ 背景前沿膨胀后，信号从"有强区分度"塌成"几乎全零"。

## moo_suite 塌缩对照实验（决定性）

在**同一个 GA 框架**内，仅把 HVPO 的 exclusive-HV 从"当前种群内"改成"对着一个不断累积、膨胀的全局 archive（FIFO cap=1024，精确复现 LLM 版 cache 行为）"，其余（问题、算子、种群规模、seed）完全相同。

**结果（4 问题 × 3 seed 平均，HV 越高越好 / IGD 越低越好）：**

| 方法 | HV | IGD |
|---|---|---|
| **hvpo（对齐版）** | **0.7008** | **0.0570** |
| hvpo_collapsed（膨胀全局前沿） | 0.5194 | 0.1652 |
| vanilla（线性加权） | 0.2195 | 0.7233 |
| random（下界） | 0.0469 | 2.2130 |

逐题一致：zdt1 / zdt2 / dtlz2 / wfg4 上 `hvpo_collapsed` 的 HV 均显著低于对齐版、明显向 vanilla 方向下沉（dtlz2、wfg4 尤为剧烈，HV 从 0.74→0.51、0.72→0.44）。

## 因果链（已闭环验证）

```
膨胀全局前沿 cache
  → group 内几乎所有点被前沿支配
  → exclusive-HV 贡献 ΔHV ≈ 0（94% 样本）
  → 训练信号塌缩为噪声
  → HVPO 退化，向 vanilla/random 下沉
```

移除膨胀 cache / 对齐到组内形式 → 信号恢复 → HVPO 重新变强（suite 已证明赢 10/22）。

## 后续行动（P0）

把 LLM 版 `verl/workers/reward_manager/amo_hvpo.py` 的前沿管理**对齐**成 suite 形式：
- 用组内 / 小滑动窗口前沿（会遗忘），而非膨胀到 1024 的持久全局 cache；
- 目标值归一化 + 每 batch 固定尺度参考点，使 HV 跨 step 可比。

## P0 已实施（2026-07）

**改动文件**：
- `verl/workers/reward_manager/amo_hvpo.py`：
  - 新增 `pareto_front_scope`，默认 **`intra_group`**（P = ∅，组内 exclusive-HV，对齐 suite 的 `HVPOSurvival`）；保留 `recent_window` / `global_cache` 仅供消融。`intra_group` 下**不再更新持久前沿**，从根上杜绝膨胀。
  - 新增 `normalize_objectives`（默认 True）：用 running min/max 把 RM 原始分数归一化到 [0,1] 再算 HV，配固定原点参考点，使 HV 跨 step 可比。验证期日志仍用**原始分数**均值，保持与 GRPO 基线可比。
- `verl/trainer/config/ppo_trainer.yaml`：新增并注释上述两个默认项。
- `tests/amo/test_hvpo.py`：新增 4 个 P0 测试（塌缩对照 + 默认配置 + 归一化），**全部 17 个测试通过**。

**验证**：
- 单元层：`intra_group` 空前沿 → 组内 3/4 正贡献、信号有区分度；人为膨胀前沿 → 正贡献塌为 0（复现病灶）。
- 端到端冒烟：真实 RM 尺度 group（有 trade-off）→ 归一化后 3/4 正贡献、std≈0.49，对比旧版训练日志 94% 塌缩为 0，信号已恢复。

**下一步**：P1（reward_scaling_mode=z-score + dynamic_batch 参考点 margin）→ 1.5B LLM 训练验证（任务待定）。

**重要提醒（任务天花板，独立于本次修复）**：PKU-SafeRLHF 两目标 trade-off 很弱（Pearson r≈0.79，冲突样本仅 0.7%），Pareto 前沿接近退化成一个点。对齐修复能让 HVPO 恢复成"真正有效的多目标方法"，但**能否在 PKU 上超过线性加权 GRPO 不作保证**——强 trade-off 任务（MATH accuracy↔conciseness、News 4 目标）才是 HVPO 真正的用武之地。

## 代码去向

塌缩对照脚本 `_collapse_experiment.py` 仅用于本次一次性诊断，**已移至 `playground/dustbin/_collapse_experiment.py`**，不保留在主基准代码中，避免未来误用塌缩版本。`playground/benchmarks/moo_suite.py` 主代码**未被改动**。
