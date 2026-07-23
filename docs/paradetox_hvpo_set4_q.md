# HVPO-Set4-Q：ParaDetox 上的 HVPO v2 优化方案

**日期**：2026-07-21  
**状态**：设计提案，尚未实现  
**适用实验**：ParaDetox，Qwen2.5-1.5B-Instruct，训练每个 prompt 采样 8 个 response，验证主指标为 HV@4

## 结论摘要

当前 200-step 训练已经完成，但正式聚合分析暂时只到 step 80。step 80 的描述性点估计显示：legacy HVPO 产生了更多非支配点、减少了重复，同时 HV@1、HV@2、HV@4 和 STA 的均值低于 linear-GRPO。这里不能据此断言 HVPO 已稳定损害 HV@4：该指标的 paired bootstrap 95% CI 仍包含 0，且结果只来自一个训练 seed。

下一版建议采用 **HVPO-Set4-Q**：

1. 把训练目标从完整 8-response 集合的 exclusive-HV 改为与评估一致的 **expected marginal HV@4**；
2. 用 singleton-HV，即三目标乘积，作为质量锚，抑制“高 SIM、低 STA”的极端点；
3. 删除 distance fallback、running min-max 和 prompt-group z-score；
4. 用冻结 base model 生成的 calibration artifact 做全局固定标准化；
5. HV 全程使用 float64，固定 `[0,1]^3` 和原点 reference point；
6. 先用 60-step 因果消融确认每项改动的贡献，再进行 3-seed、200-step 比较。

推荐的默认混合为：

\[
A_i = \frac{0.75 Z_i^{H} + 0.25 Z_i^{Q}}{s_{\mathrm{mix}}}.
\]

HV 仍是主目标，质量项只作为 guardrail。若质量门槛仍未通过，再补跑 `0.5 HV + 0.5 Q`，不预先扩大搜索空间。

## 1. 当前训练证据

### 1.1 验证轨迹

当前结果目录为 `results/ParaDetox/long200_s42_v64`。两种方法均使用 seed 42；下表来自相同 64 个 validation prompts、每个 prompt 4 个 responses 的配对比较。

| Step | GRPO HV@4 | HVPO HV@4 | HVPO - GRPO |
|---:|---:|---:|---:|
| 0 | 0.708247 | 0.708247 | 0.000000 |
| 20 | 0.689493 | 0.684735 | -0.004758 |
| 40 | 0.688725 | 0.680834 | -0.007891 |
| 60 | 0.668678 | 0.680130 | +0.011452 |
| 80 | 0.683532 | 0.639226 | -0.044305 |

step 80 的主要指标如下：

| 指标 | GRPO | HVPO | HVPO - GRPO | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| HV@1 / joint product | 0.519830 | 0.472076 | -0.047754 | `[-0.088857, -0.007692]` |
| HV@2 | 0.611615 | 0.557448 | -0.054167 | `[-0.103200, -0.007519]` |
| HV@4 | 0.683532 | 0.639226 | -0.044305 | `[-0.109311, 0.013947]` |
| Linear reward | 0.805696 | 0.778452 | -0.027244 | `[-0.048613, -0.006572]` |
| STA | 0.764504 | 0.698928 | -0.065575 | `[-0.125608, -0.006918]` |
| SIM | 0.822936 | 0.831142 | +0.008207 | `[-0.016833, 0.030268]` |
| Fluency | 0.829649 | 0.805286 | -0.024364 | `[-0.050497, 0.002682]` |

在 step 80 的描述性点估计中，HVPO 的平均非支配点数由 GRPO 的 `2.703` 增至 `2.859`，重复输出率由 `0.195` 降至 `0.172`。这些数值提示 HVPO 可能增加了 trade-off/diversity，但新增点未在该 checkpoint 的均值上把前沿推向更好的绝对位置；它们不是跨 seed 的稳定结论，而且 HV@4 差值的置信区间仍包含 0。

### 1.2 rollout 机制诊断

steps 1--20 的 rollout 重放给出以下证据：

- 训练使用 8 个 rollouts 的 full-set exclusive credit，而验证预算是 4 个 responses；
- expected marginal credit 的均值在 `k=4` 时为 `0.03238`，在 `k=8` 时仅为 `0.00949`；
- `k=4` credit 与 equal-linear reward 的 Pearson 相关系数约为 `0.678`，符号冲突约 `26.4%`；`k=8` 时相关约为 `0.389`，符号冲突约为 `42.1%`；
- dominated sample 的 distance fallback 绝对均值约为正 HV credit 均值的 `7.37x`；
- 当前 hybrid advantage 与纯 exclusive-HV advantage 的相关系数只有 `0.597`，符号不同的 rollout 占 `41.4%`；
- float32 与 float64 的 exclusive-HV 在约 `2.58%` rollout 上出现“严格正值 / 零值”分类分歧。

这些数字支持三个工作假设：

1. **预算错配**：full-set HV@8 credit 不等价于模型部署和验证时的 HV@4；
2. **fallback 可能扰动 HV 信号**：fallback 的量级和符号差异值得警惕，但现有重放不能把它与其他机制分离，是否造成性能下降必须通过关闭 fallback 的受控消融验证；
3. **缺少绝对质量约束**：极端点只要扩张前沿就可能得到正 credit，即使某个关键目标很低。

### 1.3 结论边界

当前实验只能用于设计和诊断，不能作为最终论文结论：

- 只有一个训练 seed；
- 当前长跑使用旧 split，而旧 test 中有 `271/393` 个 normalized toxic prompts 与 train 重合；
- 当前用于 step 80 比较的 64 个 validation prompts 中，也有 `43/64` 个 normalized toxic prompts 与 train 重合；
- step 200 的完整 paired summary 尚未生成；
- HV@4 在 step 80 的置信区间仍包含 0，虽然 HV@1、HV@2、linear reward 和 STA 的下降已经更明确。
- paired bootstrap 只刻画固定 seed、固定 checkpoint 下 validation prompts 的采样不确定性，不覆盖训练随机性；本文同时查看了多个 checkpoint 和多个指标，未做多重比较校正，因此这些区间和显著性线索只能作描述性诊断。

因此本文把上述机制称为“有数据支持的假设”，后续必须在 leakage-safe split 和多 seed 上验证。

## 2. 设计目标

HVPO v2 需要同时满足：

1. **目标一致**：训练 credit 对应验证时 4-response set 的 hypervolume；
2. **保留多目标性质**：HV 是主项，不退化为 linear scalarization；
3. **守住绝对质量**：低 STA 或低 fluency 的极端点不能仅凭独特性获得主要正向更新；
4. **可解释**：每个 advantage component 都对应明确目标，不由 fallback 或在线缩放隐式改变；
5. **数值稳定**：小 HV 差分不受 float32 舍入和移动 bounds 放大；
6. **可归因**：通过最小消融分别验证 fixed calibration、`k=4` 对齐和质量锚。

## 3. HVPO-Set4-Q

### 3.1 固定目标空间

对同一 prompt 的 8 个 responses，三目标记为：

\[
r_i=(r_i^{STA}, r_i^{SIM}, r_i^{FL}).
\]

统一映射为：

\[
z_{id}=\operatorname{clip}(r_{id},0,1), \qquad
h(S)=HV(\{z_j:j\in S\};\mathbf 0).
\]

规则如下：

- 目标顺序固定为 `detox_sta, detox_sim, detox_fluency`；
- bounds 固定为 `[0,1]^3`，不再使用训练中的 running min-max；
- reference point 固定为 `[0,0,0]`；
- 与当前评估保持一致，SIM 的负值在 HV 计算前截为 0；
- 全部目标都是越大越好。

固定坐标系使不同 step、不同方法、不同 seed 的 HV credit 可比，也避免 STA 饱和区间的微小噪声被在线 min-max 放大。

### 3.2 与 HV@4 对齐的 expected marginal credit

目标定义为 4 个独立 responses 构成集合时的期望 hypervolume：

\[
J_H(\theta)=
\mathbb E_{x,\,y_{1:4}\sim\pi_\theta}
\left[h(\{y_1,y_2,y_3,y_4\})\right].
\]

训练仍为每个 prompt 采样 `n=8` 个 responses。对第 `i` 个 response，计算精确 U-statistic marginal：

\[
c_i^{(4)}=
\frac{1}{\binom{7}{3}}
\sum_{\substack{T\subseteq G\setminus\{i\}\\|T|=3}}
\left[h(T\cup\{i\})-h(T)\right].
\]

用于策略梯度的 HV 原量为：

\[
g_i=4c_i^{(4)}.
\]

在 on-policy、未做 PPO ratio clipping 的 score-function estimator 下，`4c_i` 给出 `J_H` 的无偏策略梯度估计。直观上，它衡量“当部署预算为 4 时，response `i` 平均能为另外 3 个 responses 增加多少 HV”。

相比 full-set exclusive-HV@8，这个定义有三个优点：

- 与主评估指标严格使用同一个 response budget；
- 一个点即使在 8 个点的完整集合中被支配，也可能在不同 3-peer 子集中提供有效增量，因此信号更密；
- 不需要为零贡献点额外发明 distance fallback。

### 3.3 singleton-HV 质量锚

定义：

\[
q_i=h(\{i\})=\prod_d z_{id}.
\]

`q_i` 正好等于现有分析中的 HV@1 / joint product，不是另加一套任意 linear scalarization。乘积会对任一目标接近 0 的响应给出低分，因此能直接抑制当前观察到的“高 SIM、低 STA”极端点。

质量锚不替代 HV@4：HV 仍负责集合互补性和多样性，singleton-HV 只约束每个 response 的绝对位置。

### 3.4 冻结 calibration 与最终 advantage

使用冻结 base model 在独立 calibration prompts 上生成 responses，预先计算：

\[
\mu_g,s_g=\operatorname{Mean/Std}(g_i),\qquad
\mu_q,s_q=\operatorname{Mean/Std}(q_i).
\]

训练中使用固定标准化：

\[
Z_i^H=\frac{g_i-\mu_g}{s_g},\qquad
Z_i^Q=\frac{q_i-\mu_q}{s_q}.
\]

默认混合：

\[
\widetilde A_i=0.75Z_i^H+0.25Z_i^Q.
\]

为了使不同混合权重具有一致的初始更新尺度，再从 calibration artifact 中冻结：

\[
s_{\mathrm{mix}}=
\operatorname{Std}_{cal}(0.75Z^H+0.25Z^Q),
\qquad
A_i=\frac{\widetilde A_i}{s_{\mathrm{mix}}}.
\]

默认不做 advantage clipping，只监控 `|A| > 10` 的比例。任一 calibration std 小于 `1e-3` 时 fail-fast，不退回在线统计。

固定 mean 是 action-independent baseline，固定 scale 只改变目标权重。在 PPO clipping 之前，该 estimator 对相应的固定 HV/quality 混合目标是无偏的；应用 PPO clipped surrogate 后，不再声称完整更新严格无偏。

### 3.5 禁止 prompt-group z-score

legacy 路径会把 reward 再按当前 prompt 的 8 个 samples 中心化并除以组内 std。对于普通单样本 reward，这常被当作 GRPO baseline；但 `c_i` 是 set-dependent credit，其他成员 `c_j` 的 peer subset 以 `3/7` 的概率包含动作 `i`。因此 group mean 不是相对于动作 `i` 独立的 baseline，group std 还会动态改变不同 prompts 的权重。

HVPO-Set4-Q 应新增 passthrough advantage estimator：直接把已经校准的 sequence-level `A_i` 广播到 response tokens，不再中心化或 whiten。

## 4. 删除的 legacy 机制

新模式明确关闭：

- `distance_metric=chebyshev/euclidean` fallback；
- running min-max objective normalization；
- global/recent Pareto cache；
- reward-manager 内的 z-score/min-max scaling；
- GRPO prompt-group mean/std normalization；
- float32 HV difference。

保留 legacy 模式和旧配置，仅用于复现实验；不能静默改变旧 checkpoint 的语义。

## 5. 数值和复杂度规则

- 从 raw objective tensor、subset HV、marginal difference、calibration 标准化到最终 `A_i` 的整个数值链路都使用 float64；只在把最终 token reward/advantage 写回训练 tensor 时转换到模型 dtype；
- objective 名称和顺序必须严格等于 `detox_sta, detox_sim, detox_fluency`，不能仅依赖 custom reward path 或字典的插入顺序；
- 原点必须严格固定为 `[0,0,0]`：clip 后显式检查所有 objective 均弱支配原点，`hvpo_set` 不得复用 legacy `_compute_reference_point` 中按 batch minimum 下拉 reference point 的 clamp；
- 4-subsets 与 3-subsets 分别缓存，`n=8,k=4` 只需计算
  \(\binom84+\binom83=70+56=126\) 次小型 HV；
- 共形成 `8 * C(7,3) = 280` 个 marginal differences；
- 重复 responses 按采样 index 保留，符合固定 response-budget 的定义；
- 差分落在 `[-1e-12,0)` 时视为数值误差并截为 0，小于 `-1e-12` 时直接报错；
- NaN/Inf、目标顺序不符、group size 与 calibration artifact 不一致或 `n<k` 均 fail-fast；
- 不启用 epsilon-Pareto cache；reward model 推理仍应是主要开销。

## 6. 提议配置

以下是 v2 的目标配置接口，不代表这些键已经在当前代码中实现：

```yaml
reward_manager:
  source: register
  name: amo_hvpo

algorithm:
  adv_estimator: hvpo_set
  use_kl_in_reward: false

amo_strategy:
  hv_config:
    credit_assignment: expected_subset_marginal
    subset_size: 4
    subset_estimator: exact
    max_exact_hv_evals: 4096

    objective_normalization: fixed_bounds
    objective_lower_bounds: [0.0, 0.0, 0.0]
    objective_upper_bounds: [1.0, 1.0, 1.0]
    reference_point: [0.0, 0.0, 0.0]

    hv_dtype: float64
    hv_negative_tolerance: 1.0e-12
    distance_metric: none
    pareto_front_scope: intra_group
    reward_scaling_mode: none

    advantage_normalization: fixed_calibration
    calibration_path: null
    calibration_scale_floor: 1.0e-3
    advantage_clip: null

    hv_coef: 0.75
    quality_anchor: singleton_hv
    quality_anchor_coef: 0.25
```

`calibration_path=null` 在 `hvpo_set` 模式下必须报错，不能回退到 legacy 行为。

配置加载时必须双向 fail-fast：`algorithm.adv_estimator=hvpo_set` 只允许搭配
`reward_manager.name=amo_hvpo` 且
`credit_assignment=expected_subset_marginal`；反过来，启用该 credit mode
也必须使用 `hvpo_set` estimator。任何 manager、estimator、credit mode 的不一致
都应在训练启动前报错，不能静默退回 legacy HVPO 或默认 reward manager。

`hvpo_set` 还必须强制 `algorithm.use_kl_in_reward=false`。reward manager 输出的
已经是校准后的 sequence-level `A_i`，若再把逐 token KL 混入
`token_level_rewards`，passthrough estimator 的语义会被破坏。actor loss 中独立的
KL regularization 可以保留。

## 7. Calibration 与数据协议

当前 leakage-safe grouped split 含 11,689 个 train prompts 和 238 个 final-test prompts。建议从 train 按 normalized source 的稳定 hash 划分：

- calibration：256 prompts；
- dev：128 prompts；
- 实际训练：11,305 prompts；
- final test：原 238 prompts，配置冻结前完全不使用。

用冻结的 Qwen2.5-1.5B-Instruct、与训练相同的 sampling 参数、`n=8` 在 calibration 上生成 2,048 个 responses。artifact 至少保存：

```json
{
  "objective_names": ["detox_sta", "detox_sim", "detox_fluency"],
  "objective_bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
  "reference_point": [0.0, 0.0, 0.0],
  "group_size": 8,
  "subset_size": 4,
  "hv_component": {"center": "...", "scale": "..."},
  "quality_component": {"center": "...", "scale": "..."},
  "component_covariance": "...",
  "mix_scale": "...",
  "data_fingerprint": "...",
  "model_fingerprint": "..."
}
```

所有消融和 seed 共用同一个 artifact；训练中绝不更新。

## 8. 代码落点

建议以新模式增量实现：

1. `verl/workers/reward_manager/amo_utils/hybrid_reward.py`
   - 新增 `compute_expected_subset_marginals(...)`；
   - 返回 set HV、每个 `c_i` 和数值诊断；所有 HV 输入、输出和差分保持 float64。
2. `verl/workers/reward_manager/amo_hvpo.py`
   - 增加 `expected_subset_marginal` credit mode；
   - 加载并校验 fixed calibration；
   - 严格校验 `list(compute_score.keys())` 与 artifact 中的 `objective_names` 完全同序；
   - 从 raw objective 构造到最终 `A_i` 全程保持 float64，仅在写回训练 tensor 时转换 dtype；
   - 在新 mode 中使用严格固定原点并绕开 legacy reference-point clamp；
   - 计算 quality anchor 和最终 sequence advantage；
   - 日志记录 `subset_hv`、`subset_marginal`、`quality_anchor`、两个标准化 component 及最终 advantage。
3. `verl/trainer/ppo/core_algos.py`
   - 新增 `AdvantageEstimator.HVPO_SET="hvpo_set"`；
   - estimator 只广播预计算 advantage，不做组内中心化或标准化。
4. `verl/trainer/ppo/reward.py`
   - 对 `hvpo_set` 同样注入 `hv_config`。
   - 启动时校验 manager、estimator、credit mode 和 `use_kl_in_reward` 的合法组合并 fail-fast。
5. `verl/trainer/config/ppo_trainer.yaml`
   - 增加上述 v2 配置，legacy 默认值保持不变。
6. `verl/trainer/ppo/ray_trainer.py`（或其 validation metric 聚合入口）
   - 按 prompt 独立计算并记录 `val-amo/hv_at_4`；
   - 明确保留 `val-core/reward` 的现有含义：当前 manager 的 validation reward 是 raw-objective mean，不是 HV@4，不能用它替代主指标。
7. `scripts/paradetox/calibrate_hvpo.py`
   - 从冻结 base-model rollout 生成严格 JSON calibration artifact。
8. `scripts/paradetox/run_pilot.sh`
   - 增加 `hvpo_set` 方法和 artifact 参数。
9. `tests/amo/test_hvpo.py`
   - 覆盖精确枚举、排列等变、重复点、差分非负、float64、artifact 校验以及 passthrough estimator。
10. `tests/amo/test_hvpo_wiring.py`
    - 覆盖合法配置，并断言 manager/estimator/credit mode 不一致、启用 in-reward KL、objective 名称或顺序不一致时均在启动阶段失败；
    - 覆盖 validation 的独立 HV@4 指标，防止把 `val-core/reward` 误标为 HV@4。

## 9. 最小因果消融

使用新 split、seed 42、60 training steps、dev 128，每 20 steps 验证：

| 编号 | 方法 | 目的 |
|---:|---|---|
| 1 | GRPO-linear | 新 split 上的公平基线 |
| 2 | Legacy HVPO | 复现当前问题 |
| 3 | Set8-Fixed，纯 HV | 隔离 fixed bounds/calibration、无 fallback 和无 group z-score |
| 4 | Set4-Fixed，纯 HV | 在 3 的基础上隔离 response-budget 对齐 |
| 5 | Set4-Q，`0.75 HV + 0.25 Q` | 在 4 的基础上隔离质量锚 |

只有第 5 项未通过质量 guardrail 时，补跑 `0.5 HV + 0.5 Q`。

为避免把 optimizer 改动误认为 credit 改进，五个方法必须使用完全相同的优化配置。建议统一将 actor learning rate 从当前 `1e-5` 降到约 `2e-6`，增加约 5% warmup；KL 与 entropy 设置在这一轮保持一致。若所有方法仍出现明显 entropy collapse/KL 激增，再单独进行稳定性消融。

### 9.1 60-step 晋级条件

用 step 40 和 step 60 的均值筛选：

- 相对 GRPO 的 HV@4 至少 `+0.01`；
- paired bootstrap `P(delta HV@4 > 0) >= 0.8`；
- HV@1 相对 GRPO 不低于 `-0.01`；
- STA、SIM、fluency 任一均值相对 GRPO 不低于 `-0.02`。

最多晋级一个 v2，与 GRPO 进行 200-step、seeds `42/43/44` 比较。

### 9.2 进入完整 epoch 的条件

- 三 seed 平均 `delta HV@4 >= +0.01`；
- 至少 2/3 seeds 的 delta 为正；
- 分层 paired bootstrap 95% CI 下界大于 0；
- HV@1 相对 GRPO 不低于 `-0.005`；
- 任一单目标均值相对 GRPO 不低于 `-0.02`。

配置冻结后，才在 238 个 final-test prompts 上做一次最终评测。

## 10. 如何解释消融结果

- **Set8-Fixed > legacy**：主要问题来自 fallback、在线缩放或组标准化；
- **Set4-Fixed > Set8-Fixed**：response-budget 错配是关键原因；
- **Set4-Q 恢复 HV@1/STA 且保持 HV@4**：极端低质量点是关键原因，质量锚有效；
- **所有 HVPO v2 均落后，但 GRPO 稳定**：需要重新检查 reward model 的分辨率、目标定义或 ParaDetox 的真实 Pareto 结构；
- **所有方法均随 step 恶化**：优先处理 actor LR、warmup、KL/entropy，而不是继续改 HV credit；
- **Set4-Q 仍以 STA 换 SIM**：先提高质量权重到 0.5；只有仍失败时，才考虑加入基于 GRPO/dev 基线的 STA soft constraint，避免一开始就把 HVPO 变成硬阈值算法。

## 11. 成功标准

本方案的目标不是证明每个 checkpoint 都超过 GRPO，而是验证以下因果链：

```text
与 HV@4 对齐的 set credit
  + 删除量级失衡、会大幅扰动部分更新方向的 distance fallback
  + 固定且可比较的数值尺度
  + singleton-HV 质量锚
  -> 保留非支配解多样性
  -> 不再牺牲 STA / HV@1
  -> 在多 seed 上提升 paired HV@4
```

只有 leakage-safe、3-seed 的 200-step 结果同时满足 HV@4 提升和质量 guardrails，才认为 HVPO-Set4-Q 在 ParaDetox 上通过快速验证。
