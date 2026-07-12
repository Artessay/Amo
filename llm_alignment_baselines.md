# HVPO 的 LLM 多目标对齐 Baseline 调研与复现建议

> 调研日期：2026-07-11。本文只把论文原文、会议页面和作者公开仓库视为一手来源；“未找到官方代码”表示截至调研日期，论文页面和公开检索结果中没有可确认的作者实现，不代表代码一定不存在。

## 1. 结论先行

HVPO 当前最需要的不是再加一批名字相近的方法，而是补齐几条互相正交、能解释收益来源的技术路线。建议按下面顺序复现。

| 优先级 | Baseline | 回答的问题 | 与当前 Amo 的可比性 | 预计改动 |
|---|---|---|---|---|
| P0 | 固定权重 Linear Scalarization sweep | HVPO 是否只胜过某一个不合适的等权点？ | 最高 | 很小 |
| P0 | Augmented Tchebycheff-GRPO | 收益是否仅来自非线性标量化？ | 最高 | 小 |
| P0 | GDPO | 逐目标归一化能否解释收益？ | 已有 | 已完成 |
| P0 | RVPO | soft-min / reward variance penalty 能否替代 HV credit？ | 最高，是 GDPO 的直接扩展 | 很小 |
| P0 | CTWA | 保持每个目标与训练信号正协方差能否避免目标退化？ | 很高，官方实现基于 verl/GRPO | 中 |
| P0 | Safe-RLHF / Lagrangian-GRPO | 把一个目标当硬约束是否更合适？ | 高，尤其 PKU-SafeRLHF | 中 |
| P0 | Fair and Stable Reward Composition | 动态调整目标权重能否替代 HV credit？ | 高 | 中 |
| P0 | MGDA、PAMA 与 GAPO | 显式处理目标梯度冲突或近似 Pareto 方向是否更有效？ | 高，部分方法训练开销更大 | 中到大 |
| P0 | Dynamic Reward Weighting | checkpoint 级 HV 引导或梯度动态权重是否更有效？ | 很高，同样基于在线 RL/verl | 中到大 |
| P0 | NSGA-II rank/crowding 与 SMS-EMOA-style credit | HVPO 是否胜过常见 Pareto 排序/生存规则？ | 高，属于 MOO 消融 | 小到中 |
| P1 | HaM | 直接最大化多个 LLM policy 的总体 HV 是否更好？ | 概念最相关，训练范式不同 | 大 |
| P1 | VPO | 训练一个模型生成覆盖 reward front 的候选集合是否更好？ | response-set 视角很相关 | 大，需 multi-answer rollout |
| P1 | MORLHF / Fine-Grained RLHF | 经典多 reward RLHF 能达到什么前沿？ | 中到高 | 中 |
| P1 | Rewarded Soups | 单目标专家的参数插值是否足够？ | 中 | 中，需训练多个专家 |
| P1 | RiC | reward-conditioned SFT 能否低成本覆盖前沿？ | 中，离线 SFT 范式 | 大 |
| P1 | Panacea / CLP | 单个 preference-conditioned policy 能否覆盖整条前沿？ | 中，需改模型输入或 LoRA | 大 |
| P1 | MODPO / CPO / MO-ODPO | 直接偏好优化能否胜过在线 policy gradient？ | 训练数据与损失不同 | 大 |
| P2 | DPA、MetaAligner、HoE、MOD/DeAL | 后处理、输出改写或解码时组合是否足够？ | 较低，适合作为扩展实验 | 中到大 |

**推荐的主表最小集合**：`GRPO-LS sweep`、`Tchebycheff-GRPO`、`GDPO`、`RVPO`、`CTWA`、`Lagrangian-GRPO`、`GAPO/PAMA`、`Dynamic Reward Weighting`、`NSGA-II-style`、`HVPO`。另开一张“策略前沿覆盖”表比较 `MORLHF`、`Rewarded Soups`、`RiC`、`Panacea/CLP`、`HaM`，再用 response-set 表比较 `VPO` 与 `HVPO`；三张表不要直接混合排名。

## 2. 当前 Amo 的比较锚点

当前实现已经形成一个很干净的受控实验接口：

- `amo_vanilla + GRPO`：先对多个原始 reward 做等权平均，再在同一 prompt 的 rollout group 内标准化。
- `amo_vanilla + GDPO`：每个 reward 在 rollout group 内分别标准化，求和后再做 batch-wise whitening。
- `amo_hvpo + HVPO`：先把 reward vector 变成 exclusive hypervolume contribution（以及 dominated point 的距离惩罚），再使用与 GRPO 同形的 group-relative advantage。

因此，最强的第一组证据应尽量保持模型、数据、rollout、reward functions、KL、训练 token 数完全相同，只替换下面三处之一：

1. **reward scalarization**：LS、Tchebycheff、约束/Lagrangian、动态权重、Pareto rank、HV contribution。
2. **advantage construction**：GRPO、GDPO、每目标独立 advantage 后组合。
3. **gradient aggregation**：MGDA、GAPO、PCGrad 类方法。

需要特别注意：当前等权 `amo_vanilla` 只是 MORLHF/线性标量化前沿上的一个点，不足以代表完整的 linear scalarization baseline。至少应在二目标任务上跑

$$
\mathbf{w}\in\{(0,1),(0.1,0.9),\ldots,(0.9,0.1),(1,0)\},
$$

在五目标 HelpSteer2 上则用固定的 simplex design（例如 simplex-lattice 或固定 Dirichlet 样本），并对所有方法复用同一组权重。

## 3. 第一组：与 HVPO 同构的在线 RL Baseline

### 3.1 Linear Scalarization / MORLHF sweep

对归一化后的 reward vector $\tilde{\mathbf r}$ 使用

$$
r_{\mathrm{LS}}(x,y;\mathbf w)=\sum_{j=1}^{m}w_j\tilde r_j(x,y),\qquad
\mathbf w\in\Delta^{m-1}.
$$

这是多目标 RLHF 最常见的基线，也是 [MODPO](https://arxiv.org/abs/2310.03708)、[RiC](https://proceedings.mlr.press/v235/yang24q.html)、[HaM](https://arxiv.org/abs/2412.05469) 和 [GAPO](https://aclanthology.org/2025.acl-long.549/) 等论文都采用或比较的基线。

**复现建议**：

- 新增可配置的 `weights`，不要只保留当前等权平均。
- 原始 reward 尺度差异大时，先用训练前冻结的 calibration split 估计每维的 affine transform；不要让每个方法用自己的 test statistics。
- 二目标跑完整 11 点权重网格；多目标使用同一批固定权重向量。
- 同时保留“先加权再 group-normalize”的 GRPO 和“先逐维 normalize 再加权”的 weighted GDPO，两者不是同一个 baseline。

**价值**：它是所有复杂方法必须击败的最低门槛，也是判断非凸 Pareto front 是否真的需要 HV/Tchebycheff 的基础。

### 3.2 Augmented Tchebycheff-GRPO

对最大化问题，可使用理想点 $\mathbf z^*$ 定义

$$
r_{\mathrm{Tche}}(x,y;\mathbf w)
=-\max_j w_j\left(z_j^*-\tilde r_j(x,y)\right)
+\rho\sum_jw_j\tilde r_j(x,y),
$$

其中很小的 $\rho>0$ 用于减少弱 Pareto 解和并列。Tchebycheff 可以覆盖线性加权遗漏的非凸前沿；[Panacea](https://arxiv.org/abs/2402.02030) 同时研究了 LS 与 Tche 聚合。Amo 的 MOO benchmark 也已经包含 Chebyshev survival，可复用其归一化和边界处理经验。

**复现建议**：

- $\mathbf z^*$ 必须从 training/calibration 数据或单目标专家估计，不能偷看 test set。
- 同时报告 `Tche` 和 `augmented Tche`，并固定 $\rho$ 或只在 validation 上选择。
- 实现位置优先放在 reward manager；advantage 仍用 GRPO，形成“只换 scalarizer”的严格对照。

**优先级：P0。** 这是最便宜、同时最能反驳“HVPO 只是任意非线性聚合”的 baseline。

### 3.3 GDPO

[GDPO: Group reward-Decoupled Normalization Policy Optimization（ICML 2026）](https://arxiv.org/abs/2601.05242) 在每个 prompt group 内对每一维 reward 独立标准化：

$$
A_{i,j}^{(k)}=\frac{r_{i,j}^{(k)}-\mu_i^{(k)}}{\sigma_i^{(k)}+\epsilon},\qquad
A_{i,j}=\sum_k w_kA_{i,j}^{(k)},
$$

随后对整个 batch 的 $A$ 再标准化，避免 reward 维数增多时 advantage 尺度增长。论文提供 [TRL、verl 和 NeMo-RL 三套官方实现](https://github.com/NVlabs/GDPO)。

当前 Amo 的实现与论文主公式一致，已经是合格的强近邻 baseline。建议补充两项实验：

- `weighted GDPO`：在逐维标准化后扫描 $\mathbf w$，而不只做等权求和。
- `GDPO w/o batch whitening`：作为稳定性消融，不进入主排名。

### 3.4 Safe-RLHF / Lagrangian-GRPO

[Safe-RLHF（ICLR 2024）](https://arxiv.org/abs/2310.12773) 不把 helpfulness 与 harmlessness 对称求和，而是求解约束问题：

$$
\max_\theta\ \mathbb E[r_{\text{help}}]
\quad\text{s.t.}\quad
\mathbb E[c_{\text{harm}}]\le d,
$$

其 Lagrangian reward 为 $r_{\text{help}}-\lambda(c_{\text{harm}}-d)$，训练中动态更新 $\lambda\ge0$。官方实现为 [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)。

**为什么重要**：PKU-SafeRLHF 的实际部署语义往往是“在安全阈值内最大化有用性”，而不是两个目标完全对称。若 HVPO 只在平均 trade-off 上更好、却不能满足指定安全阈值，结论会偏弱。

**Amo 复现路径**：

- 先实现 `Lagrangian-GRPO`，复用 GRPO rollout 与 actor loss，只在 reward manager/trainer 中维护标量 $\lambda$。
- 在 validation 上给出多个预先声明的 cost budget $d$，形成约束前沿。
- `lambda_lr`、初值、上界和更新窗口应作为公开超参数；不要按 test harmlessness 调参。
- 对非安全任务，可以把“长度、格式、事实性下限”等当约束，验证方法是否一般化。

**优先级：P0（安全任务）/ P1（其他任务）。**

### 3.5 Fair and Stable Reward Composition

[Optimizing Language Models with Fair and Stable Reward Composition in Reinforcement Learning（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.565/) 把总 reward 写成动态加权和，交替更新 policy 与 reward weights。其核心动机是抑制高尺度或容易优化的 reward 长期支配训练，并用基于 mirror descent 的无梯度估计更新权重。GAPO 论文把该方法简称为 `Fast RL` 并作为强 baseline。

**与 HVPO 的关系**：两者都不接受固定的目标权衡；该方法在“目标权重”层自适应，HVPO 在“样本 credit”层用几何贡献自适应。因此它能直接检验 HVPO 的优势究竟来自动态平衡，还是来自 Pareto/HV 几何本身。

**Amo 复现路径**：

- 保留每维原始 reward 和当前权重状态；每隔固定 update window 更新一次权重。
- 先严格复现论文的 mirror-descent estimator，再做一个简单的 inverse-progress / inverse-scale 对照，后者只作为消融。
- 主实验固定初始权重、更新频率、权重下界和随机种子。
- 截至调研日期，未找到可确认的作者官方仓库；需要按论文复现。

**优先级：P0。**

### 3.6 MGDA、PAMA 与 GAPO

经典 [MGDA](https://doi.org/10.1016/j.crma.2012.03.014) 对每个目标的 policy gradient $g_j$ 求解

$$
\min_{\boldsymbol\alpha\in\Delta^{m-1}}
\left\|\sum_j\alpha_jg_j\right\|_2^2,
$$

得到尽可能同时改善所有目标的共同方向。[PAMA（ECML PKDD 2025）](https://arxiv.org/abs/2508.07768) 为避免对数十亿参数保存每目标完整梯度，在 Noon PPO 中把 min-norm 问题近似为对每目标 clipped advantage 的一维凸组合，并给出闭式解，把声称的复杂度从 $O(m^2d)$ 降到 $O(m)$。这里的代价是它同时把 PPO advantage 截断为 $A=\max(A',0)$，所以原版 PAMA 并非“只换聚合器”的 GRPO 对照。

[GAPO（ACL 2025）](https://aclanthology.org/2025.acl-long.549/) 则先按梯度范数重缩放

$$
\bar g_j=\frac{g_j}{\|g_j\|_2^p},
$$

再在 $\bar g_j$ 上求 MGDA 系数；P-GAPO 则用用户偏好对单位范数梯度加权。论文基于 PPO，在 PKU-SafeRLHF 上比较了 PPO-H/PPO-S、Safe-RLHF、Fair-and-Stable composition、MGDA、MORLHF 和 Rewarded Soups。

**Amo 复现路径**：

- 每个 objective 独立构造 GRPO/PPO surrogate loss，得到 $g_j$；不能只在 reward manager 中提前把 reward 合成标量。
- 先实现 `MGDA-last-layer` 与 `GAPO-last-layer`：像论文一样只用最后一层梯度估计 $\alpha$，随后用该 $\alpha$ 加权完整 per-objective losses 做一次反向传播。
- PAMA 先跑论文原版 `Noon-PPO`；若再做 `PAMA-GRPO`，必须标注为适配版本。CTWA 官方仓库已提供 verl 版 PAMA，可作为移植起点。
- 完整参数梯度需要约 $m$ 倍显存/反向开销，不适合作为首版。
- 运行 $p\in\{1,2\}$；P-GAPO 复用与 LS sweep 相同的偏好向量。
- 截至调研日期，论文未链接可确认的官方代码。

**优先级：P0。** `MGDA` 是通用梯度基线，`GAPO` 是面向 LLM 的强方法，两者应同时保留以隔离 gradient normalization 的作用。

### 3.7 Dynamic Reward Weighting

[Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting（TACL 2026）](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.696/137195) 提出两种与当前项目非常接近的方法，且[官方仓库](https://github.com/yining610/dynamic-reward-weighting)基于 verl，支持 GRPO、REINFORCE 和 RLOO。

**Hypervolume-guided weight adaptation** 维护 checkpoint 在 validation objectives 上的 Pareto buffer，用

$$
r_{\text{pareto}}(\mathbf r,\mathcal B)
=0.5+1.5\tanh\left(\Delta HV(\mathbf r,\mathcal B)\right)
$$

放大或减弱给定权重下的训练 reward。它与 HVPO 的关键区别是：前者在 **checkpoint/validation 级** 用 HV 形成 meta-reward，后者在 **response/rollout 级** 用 exclusive HV 分配 credit。

**Gradient-based weight optimization** 使用每目标梯度的影响量

$$
I_j(t)=\left\langle g_j(t),\sum_kg_k(t)\right\rangle,
$$

通过指数 mirror update 动态调整权重：

$$
\tilde w_j(t)=w_j(t-1)\exp\left(\eta_t I_j(t)/\mu\right),\qquad
\mathbf w(t)=\frac{\tilde{\mathbf w}(t)}{\sum_k\tilde w_k(t)}.
$$

**复现注意**：

- 第一种方法每步或每若干步需要 validation evaluation，必须把额外 RM calls 和生成 token 计入预算。
- 第二种方法原文为了梯度线性假设基本关闭 clipping，并只使用中间层梯度；直接照搬到标准 GRPO 可能改变训练稳定性。
- 论文主实验使用 accuracy、conciseness、clarity 三个可验证 reward，Qwen3-8B/DeepSeek-7B，8 张 H200；Amo 可先在现有 1.5B/3B 模型和 MATH/RLLA 上做缩小版。

**优先级：P0，且是除 HaM 外最重要的直接相关工作。**

### 3.8 RVPO：Reward-Variance Policy Optimization

[RVPO: Risk-Sensitive Alignment via Variance Regularization（2026 预印本）](https://arxiv.org/abs/2605.05750) 是 GDPO 最直接、成本最低的后续基线。它先像 GDPO 一样得到每个 rollout、每个目标的 group-wise Z-score $Z_j(g)$，再用 soft-min 代替算术平均：

$$
A_{\mathrm{RVPO}}(g)
=-\frac{1}{k}\log\left(\frac{1}{m}\sum_{j=1}^{m}e^{-kZ_j(g)}\right).
$$

当 $k\to0$ 时退化为 GDPO 的均值聚合；当 $k\to\infty$ 时趋向最差目标。二阶展开给出

$$
A_{\mathrm{RVPO}}\approx \mu_Z-\frac{k}{2}\sigma_Z^2,
$$

因此它可理解为惩罚同一 response 在目标间表现不均衡。论文在 RLLA-4k 上沿用 GDPO 的 Qwen2.5-1.5B/3B + verl 配方，也研究 5--17 个 rubric rewards；这与 Amo 的 tool 和 HelpSteer 多目标设置都很贴近。

**Amo 复现路径**：只需在 GDPO 的“逐维标准化之后、batch whitening 之前”增加 soft-min 聚合。主实验至少包括 `k=0`（严格等于 GDPO）、固定小 $k$、固定中等 $k$ 和从小到大的 annealing schedule。

**重要边界**：RVPO 优先拉升当前最差目标，适合安全/格式等 bottleneck constraints，但可能放大噪声 reward model，并不等价于扩大整条 Pareto front。论文和 Apple 页面截至调研日期未链接公开代码。

**优先级：P0。** 它几乎不增加训练开销，却能强力检验 HVPO 的收益是否仅来自“非平均、偏最差目标”的聚合。

### 3.9 CTWA：Covariance Targeted Weight Adaptation

[Uncovering Cross-Objective Interference in Multi-Objective Alignment（2026 预印本）](https://arxiv.org/abs/2602.06869) 指出：优化 scalar score $s$ 时，第 $j$ 个目标的一阶改善由 on-policy covariance 决定，

$$
\Delta\mathbb E[r_j]
=\eta\,\mathbb E_x\!\left[\operatorname{Cov}_{y\sim\pi(\cdot\mid x)}(r_j(x,y),s(x,y))\right]
+O(\eta^2).
$$

CTWA 对每个 prompt 估计目标 reward 与 clipped advantage weight 的 covariance，维护 EMA；若某目标低于预注册 target $c_j^*$，就在 log-space 增大其 scalarization weight：

$$
\bar c_j\leftarrow(1-\tau)\bar c_j+\tau c_j,\quad
\delta_j=[c_j^*-\bar c_j]_+,\quad
u_j\leftarrow u_j+\eta_\lambda\delta_j,\quad \lambda_j=e^{u_j}.
$$

[官方 verl 仓库](https://github.com/yining610/ctwa)不仅实现 CTWA，还包含 Linear、Dynamic Reweighting、GradNorm、Lagrangian、MGDA、PAMA、Tchebycheff、Smooth Tchebycheff、Nash-MTL 和 FAMO。这是 Amo 扩充 baseline 最有价值的代码来源之一。

**Amo 复现路径**：复用当前每维 reward tensors，在同 UID 的 rollouts 内计算 covariance；weight update 不需要逐目标 backward，额外成本很低。covariance targets 必须仅在 validation 上选择，并报告目标值敏感性。

**优先级：P0。** 它直接检验 HVPO 是否比“保证每维训练信号方向正确”的自适应线性标量化更有效。

### 3.10 NSGA-II / NSGA-III / SMS-EMOA-style response credit

这些不是现成的 LLM alignment 论文算法，但已在 Amo 的 MOO benchmark 中使用，适合做“相同 reward vector、相同 rollout，只替换 survival/credit rule”的机制消融。

**可落地版本**：

- `Pareto-rank GRPO`：对每个 UID 的 $G$ 个 responses 做 non-dominated sorting，以负 rank 作为 scalar reward。
- `NSGA-II credit`：同 rank 内加入 crowding distance，鼓励前沿覆盖。
- `NSGA-III credit`：用固定 reference directions 做 niche assignment；更适合 HelpSteer2 这类 $m>3$，但 $G=4$ 太小，应提高 rollout 数或使用 global archive。
- `SMS-EMOA credit`：给非支配 responses 使用 exclusive HV contribution，删除/惩罚贡献最小者；它和 HVPO 很接近，最适合作为 HVPO 的组件消融。
- `MOEA/D-style`：每个 preference direction 对应一个 LoRA/policy 或 rollout subgroup，使用 Tchebycheff subproblem；成本较高，放到 P2。

**解释边界**：把 NSGA-II 排序变成 policy-gradient advantage 是一种合理适配，不应写成“复现 NSGA-II 训练 LLM”。论文中应称 `NSGA-II-style response selection/credit`。

## 4. 第二组：直接相关但训练范式不同的方法

### 4.1 HaM：Hypervolume maximization Method

[Multi-Objective Alignment of Large Language Models Through Hypervolume Maximization（2024）](https://arxiv.org/abs/2412.05469) 是概念上与 HVPO 最直接的工作。HaM 联合训练 $K$ 个 policy heads，最大化它们在 $m$ 个目标上的总体 hypervolume。每个目标不是在线 rollout reward，而是 reward-weighted log-likelihood：

$$
L_j(\theta_k)=\mathbb E_{(x,y)\sim\mathcal D}
\left[r_j(x,y)\log p(y\mid x;\theta_k)\right].
$$

将归一化后的 $\bar L_j(\theta_k)$ 看作第 $k$ 个 policy 的目标向量，再最大化 $K$ 个超矩形并集的 HV。论文用共享 transformer backbone 加 $K$ 个输出 heads 降低空间开销，默认 $K=5$；实验覆盖 helpfulness、harmlessness、humor、faithfulness、hallucination，并比较 SCA 与 RiC。

**HaM 与 HVPO 不能混同**：

| 维度 | HVPO | HaM |
|---|---|---|
| HV 作用对象 | 同 prompt 的 sampled responses / Pareto cache | 多个 policy heads 的数据集级目标值 |
| 训练信号 | 在线 reward vector -> scalar advantage | 离线 reward-weighted log-likelihood |
| 输出 | 通常一个 policy | $K$ 个共享 backbone 的 policies |
| 参考点 | reward 空间中的显式 reference point | 归一化 $[0,1]^m$ 中的原点 |
| 主要目标 | rollout credit assignment | policy-level Pareto front coverage |

**复现建议**：先按论文做 `offline HaM-LoRA`，使用当前已打分的数据和共享 backbone；不要一开始尝试把它硬改成 GRPO。公平比较应放在“policy-level front”轨道，给每种方法相同的 operating points 数量 $K$ 和总训练预算。论文页面未提供可确认的官方代码，需要自行实现。

### 4.2 MORLHF 与 Fine-Grained RLHF

`MORLHF` 通常指“每个维度独立 reward model + 线性组合 + PPO/RLHF”，而不是唯一一篇论文。一个经典、可复现的具体实例是 [Fine-Grained RLHF（NeurIPS 2023）](https://finegrainedrlhf.github.io/)，它同时研究 reward 的细粒度位置和多个 feedback types，在长文本问答中联合 factuality、relevance、completeness 等 reward，并公开数据和代码。

**建议用途**：

- 把当前 `GRPO-LS sweep` 作为 MORLHF 的 Amo 原生版本。
- 若需要对照传统 PPO-RLHF，再复现 Fine-Grained RLHF 的多 reward PPO；这会同时改变 advantage estimator、critic 和 reward density，放在扩展表而非最严格主表。
- `P-MORL` 则给 policy 输入 preference weights，用同一个 RL policy 覆盖多种权衡；可作为 preference-conditioned RL 基线。

### 4.3 Rewarded Soups / Personalized Soups

[Rewarded Soups（NeurIPS 2023）](https://arxiv.org/abs/2306.04488) 从同一初始模型分别训练 $m$ 个单目标专家 $\theta_j$，推理前按偏好做参数插值

$$
\theta(\mathbf w)=\sum_jw_j\theta_j.
$$

官方代码为 [alexrame/rewardedsoups](https://github.com/alexrame/rewardedsoups)。这是非常常用的 policy merging baseline，优点是实现简单、只训练 $m$ 个专家；缺点是参数空间线性连接不保证 reward 空间 Pareto 最优。

**Amo 复现路径**：

- 用现有训练脚本分别跑每个单目标 LoRA expert。
- 只合并 adapter 参数，所有专家必须来自同一 base checkpoint、同一 LoRA target/rank。
- 在固定 $\mathbf w$ 网格上合并并评测；总训练 token 预算按所有专家之和计。
- 可额外比较 `model soup` 与 `logit mixture`，但后者推理成本不同，应单独标注。

### 4.4 Rewards-in-Context（RiC）

[RiC（ICML 2024）](https://proceedings.mlr.press/v235/yang24q.html) 把每个 response 的多维 reward 数值写入 prompt，通过 SFT 学习

$$
p_\theta\left(y\mid x, r_1(x,y),\ldots,r_m(x,y)\right),
$$

推理时再把用户偏好映射为目标 reward values。它只训练一个模型，支持 inference-time 动态调整；论文报告约为 MORLHF 10% 的 GPU hours，并在原实验中优于 Rewarded Soups。

**Amo 复现要求**：需要一个覆盖足够 reward 区域的离线 response pool、冻结 reward 打分、reward token/数值编码和 preference-to-target-reward 映射。若当前训练数据每个 prompt 只有一个窄分布 response，RiC 会因覆盖不足吃亏，应先用同一个 base model 生成候选池。未找到可确认的官方训练仓库。

### 4.5 Panacea 与 Conditional Language Policy（CLP）

[Panacea（NeurIPS 2024）](https://arxiv.org/abs/2402.02030) 把 preference vector 注入 SVD-LoRA 的 singular values，每个 batch 随机采样 $\mathbf w$，支持 RLHF 或 DPO、LS 或 Tchebycheff 聚合。训练后一个模型即可按 $\mathbf w$ 改变行为。论文基于 Safe-RLHF，在 BeaverTails helpful/harmless 上与 Rewarded Soups 和每权重点单训模型（DPS）比较。

[CLP（Findings of EMNLP 2024）](https://aclanthology.org/2024.findings-emnlp.118/) 是更一般的 multi-task + parameter-efficient conditional policy 框架，可组合 prompt conditioning 与 parameter-space conditioning，在摘要和长文任务上覆盖 reward trade-off。

**复现建议**：二者回答的是“一个 steerable model 能否表示整个 Pareto set”，而不是“无偏好输入时哪个 scalar advantage 更好”。应在固定 preference grid 上比较：

- Pareto HV / empirical IGD+；
- preference controllability（目标权重变化与各维 reward 的单调性、rank correlation）；
- 单模型参数量、训练总 token、切换偏好的推理开销。

两篇论文均未在论文页面链接可确认的官方代码；Panacea 可按论文在 Safe-RLHF 的 LoRA 层上实现。

### 4.6 MODPO、CPO 与 MO-ODPO

**MODPO**：[Beyond One-Preference-Fits-All Alignment（Findings of ACL 2024）](https://arxiv.org/abs/2310.03708) 是 RL-free 的 multi-objective DPO。它为每个固定 preference vector 训练一个 policy，理论上对应相同权重下的 MORLHF 解。官方代码：[ZHZisZZ/modpo](https://github.com/ZHZisZZ/modpo)。

**CPO**：[Controllable Preference Optimization（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.85/) 把各目标的显式 preference scores/requirements 条件化到模型中，用单一可控 policy 覆盖 3H（helpfulness、honesty、harmlessness）。官方代码：[OpenBMB/CPO](https://github.com/OpenBMB/CPO)。它要求训练样本具有每个目标的标签/score。

**MO-ODPO**：[Robust Multi-Objective Preference Alignment with Online DPO（2025）](https://arxiv.org/abs/2503.00295) 每步从 Dirichlet 采样 $\mathbf w$，把权重前缀写入 prompt，在线采样两个 responses，用 $\sum_jw_jr_j$ 选 chosen/rejected，再做标准 DPO loss。它只训练一个可控 policy，不要求预先为同一 response 收集所有目标的人工偏好标签；论文在 Anthropic-HH 与 TL;DR 上比较 Rewarded Soups、ODPO Soups、P-MORL、RiC、MORLHF 和 MODPO。

**与 HVPO 比较时的边界**：

- 这些方法使用 pairwise preference loss，HVPO 使用 on-policy scalar advantage，训练效率和稳定性可以比较，但不能声称“只换多目标聚合规则”。
- 最公平的 MO-ODPO 版本可复用当前 reward functions 在线构造 pairs，并匹配生成 token/RM calls 预算。
- MODPO/CPO 需要按 objective 构造 preference pairs；不能从仅有 scalar online reward 的实验无损得到。

### 4.7 VPO：Vector Policy Optimization

[Vector Policy Optimization（2026 预印本）](https://arxiv.org/abs/2605.22817) 是 response-set 视角下与 HVPO 很相关的新方法。它让一次 autoregressive rollout 输出 $q$ 个候选答案组成集合 $S$，从 Dirichlet 分布采样 $K$ 个 reward weights，并用

$$
\hat R(S)=\frac{1}{K}\sum_{k=1}^{K}\max_{y\in S}\mathbf w_k^\top\mathbf r(x,y)
$$

作为整个集合的 GRPO reward。不同候选因而会专门覆盖不同 reward trade-offs。论文在 MuSiQue、EUREQA、Maze、ToolRL 和 LiveCodeBench 上比较 GRPO、GDPO、random-weight GRPO、goal-conditioned GRPO、Multi-RLVR 及 best@$k$ 方法。

**与 HVPO 的区别**：VPO 用随机线性标量化的期望 best-of-set 奖励整个多答案序列；HVPO 对独立 sampled responses 计算几何 exclusive-HV credit。VPO 主要优化 test-time `best@k` 和 reward-space diversity，并明确可能牺牲 pass@1。因此它不应进入只比较单回答均值的主表。

**复现建议**：先实现 `Random-weight GRPO` 作为便宜消融，再做 $q=3$ 的 multi-answer chain；对齐 evaluator calls、生成 token 和上下文复用收益。主指标应包含 pass@1、best@$k$、response-set HV、非支配点数与 reward-space pairwise distance。论文截至调研日期未链接作者代码。

**优先级：P1。** 若 HVPO 的主张包括“同 prompt 下维持多样且 Pareto 有效的回答集合”，则提升为必跑。

## 5. 第三组：可作为扩展实验的方法

| 方法 | 核心思想 | 适合回答的问题 | 来源/代码 |
|---|---|---|---|
| DPA | 多目标 reward model + preference-conditioned rejection sampling；支持模型权重算术 | 不做 RL 能否获得方向可控性？ | [ACL 2024](https://aclanthology.org/2024.acl-long.468/)，[代码](https://github.com/RLHFlow/directional-preference-alignment) |
| MetaAligner | 训练一个 policy-agnostic response rewriter，根据目标偏好改写任意模型输出 | 后处理是否已足够？ | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d03800841fa1bb2f43ef1750aafcce4-Abstract-Conference.html)，[代码](https://github.com/SteveKGYang/MetaAligner) |
| HoE | 复用单目标 LoRA experts，通过 hierarchical routing 和 preference routing 组合 | 无额外 policy 训练的 expert 组合能否覆盖前沿？ | [arXiv 2025](https://arxiv.org/abs/2505.20925) |
| Projection Optimization | 把 nonlinear multi-objective aggregation 分解成一系列线性子问题，并可扩展到 multi-group | max-min、Nash welfare 等非线性社会福利目标 | [ICML 2025](https://arxiv.org/abs/2502.15145) |
| MOD / DeAL | 在 decoding time 组合多个单目标 policy/reward，不再训练 policy | 训练时优化是否必要？ | [MOD](https://arxiv.org/abs/2406.18853)，[DeAL](https://arxiv.org/abs/2402.06147) |
| NSGA-III / MOEA/D policy population | reference directions 或 decomposition 对多个 policy/LoRA 分工 | many-objective 下显式 policy population 是否更好？ | [pymoo](https://pymoo.org/)；Amo MOO harness 已有参考实现 |

这些方法不建议进入首轮主表：它们改变了模型结构、推理流程、训练数据或输出形态。但在完整论文中至少应在 related work 和扩展实验中覆盖 DPA、MetaAligner、Projection Optimization 与 decoding-time methods。

## 6. 建议的三条评测轨道

### Track A：单 policy、同在线 RL 链路

目标是判断 HVPO 的 **credit assignment** 是否有效。所有方法使用同一 base model、prompt batch、rollout 数、reward functions、KL、optimizer、训练 token 与 seeds。

建议方法：

1. Equal-weight GRPO。
2. LS-GRPO weight sweep。
3. Augmented-Tchebycheff-GRPO。
4. GDPO / weighted GDPO。
5. RVPO。
6. CTWA。
7. Lagrangian-GRPO。
8. Fair-and-Stable dynamic composition。
9. MGDA/PAMA/GAPO。
10. NSGA-II-style credit。
11. SMS-EMOA-style credit。
12. HVPO。

每次 run 的最终模型是一个 operating point。主要报告各原始目标、约束满足率、训练稳定性，以及从同一 prompt 采样 $G$ 个 responses 得到的 **response-set HV**。

### Track B：policy-level Pareto front / inference-time steerability

目标是判断方法能否覆盖不同用户偏好。给所有方法相同的 $K$ 个 evaluation preferences 或相同数量的 policy heads/experts。

建议方法：

1. MORLHF/LS：每个 $\mathbf w$ 单独训练。
2. P-GAPO：每个 $\mathbf w$ 单独训练。
3. Rewarded Soups：训练 $m$ 个单目标 experts 后插值。
4. RiC：单 reward-conditioned policy。
5. Panacea 或 CLP：单 preference-conditioned policy。
6. HaM：$K$ 个共享 backbone policy heads。
7. MODPO：每个 $\mathbf w$ 单独训练。
8. CPO 或 MO-ODPO：单 preference-conditioned policy。

主要报告 **policy-front HV、empirical IGD+、epsilon indicator、front spacing、controllability、总训练成本、部署参数量和每次切换偏好的推理成本**。

### Track C：同 prompt 的 response-set / test-time search

目标是判断一个模型能否生成多样且 Pareto 有效的候选集合。建议比较 `GRPO`、`GDPO`、`Random-weight GRPO`、`NSGA-II/SMS-EMOA-style credit`、`HVPO`、`VPO`，并固定每个 prompt 的总生成 token 或候选数。

主要报告 **pass@1、best@$k$、response-set HV、非支配候选数、reward-space diversity 与随 $k$ 增长的收益曲线**。单回答部署场景以 pass@1 为主；带 reranker/verifier 的场景以 best@$k$ 为主。

**不要做的比较**：

- 不要把 HaM 的 $K$ 个 heads 形成的前沿，与 HVPO 单个 final checkpoint 的一个均值点直接比较。
- 不要把一个方法沿训练过程挑出的所有 checkpoints 当作它的 policy front，而只给其他方法 final checkpoints；这会引入选择偏差。
- 不要把同一模型的多个随机采样 response points 与多个独立 policy 的 dataset-average reward points混成同一种 HV。
- 不要用 VPO 的 best@$k$ 对比其他方法的 pass@1；所有方法都必须接受同一个 test-time selection budget。

## 7. 公平实验协议

### 7.1 Reward 方向与归一化

所有目标统一为“越大越好”。cost、长度、hallucination rate 等先变号或转为 satisfaction score。归一化常数只从训练前固定的 calibration split 得到，并对全部方法冻结。

推荐保存两套值：

- `raw_reward_j`：用于解释真实任务含义。
- `normalized_reward_j`：只用于 Tchebycheff、HV、IGD+ 等几何运算。

如果没有真实 objective bounds，可使用单目标专家与 base model 在 calibration split 上的稳健分位数构造 bounds，并报告超界 clipping 比例。

### 7.2 Hypervolume reference point

LLM 任务没有已知 true Pareto front，reference point 不能按每个方法或每个 seed 动态选择。建议：

1. 在 calibration split 上预注册固定 reference point。
2. 所有方法、所有 seeds 共用。
3. 主结果至少做一组 reference-point sensitivity analysis。
4. 明确区分 response-level HV 与 policy-level HV。

### 7.3 IGD 的处理

MOO benchmark 有真实 Pareto front，LLM alignment 没有。因此不能把 LLM 的 IGD 称为“到 true front 的 IGD”。可使用所有方法、所有 seeds 的 test points 并集构建 empirical non-dominated reference set，报告 `empirical IGD+`；同时保留 coverage 和 epsilon indicator，避免 reference set 偏向采样更多的方法。

### 7.4 预算匹配

至少同时报告：

- policy optimizer steps；
- generated tokens；
- reward model calls；
- backward passes（多梯度方法尤其重要）；
- GPU hours 与峰值显存；
- 训练/保存的独立 policy 或 adapter 数量。

对 Rewarded Soups、MORLHF、MODPO 等方法，总预算必须累加所有专家/权重点。对 Dynamic Reward Weighting，validation generation 和 RM calls 也必须计入。

### 7.5 统计与选择

- 首轮 smoke test 可用 1 seed；进入主表至少 3 seeds，关键结论建议 5 seeds。
- 超参数只在 validation 上选，并给所有方法相当的搜索预算。
- 报告 mean、standard error/置信区间、Pareto dominance frequency；不要只数 `# best`。
- 除训练 reward model 外，加入独立 held-out judge、规则 evaluator 或人工抽样，检查 reward hacking。

## 8. 面向 Amo 的实现顺序

### 阶段一：低成本机制消融

1. 把 `amo_vanilla` 的等权重改为配置化权重，完成 LS sweep。
2. 新增 Tchebycheff reward manager，共享固定 normalization/ideal point。
3. 为现有 GDPO 增加 objective weights。
4. 新增 Pareto-rank、NSGA-II crowding 和 SMS-EMOA-style reward manager。
5. 用同一小模型、同一 seed、同一短训练先检查每维 reward 与 response-set HV。

### 阶段二：在线 RL 强 baseline

1. Lagrangian-GRPO。
2. RVPO（直接复用 GDPO tensors）。
3. 从官方 verl 代码移植 CTWA 及其 Smooth Tchebycheff/PAMA baselines。
4. Fair-and-Stable dynamic reward composition。
5. MGDA-last-layer 与 GAPO-last-layer。
6. 从官方 verl 代码移植 Dynamic Reward Weighting 的两种版本。

### 阶段三：前沿覆盖实验

1. 训练单目标 LoRA experts，完成 Rewarded Soups。
2. 选择 Panacea 或 CLP 中一个 preference-conditioned policy；Panacea 与现有 LoRA 栈更贴近。
3. 实现 offline HaM 多头版本，作为超体积方法的直接概念对手。
4. 若有多维 preference pairs，再加入 MODPO/CPO；否则优先 MO-ODPO，因为它可用现有 online reward functions 构造 pairs。
5. 若目标包含候选集合或 test-time search，加入 VPO 的 multi-answer rollout。

## 9. 推荐最终实验矩阵

| 层级 | 必跑 | 可选 | 目标 |
|---|---|---|---|
| 最小可发表对照 | LS sweep、Tche、GDPO、RVPO、CTWA、Lagrangian、GAPO/PAMA、Dynamic Weighting、HVPO | NSGA-II-style、SMS-EMOA-style | 证明在线多目标 credit 的有效性 |
| 超体积专项 | SMS-EMOA-style、Dynamic HV-guided、HaM、HVPO | random-HV scalarization | 区分不同 HV 使用层级 |
| 完整 Pareto 对照 | MORLHF、Rewarded Soups、Panacea/CLP、HaM、HVPO 的等预算变体 | RiC、MODPO、CPO、MO-ODPO | 比较前沿覆盖与可控性 |
| 安全专项 | Safe-RLHF、LS/GDPO、GAPO、HVPO | BFPO/CPO | helpfulness-harmlessness 与约束满足 |
| many-objective | LS design、weighted GDPO、NSGA-III-style、GAPO、HVPO | MOEA/D policies、HoE | HelpSteer2 5 维扩展性 |
| response-set / search | GRPO、GDPO、Random-weight GRPO、HVPO、VPO | NSGA-II/SMS-EMOA-style | 比较集合覆盖、best@$k$ 与多样性 |

最关键的论文叙事应当是：**HVPO 是否在相同在线 RL 预算下，比固定/动态标量化、约束法、逐维归一化、梯度冲突处理和常见 Pareto selection 都产生更好的 response-level 多目标 credit；以及这种优势能否进一步转化为 policy-level Pareto coverage。**

## 10. 一手来源索引

| 方法 | 论文 | 官方实现状态 |
|---|---|---|
| Fine-Grained RLHF | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html) | [GitHub](https://github.com/allenai/FineGrainedRLHF) |
| Rewarded Soups | [NeurIPS 2023](https://arxiv.org/abs/2306.04488) | [GitHub](https://github.com/alexrame/rewardedsoups) |
| Safe-RLHF | [ICLR 2024 / arXiv](https://arxiv.org/abs/2310.12773) | [GitHub](https://github.com/PKU-Alignment/safe-rlhf) |
| MODPO | [Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.630/) | [GitHub](https://github.com/ZHZisZZ/modpo) |
| RiC | [ICML 2024](https://proceedings.mlr.press/v235/yang24q.html) | [GitHub](https://github.com/YangRui2015/RiC) |
| Panacea | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89f39d0b3d49a47606a165eefba2778c-Abstract-Conference.html) | 未找到可确认的官方仓库 |
| CPO | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.85/) | [GitHub](https://github.com/OpenBMB/CPO) |
| CLP | [Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.118/) | 未找到可确认的官方仓库 |
| Fair/Stable Reward Composition | [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.565/) | 未找到可确认的作者仓库 |
| DPA | [ACL 2024](https://aclanthology.org/2024.acl-long.468/) | [GitHub](https://github.com/RLHFlow/directional-preference-alignment) |
| MetaAligner | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d03800841fa1bb2f43ef1750aafcce4-Abstract-Conference.html) | [GitHub](https://github.com/SteveKGYang/MetaAligner) |
| HaM | [arXiv 2024](https://arxiv.org/abs/2412.05469) | 未找到可确认的官方仓库 |
| MO-ODPO | [arXiv 2025](https://arxiv.org/abs/2503.00295) | 未找到可确认的官方仓库 |
| Projection Optimization | [ICML 2025 / arXiv](https://arxiv.org/abs/2502.15145) | 未找到可确认的官方仓库 |
| GAPO/P-GAPO | [ACL 2025](https://aclanthology.org/2025.acl-long.549/) | 未找到可确认的官方仓库 |
| PAMA | [ECML PKDD 2025](https://arxiv.org/abs/2508.07768) | 未找到作者仓库；[CTWA 仓库含 verl 复现](https://github.com/yining610/ctwa) |
| Hierarchical Experts (HoE) | [arXiv 2025](https://arxiv.org/abs/2505.20925) | 论文称代码在 supplementary materials；需进一步核对可用性 |
| Dynamic Reward Weighting | [TACL 2026](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.696/137195) | [GitHub](https://github.com/yining610/dynamic-reward-weighting) |
| GDPO | [ICML 2026 / arXiv](https://arxiv.org/abs/2601.05242) | [GitHub](https://github.com/NVlabs/GDPO) |
| CTWA | [arXiv 2026](https://arxiv.org/abs/2602.06869) | [GitHub](https://github.com/yining610/ctwa) |
| RVPO | [arXiv 2026](https://arxiv.org/abs/2605.05750) | 截至调研日期未公开代码 |
| VPO | [arXiv 2026](https://arxiv.org/abs/2605.22817) | 截至调研日期未找到作者代码 |
