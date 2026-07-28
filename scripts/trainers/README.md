# 统一训练入口

`scripts/trainers/` 是 GRPO、GDPO、HVPO 与 12 个对比方法的统一训练入口。每个方法独占一个目录，每个受支持的数据集对应一个可独立执行的 `run_<dataset>.sh`；公共模型、数据集与启动逻辑只维护一份。

新实验应从本目录启动，不再依赖旧 trainer 目录中的入口。

## 快速开始

所有公开入口使用同一接口：

```bash
bash scripts/trainers/<method>/run_<dataset>.sh [MODEL] [EPOCH] [--dry-run] [Hydra overrides...]
```

- `MODEL` 可写为 `1.5b`、`3b`、`llama3b`，也可写 canonical tag：`qwen2.5-1.5b`、`qwen2.5-3b`、`llama3.2-3b`。省略时默认 `qwen2.5-1.5b`。
- `EPOCH` 是可选的正整数；省略时使用数据集 profile 的默认值。
- `--dry-run` 必须放在可选的 `MODEL`、`EPOCH` 之后、Hydra overrides 之前。
- 最后的 Hydra overrides 会在所有 profile 与方法参数之后追加，因此相同配置项以用户 override 为准。

例如：

```bash
# Qwen2.5-1.5B + HVPO + MATH-LightEval，使用数据集默认 epoch
bash scripts/trainers/hvpo/run_math-lighteval.sh 1.5b

# Llama-3.2-3B + GDPO + PKU-SafeRLHF，共 2 epoch
bash scripts/trainers/gdpo/run_pku-saferlhf.sh llama3b 2

# 只检查最终命令，不启动训练
bash scripts/trainers/grpo/run_math-lighteval.sh 3b 1 --dry-run \
  data.train_batch_size=64 actor_rollout_ref.rollout.n=2
```

## Learning rate 最佳实践

以下 actor learning rate 从目录重构前的 GDPO、GRPO、HVPO 启动脚本中恢复。同一
backbone 的值在三个方法及其已有数据集之间一致，因此统一入口将其作为 model profile
默认值：

| LLM backbone | MATH-500 | MATH-LightEval | NEWS | PKU-SafeRLHF | RLLA |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | `2e-4` | `2e-4` | `2e-4` | `2e-4` | `2e-4` |
| Qwen2.5-3B-Instruct | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Llama-3.2-3B-Instruct | — | `5e-6` | `5e-6` | `5e-6` | `5e-6` |

`—` 表示旧脚本没有对应的调参记录。ParaDetox 同样不在这批历史调参脚本中；统一入口
仍会沿用所选 backbone 的 model profile 默认值，但这不应视为已有最佳结果。临时实验可用
`ACTOR_LR=<value>` 或末尾的 Hydra override 覆盖。

## 目录结构

```text
scripts/trainers/
├── _common/
│   ├── base.sh                 # 全局优化与运行时默认值
│   ├── launch.sh               # 唯一公共启动实现
│   ├── models/                 # 模型路径和模型级参数
│   └── datasets/               # 数据路径、reward functions 和数据集级参数
├── grpo/ ... smsemoa/          # 15 个方法目录；method.sh + 数据集入口
├── hvpo/ablations/             # HVPO 的 MATH-LightEval 消融入口
├── orchestration/
│   ├── run_safe_matrix.sh      # PKU-SafeRLHF 串行矩阵
│   └── run_priority_baselines.sh # 强 trade-off 数据集的 baseline 优先队列
└── tools/
    ├── calibrate_safe.py       # 冻结安全奖励标定
    ├── eval_safe.sh            # 单个安全实验的合并、生成与评分
    └── aggregate_safe.py       # 汇总安全实验结果
```

## 方法与数据集

全部 13 个方法如下。目录名同时也是默认实验名中的 method tag。

| 目录 | 方法 | `adv_estimator` | `reward_manager` |
|---|---|---|---|
| `gdpo` | GDPO | `gdpo` | `amo_vanilla` |
| `hvpo` | HVPO | `hvpo` | `amo_hvpo` |
| `grpo` | GRPO（Linear Scalarization / MORLHF） | `grpo` | `amo_scalarize` |
| `tchebycheff` | Augmented Tchebycheff | `grpo` | `amo_scalarize` |
| `rvpo` | RVPO | `rvpo` | `amo_vanilla` |
| `ctwa` | CTWA | `grpo` | `amo_adaptive` |
| `lagrangian` | Lagrangian / Safe-RLHF | `grpo` | `amo_adaptive` |
| `fair_stable` | Fair-and-Stable | `grpo` | `amo_adaptive` |
| `mgda` | MGDA | `mgda` | `amo_vanilla` |
| `gapo` | GAPO | `gapo` | `amo_vanilla` |
| `dynamic_hv` | Dynamic-HV weighting | `grpo` | `amo_adaptive` |
| `nsga2` | NSGA-II-style credit | `grpo` | `amo_pareto` |
| `smsemoa` | SMS-EMOA-style credit | `grpo` | `amo_pareto` |


| dataset id / 入口后缀 | 数据目录 |
|---|---|
| `math-lighteval` | `data/MATH-LightEval` |
| `pku-saferlhf` | `data/PKU-SafeRLHF` |
| `rlla` | `data/RLLA` |
| `news` | `data/CNN_DailyMail` |
| `math-500` | `data/MATH-500` |
| `paradetox` | `data/ParaDetox` |


## Baseline 优先队列

Qwen2.5-1.5B 的优先队列使用 method-outer、dataset-inner 顺序：

```bash
mkdir -p train_logs/priority_baselines
nohup bash scripts/trainers/orchestration/run_priority_baselines.sh \
  > train_logs/priority_baselines/nohup.log 2>&1 &
```

默认 baseline 顺序为
每个方法默认依次跑 `math-lighteval -> pku-saferlhf -> rlla -> news`，一个数据集的全部 variant
完成后再进入下一个数据集，全部数据集成功后才进入下一方法。

四类任务在矩阵中的定位不同：

- MATH-LightEval 用 accuracy、conciseness、format 检验结果质量、推理效率与格式之间的
  构造型 trade-off，默认约有 700 个 batch-iterations/variant。
- PKU-SafeRLHF 的 helpfulness/harmlessness 是核心的内生冲突，默认约 144 个
  batch-iterations/variant，必须进入默认矩阵。
- RLLA 用 tool correctness/tool format 检验工具调用质量与结构约束，默认约 115 个
  batch-iterations/variant。
- NEWS 的四个质量目标大体同向，主要作为 aligned-objective control；其默认成本为
  `287113 * 15 / 512 ~= 8411` batch-iterations/variant，因此不进入默认集合。
  需要该 control 时显式设置
  `BASELINE_DATASETS="math-lighteval pku-saferlhf rlla news"`，其 mapping、权重网格和
  reward server 仍完整保留。

HelpSteer2 尚未进入统一 baseline/评测矩阵。ParaDetox 已提供全部方法的训练入口，
但尚未纳入默认优先队列。

先跑无 suffix 的 uniform centroid（保留历史 identity）。MATH-LightEval、PKU-SafeRLHF、
RLLA、NEWS 的 centroid 依次为 `[1/3,1/3,1/3]`、`[0.5,0.5]`、`[0.5,0.5]`、
`[0.25,0.25,0.25,0.25]`。随后共用 H=2 非均匀权重网格：

- MATH-LightEval：`h2w200 h2w020 h2w002 h2w110 h2w101 h2w011`
- PKU-SafeRLHF：`h2w20 h2w02`
- RLLA：`h2w20 h2w02`
- NEWS：`h2w2000 h2w0200 h2w0020 h2w0002 h2w1100 h2w1010 h2w1001 h2w0110 h2w0101 h2w0011`

`h2w` 后每位数字只能是 0、1、2，数字和必须为 2，实际权重为各位数字除以 2。
digit 顺序分别为 MATH-LightEval 的 `accuracy/conciseness/format`、PKU-SafeRLHF 的
`safe_helpfulness/safe_harmlessness`、RLLA 的 `tool_correctness/tool_format`、
NEWS 的 `coherence/fluency/relevance/consistency`。
例如 `h2w101` 生成 `[0.5,0.0,0.5]`。uniform 实验继续使用
variant suffix。checkpoint、result、完成 marker 和训练日志均使用该唯一 suffix；
base 日志仍为 `<method>.<dataset>.train.log`，sweep 日志为
`<method>.<dataset>.<variant>.train.log`。

PKU-SafeRLHF 的 scale-sensitive 方法必须复用
`results/PKU-SafeRLHF/safe_calibration.json`，同一矩阵中不得重新估计。队列在第一个
PKU cell 前检查 helpful RM 的 `50051` 和 harmless CM 的 `50052`；若本地端口未就绪，
默认分别在 GPU 0、1 启动两个 owned server，队列退出时只清理自己启动的进程。NEWS
`50053` 服务在 opt-in 后同样按需启动；本机显存允许三项服务常驻，而训练固定使用
GPU 2、3。端口、host 与 GPU 可通过 `HELPFUL_TARGET_*`、`HARMLESS_TARGET_*`、
`SAFE_SERVER_GPUS` 和 `NEWS_SERVER_GPUS` 覆盖。

> PKU 训练可以先进行，但当前 `amo_eval` 的 raw-logit、origin-reference、
> pooled-prompt HV 不可作为正式主结论。正式评测应使用同一份 frozen calibration，
> 在 policy mean/front 上计算 HV；另一种有效口径是对每个 prompt 生成 `n > 1`
> responses 后计算 response-set HV，而不是把不同 prompt 混成一张前沿。

任一 cell 失败时队列立即停止；每个实验默认保留最新 3 个 actor checkpoint，以兼顾
故障回退与磁盘占用。

进度记录在 `train_logs/priority_baselines/queue_progress.log`，各 cell 日志为
上述唯一命名的 `.train.log`。

## Profile 与覆盖顺序

启动器依次加载并组装：

```text
_common/base.sh
  -> _common/models/<model>.sh
  -> _common/datasets/<dataset>.sh
  -> <method>/method.sh
  -> 可选 variant
  -> 命令行 Hydra overrides
```

职责划分如下：

- base profile：共同的 batch、rollout、KL、保存频率、logger 等基础值。
- model profile：模型路径、actor learning rate、LoRA 参数和 canonical model tag。
- dataset profile：train/validation parquet、reward functions、长度、batch、epoch、项目名和结果数据集名。
- method profile：`adv_estimator`、`reward_manager` 与方法专属参数。
- variant：只用于已声明的消融实验。

常用环境覆盖包括：

| 环境变量 | 作用 |
|---|---|
| `AMO_PY` | Python 可执行文件；训练默认 `python3` |
| `AMO_MODEL_PATH` | 覆盖当前所选模型的本地路径 |
| `ACTOR_LR` | 覆盖当前所选模型的 actor learning rate |
| `TRAIN_GPUS` | 训练 GPU 列表；未设置时可沿用已有 `CUDA_VISIBLE_DEVICES` |
| `TRAIN_BATCH_SIZE`、`PPO_MINI_BATCH_SIZE`、`MICRO_BATCH_SIZE_PER_GPU` | batch 配置 |
| `MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH` | token 长度 |
| `NUM_NODES`、`NUM_GPUS_PER_NODE`、`TENSOR_MODEL_PARALLEL_SIZE` | 分布式与并行配置 |
| `GPU_MEM_UTIL`、`ROLLOUT_N` | vLLM 显存比例与每个 prompt 的 rollout 数 |
| `SAVE_FREQ`、`TEST_FREQ`、`VAL_BEFORE_TRAIN` | 保存与验证策略 |
| `RESUME_MODE` | 续训策略，profile 默认 `auto` |
| `TRAINER_LOGGER` | trainer logger 列表 |
| `EXPERIMENT_NAME` | 显式覆盖实验名 |
| `CHECKPOINT_DIR` | 显式覆盖 checkpoint 目录 |
| `DRY_RUN=1` | 等价于在正确位置传入 `--dry-run` |

为避免 checkpoint 和结果身份不一致，命令行会拒绝通过 Hydra override 修改 `trainer.project_name`、`trainer.experiment_name`、`trainer.default_local_dir` 和 resume identity；实验名、checkpoint 路径与续训方式分别使用上述专用环境变量控制。

## Canonical checkpoint 与 results 路径

默认实验名为：

```text
<model_tag>_<method>
```

例如 `qwen2.5-1.5b_hvpo`。HVPO 消融会追加 variant，例如 `qwen2.5-1.5b_hvpo_lag3`。

训练 checkpoint 始终写入：

```text
checkpoints/<project_name>/<experiment_name>/
```

推理/评测的 canonical 输出位置为：

```text
results/<results_dataset>/<experiment_name>.parquet
results/<results_dataset>/<experiment_name>.json
```

| dataset id | `project_name` | `results_dataset` |
|---|---|---|
| `math-lighteval` | `amo_math-lighteval` | `MATH-LightEval` |
| `math-500` | `amo_math-500` | `MATH-500` |
| `pku-saferlhf` | `amo_pku-saferlhf` | `PKU-SafeRLHF` |
| `paradetox` | `amo_paradetox` | `ParaDetox` |
| `rlla` | `amo_rlla` | `RLLA` |
| `news` | `amo_cnn_dailymail` | `CNN_DailyMail` |

训练入口负责写 checkpoint；它只打印预期的 results 路径，不会自动生成结果文件。PKU-SafeRLHF 可用 `tools/eval_safe.sh` 生成 `.parquet` 与 `.json`，并用统一 matrix 串起训练和评测。详见 [SAFE_MATRIX.md](SAFE_MATRIX.md)。

## Dry run

dry run 会完成 profile 解析、路径检查与命令组装，打印 shell-escaped 的最终训练命令后退出，不启动 `verl.trainer.main_ppo`：

```bash
bash scripts/trainers/gdpo/run_news.sh llama3b 10 --dry-run
```

普通数据集在 dry run 时只警告缺失的数据/模型路径。PKU-SafeRLHF 的 `grpo`、`hvpo`、`tchebycheff`、`lagrangian`、`dynamic_hv` 需要读取冻结的 `results/PKU-SafeRLHF/safe_calibration.json`，因此即使 dry run，该文件也必须存在。HVPO 对 MATH、News 和 RLLA 使用奖励函数已知的 `[0,1]` 自然边界；ParaDetox 的 LaBSE cosine 维度使用 `[-1,1]`。

## HVPO 消融

MATH-LightEval 提供三个评测频率入口，接口仍是 `[MODEL] [EPOCH] [--dry-run] [overrides...]`：

| 入口 | variant | 主要变化 |
|---|---|---|
| `hvpo/ablations/run_math-lighteval_lag1.sh` | `lag1` | micro batch 16，`test_freq=1` |
| `hvpo/ablations/run_math-lighteval_lag3.sh` | `lag3` | micro batch 16，`test_freq=3` |
| `hvpo/ablations/run_math-lighteval_lag7.sh` | `lag7` | micro batch 16，`test_freq=7` |

例如：

```bash
bash scripts/trainers/hvpo/ablations/run_math-lighteval_lag3.sh 1.5b 50
```
