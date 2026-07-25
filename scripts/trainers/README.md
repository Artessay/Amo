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
bash scripts/trainers/ls/run_math-lighteval.sh 3b 1 --dry-run \
  data.train_batch_size=64 actor_rollout_ref.rollout.n=2
```

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

全部 15 个方法如下。目录名同时也是默认实验名中的 method tag。

| 目录 | 方法 | `adv_estimator` | `reward_manager` |
|---|---|---|---|
| `grpo` | GRPO（等权） | `grpo` | `amo_vanilla` |
| `gdpo` | GDPO | `gdpo` | `amo_vanilla` |
| `hvpo` | HVPO | `hvpo` | `amo_hvpo` |
| `ls` | Linear Scalarization / MORLHF | `grpo` | `amo_scalarize` |
| `tchebycheff` | Augmented Tchebycheff | `grpo` | `amo_scalarize` |
| `gdpo_weighted` | Weighted GDPO | `gdpo_weighted` | `amo_vanilla` |
| `rvpo` | RVPO | `rvpo` | `amo_vanilla` |
| `mgda` | MGDA | `mgda` | `amo_vanilla` |
| `gapo` | GAPO | `gapo` | `amo_vanilla` |
| `lagrangian` | Lagrangian / Safe-RLHF | `grpo` | `amo_adaptive` |
| `fair_stable` | Fair-and-Stable | `grpo` | `amo_adaptive` |
| `ctwa` | CTWA | `grpo` | `amo_adaptive` |
| `dynamic_hv` | Dynamic-HV weighting | `grpo` | `amo_adaptive` |
| `nsga2` | NSGA-II-style credit | `grpo` | `amo_pareto` |
| `smsemoa` | SMS-EMOA-style credit | `grpo` | `amo_pareto` |

`grpo`、`gdpo`、`hvpo` 支持全部六个数据集；其余 12 个方法支持
`math-lighteval`、`pku-saferlhf`、`rlla` 与 `news`。

| dataset id / 入口后缀 | 数据目录 | 支持的方法 |
|---|---|---|
| `math-lighteval` | `data/MATH-LightEval` | 全部 15 个方法 |
| `pku-saferlhf` | `data/PKU-SafeRLHF` | 全部 15 个方法 |
| `rlla` | `data/RLLA` | 全部 15 个方法 |
| `news` | `data/CNN_DailyMail` | 全部 15 个方法 |
| `math-500` | `data/MATH-500` | GRPO、GDPO、HVPO |
| `paradetox` | `data/ParaDetox` | GRPO、GDPO、HVPO |

不存在的 `run_<dataset>.sh` 组合即尚未定义默认参数，不应直接调用内部 `launch.sh` 绕过该限制。

## Baseline 优先队列

Qwen2.5-1.5B 的强 trade-off 队列使用 method-outer、dataset-inner 顺序：

```bash
mkdir -p train_logs/priority_baselines
nohup bash scripts/trainers/orchestration/run_priority_baselines.sh \
  > train_logs/priority_baselines/nohup.log 2>&1 &
```

默认 baseline 顺序为
`ls -> tchebycheff -> gdpo_weighted -> rvpo -> mgda -> gapo -> lagrangian -> fair_stable -> ctwa -> dynamic_hv -> nsga2 -> smsemoa`；
每个方法依次跑 `math-lighteval -> news -> rlla`，全部成功后才进入下一方法。
任一 cell 失败时队列立即停止。News reward server 使用 GPU 0、1，训练使用
GPU 2、3；每个实验只保留最新 1 个 actor checkpoint，以控制磁盘占用。

进度记录在 `train_logs/priority_baselines/queue_progress.log`，各 cell 日志为
`<method>.<dataset>.train.log`。

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
- model profile：模型路径、LoRA 参数和 canonical model tag。
- dataset profile：train/validation parquet、reward functions、长度、batch、epoch、项目名和结果数据集名。
- method profile：`adv_estimator`、`reward_manager` 与方法专属参数。
- variant：只用于已声明的消融实验。

常用环境覆盖包括：

| 环境变量 | 作用 |
|---|---|
| `AMO_PY` | Python 可执行文件；训练默认 `python3` |
| `AMO_MODEL_PATH` | 覆盖当前所选模型的本地路径 |
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

普通数据集在 dry run 时只警告缺失的数据/模型路径。PKU-SafeRLHF 的 `ls`、`tchebycheff`、`lagrangian`、`dynamic_hv` 需要读取冻结的 `results/PKU-SafeRLHF/safe_calibration.json`，因此即使 dry run，该文件也必须存在。

## HVPO 消融

MATH-LightEval 提供五个独立入口，接口仍是 `[MODEL] [EPOCH] [--dry-run] [overrides...]`：

| 入口 | variant | 主要变化 |
|---|---|---|
| `hvpo/ablations/run_math-lighteval_distance.sh` | `distance` | `distance_metric=none` |
| `hvpo/ablations/run_math-lighteval_euclidean.sh` | `euclidean` | `distance_metric=euclidean` |
| `hvpo/ablations/run_math-lighteval_lag1.sh` | `lag1` | micro batch 16，`test_freq=1` |
| `hvpo/ablations/run_math-lighteval_lag3.sh` | `lag3` | micro batch 16，`test_freq=3` |
| `hvpo/ablations/run_math-lighteval_lag7.sh` | `lag7` | micro batch 16，`test_freq=7` |

例如：

```bash
bash scripts/trainers/hvpo/ablations/run_math-lighteval_lag3.sh 1.5b 50
```
