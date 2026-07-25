# PKU-SafeRLHF 统一训练与评测矩阵

本指南覆盖 PKU-SafeRLHF 上的完整受控矩阵：3 个模型 × 15 个方法，共 45 个 train/eval cell。矩阵只使用 `scripts/trainers/` 下的统一入口。

## 矩阵定义

模型短名：

```text
1.5b     -> qwen2.5-1.5b
3b       -> qwen2.5-3b
llama3b  -> llama3.2-3b
```

方法：

```text
grpo gdpo hvpo tchebycheff rvpo mgda gapo
lagrangian fair_stable ctwa dynamic_hv nsga2 smsemoa
```

每个 cell 的默认身份为 `<model_tag>_<method>`，例如 `qwen2.5-1.5b_grpo`。

## 运行前准备

数据必须位于：

```text
data/PKU-SafeRLHF/train.parquet
data/PKU-SafeRLHF/test.parquet
```

默认模型路径为：

```text
/data/Qwen/Qwen2.5-1.5B-Instruct
/data/Qwen/Qwen2.5-3B-Instruct
/data/meta-llama/Llama-3.2-3B-Instruct
```

训练时可用 `AMO_MODEL_PATH` 覆盖所选模型路径。当前 `tools/eval_safe.sh` 使用上述默认 base-model 路径，因此执行统一安全评测的机器也应提供这些路径。

完整 matrix 默认使用
`/home/rihongqiu/data/miniconda3/envs/amo/bin/python`；其他机器应通过
`AMO_PY` 指定已安装 Amo/verl 依赖的 Python。

PKU-SafeRLHF 的两个 reward functions 通过 gRPC reward servers 评分。先启动服务：

```bash
bash scripts/amo_exp/serve_rewards.sh safe
```

默认连接为 helpful `localhost:50051`、harmless `localhost:50052`。远程服务可通过以下变量配置：

```bash
export HELPFUL_TARGET_HOST=localhost
export HELPFUL_TARGET_PORT=50051
export HARMLESS_TARGET_HOST=localhost
export HARMLESS_TARGET_PORT=50052
```

停止本地服务：

```bash
bash scripts/amo_exp/serve_rewards.sh stop
```

## 生成并冻结 calibration

安全奖励是无界 reward-model logits。`grpo`、`tchebycheff`、`lagrangian` 与 `dynamic_hv` 必须共享同一份冻结标定：

```bash
${AMO_PY:-python3} scripts/trainers/tools/calibrate_safe.py --n 512 --seed 0
```

reward servers 必须已启动。命令写入：

```text
results/PKU-SafeRLHF/safe_calibration.json
```

该文件包含 affine bounds、Tchebycheff ideal、Dynamic-HV reference 与 Lagrangian harmlessness budget。一次实验矩阵只生成一次；跨机器执行时应复制同一文件，不要按模型、方法或机器重新估计。

可选参数包括 `--data`、`--out`、`--n`、`--seed`、`--workers`、`--lower_pct` 与 `--upper_pct`。

## 先 dry-run 一个 cell

```bash
bash scripts/trainers/grpo/run_pku-saferlhf.sh 1.5b 1 --dry-run
```

这会打印最终训练命令但不启动训练。上述四个 scale-sensitive 方法在 dry run 时仍需读取 `safe_calibration.json`。

## 手动运行一个 cell

训练：

```bash
bash scripts/trainers/<method>/run_pku-saferlhf.sh [MODEL] [EPOCH] [--dry-run] [Hydra overrides...]
```

例如：

```bash
bash scripts/trainers/rvpo/run_pku-saferlhf.sh 1.5b 1
```

默认 checkpoint 为：

```text
checkpoints/amo_pku-saferlhf/qwen2.5-1.5b_rvpo/
```

训练默认 `resume_mode=auto`。常用资源覆盖示例：

```bash
TRAIN_GPUS=0,1 GPU_MEM_UTIL=0.4 MICRO_BATCH_SIZE_PER_GPU=8 \
  bash scripts/trainers/rvpo/run_pku-saferlhf.sh 1.5b 1
```

评测 latest checkpoint：

```bash
bash scripts/trainers/tools/eval_safe.sh qwen2.5-1.5b_rvpo 1.5b
```

第二个参数可省略；当实验名以 `qwen2.5-1.5b_`、`qwen2.5-3b_` 或 `llama3.2-3b_` 开头时，脚本会自动推断模型。评测依次执行：

1. 将 latest step 的 LoRA adapter 或 FSDP actor 合并为完整模型；已有
   `merge/config.json` 时复用。
2. 对完整 `data/PKU-SafeRLHF/test.parquet` 生成回答。
3. 使用两个安全 reward functions 评分并计算 hypervolume。

输出为：

```text
results/PKU-SafeRLHF/qwen2.5-1.5b_rvpo.parquet
results/PKU-SafeRLHF/qwen2.5-1.5b_rvpo.json
```

评测默认使用 `EVAL_GPUS=0,1`、`GPU_MEM_UTIL=0.35`。可用 `EVAL_GPUS`、`GPU_MEM_UTIL`、`EVAL_DATA` 与 `AMO_PY` 覆盖。

## 运行完整矩阵

```bash
bash scripts/trainers/orchestration/run_safe_matrix.sh [MODELS] [METHODS] [EPOCH]
```

- `MODELS`：空格或逗号分隔的短名，默认 `1.5b 3b llama3b`。matrix 应使用这三个短名。
- `METHODS`：空格或逗号分隔，默认全部 15 个方法。
- `EPOCH`：每个 cell 的 total epochs，默认 `1`。

完整 45-cell 串行执行：

```bash
nohup bash scripts/trainers/orchestration/run_safe_matrix.sh \
  > train_logs/safe_baselines/matrix.log 2>&1 &
```

仅运行一部分：

```bash
# 一个模型、全部方法
bash scripts/trainers/orchestration/run_safe_matrix.sh "1.5b"

# 一个模型、指定方法
bash scripts/trainers/orchestration/run_safe_matrix.sh \
  "3b" "grpo gdpo hvpo tchebycheff" 1

# 逗号分隔也可用
bash scripts/trainers/orchestration/run_safe_matrix.sh \
  "1.5b,llama3b" "rvpo,mgda,gapo" 1
```

driver 严格串行执行 train → eval，行为如下：

- 若 `results/PKU-SafeRLHF/<experiment>.json` 已存在，则跳过该 cell。
- 训练入口使用 `resume_mode=auto`，中断后再次运行可续训。
- 单个 cell 失败时记录失败并继续下一个 cell。
- 每个成功 cell 后刷新聚合表。
- 默认安全 profile 与评测都使用 GPU 0、1；不要在同一 GPU pair 上并发启动多个 matrix 或 generation。

日志位置：

```text
train_logs/safe_baselines/<experiment>.train.log
train_logs/safe_baselines/<experiment>.eval.log
train_logs/safe_baselines/matrix_progress.log
```

查看进度：

```bash
tail -f train_logs/safe_baselines/matrix_progress.log
```

## 聚合结果

matrix 会在每个完成的 cell 后自动聚合，也可随时手动执行：

```bash
${AMO_PY:-python3} scripts/trainers/tools/aggregate_safe.py
```

默认扫描 `results/PKU-SafeRLHF/*.json`，写入：

```text
results/PKU-SafeRLHF/baselines_table.md
```

表格按固定的 3 模型 × 15 方法顺序展示 `safe_helpfulness`、`safe_harmlessness`、hypervolume 与 prompt 数；缺失 cell 显示为 `-`。也可用 `--results DIR --out FILE` 聚合其他目录。

## 跨机器拆分

最简单的拆分方式是每台机器负责一个模型：

```bash
# machine A
bash scripts/trainers/orchestration/run_safe_matrix.sh "1.5b"

# machine B
bash scripts/trainers/orchestration/run_safe_matrix.sh "3b"

# machine C
bash scripts/trainers/orchestration/run_safe_matrix.sh "llama3b"
```

每台机器必须使用相同代码、数据和 `safe_calibration.json`，并能访问 reward servers。完成后收集各机器的以下文件到同一个 canonical results 目录：

```text
results/PKU-SafeRLHF/<model_tag>_<method>.parquet
results/PKU-SafeRLHF/<model_tag>_<method>.json
```

再运行 `aggregate_safe.py` 即可生成完整表格。不要让两台机器同时写同一个 experiment 的 checkpoint 或结果文件。

## 完成标准

完整矩阵应有 45 个方法结果 JSON（不计 `safe_calibration.json`），即 3 个模型 × 15 个方法；对应 checkpoint 位于：

```text
checkpoints/amo_pku-saferlhf/<model_tag>_<method>/
```

对应评测结果位于：

```text
results/PKU-SafeRLHF/<model_tag>_<method>.{parquet,json}
```
