#!/bin/bash
# 评测流水线: 对一个已训练实验做 merge -> generate -> amo_eval(HV)
# 用法: bash eval_pipeline.sh <EXPERIMENT_NAME>   (如 qwen2.5-1.5b_grpo)
# 生成用 GPU 2,3 (不干扰 GPU0,1 上的奖励服务); amo_eval 打分复用奖励服务
set -x
set -e

EXP=${1:?need EXPERIMENT_NAME e.g. qwen2.5-1.5b_grpo}
BASE_MODEL=${BASE_MODEL:-/data/Qwen/Qwen2.5-1.5B-Instruct}
PROJECT=amo_pku-saferlhf
DATASET=PKU-SafeRLHF

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
PY=/home/rihongqiu/data/miniconda3/envs/amo/bin/python
CKDIR=$WORKSPACE/checkpoints/$PROJECT/$EXP
STEP=$(cat $CKDIR/latest_checkpointed_iteration.txt)
ADAPTER=$CKDIR/global_step_$STEP/actor/lora_adapter
MERGE=$CKDIR/global_step_$STEP/actor/merge
DATA=$WORKSPACE/data/$DATASET/test_small.parquet     # 200 子集, 与训练验证一致
OUT=$WORKSPACE/results/$DATASET/${EXP}.parquet
REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"

mkdir -p $WORKSPACE/results/$DATASET

# 1) merge LoRA -> full HF model
if [ ! -d "$MERGE" ]; then
  echo "[eval] merging LoRA adapter for $EXP (step $STEP)"
  $PY $WORKSPACE/playground/lora_merger.py \
      --model_path $BASE_MODEL --adapter_path $ADAPTER --save_path $MERGE
else
  echo "[eval] merge exists: $MERGE"
fi

# 2) generate responses on GPU 2,3
echo "[eval] generating responses -> $OUT"
CUDA_VISIBLE_DEVICES=2,3 $PY $WORKSPACE/playground/generation.py \
    --model $MERGE --data $DATA --output $OUT --max_tokens 512

# 3) amo_eval: 复用奖励服务打分 + 计算超体积
echo "[eval] scoring + hypervolume"
XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 $PY -m verl.trainer.amo_eval \
    data.path=$OUT \
    data.reward_model_key=extra_info \
    custom_reward_function.path=$REWARD_FUNCTION_PATH
