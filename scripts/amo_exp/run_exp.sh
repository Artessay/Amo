#!/bin/bash
# Amo 多目标对齐受控实验统一启动脚本 (HVPO vs GRPO)
# 用法: bash run_exp.sh <DATASET> <MODEL> <METHOD> [TRAIN_STEPS] [extra hydra overrides...]
#   DATASET : safe | news
#   MODEL   : 1.5b | 3b
#   METHOD  : grpo | hvpo
#   TRAIN_STEPS: 可选, 纯数字则限定训练步数(快速模式). 缺省用 EPOCH.
#
# 硬约束: 训练仅使用 GPU 2,3 (奖励服务在 GPU 0,1).
set -x
set -e

DATASET=${1:?need DATASET: safe|news}
MODEL=${2:?need MODEL: 1.5b|3b}
METHOD=${3:?need METHOD: grpo|hvpo}
TRAIN_STEPS=${4:-}
shift 3
# 若第4个位置参数是纯数字, 视为 TRAIN_STEPS 并 shift 掉; 其余 $@ 作为 hydra 覆盖透传
if [ -n "$TRAIN_STEPS" ] && [[ "$TRAIN_STEPS" =~ ^[0-9]+$ ]]; then
  shift 1
else
  TRAIN_STEPS=""
fi

# amo 环境的 python (base 环境没有 numpy/verl 依赖)
PY=/home/rihongqiu/data/miniconda3/envs/amo/bin/python

# --- GPU 分配: 训练与奖励模型同卡 (GPU 0,1). 可用 TRAIN_GPUS 覆盖 ---
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

# --- 模型路径 ---
case "$MODEL" in
  1.5b) MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct"; MODEL_TAG="qwen2.5-1.5b" ;;
  3b)   MODEL_PATH="/data/Qwen/Qwen2.5-3B-Instruct";   MODEL_TAG="qwen2.5-3b"  ;;
  *) echo "bad MODEL $MODEL"; exit 1 ;;
esac

# --- 方法 -> adv_estimator + reward_manager ---
case "$METHOD" in
  grpo) ADV="grpo"; RM="amo_vanilla" ;;
  hvpo) ADV="hvpo"; RM="amo_hvpo"    ;;
  *) echo "bad METHOD $METHOD"; exit 1 ;;
esac

# --- 数据集 -> 数据文件 + 多目标 reward 函数列表 + 长度 ---
case "$DATASET" in
  safe)
    PROJECT_NAME="amo_pku-saferlhf"
    TRAIN_FILES="$WORKSPACE/data/PKU-SafeRLHF/train.parquet"
    VAL_FILES="$WORKSPACE/data/PKU-SafeRLHF/test.parquet"
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"
    MAX_PROMPT=512; MAX_RESP=512; MICRO_BS=16
    ;;
  news)
    PROJECT_NAME="amo_cnn_dailymail"
    TRAIN_FILES="$WORKSPACE/data/CNN_DailyMail/train.parquet"
    VAL_FILES="$WORKSPACE/data/CNN_DailyMail/test.parquet"
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_news/news_coherence.py','$WORKSPACE/recipe/amo_news/news_fluency.py','$WORKSPACE/recipe/amo_news/news_relevance.py','$WORKSPACE/recipe/amo_news/news_consistency.py']"
    MAX_PROMPT=2048; MAX_RESP=1024; MICRO_BS=8
    ;;
  *) echo "bad DATASET $DATASET"; exit 1 ;;
esac

EXPERIMENT_NAME="${MODEL_TAG}_${METHOD}"
EPOCH=15

# 快速模式: 若给了 TRAIN_STEPS 则限定步数
STEP_ARG=""
if [ -n "$TRAIN_STEPS" ]; then
  STEP_ARG="trainer.total_training_steps=$TRAIN_STEPS"
fi

NUM_NODES=1
NUM_GPUS_PER_NODE=2
TENSOR_MODEL_PARALLEL_SIZE=1

$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    amo_strategy.enable=True \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=512 \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$RM \
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-20} \
    trainer.total_epochs=$EPOCH \
    $STEP_ARG "$@"
