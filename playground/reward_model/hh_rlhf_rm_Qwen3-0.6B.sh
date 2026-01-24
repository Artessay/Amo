#!/usr/bin/env bash 
set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export DEEPSPEED_BUILD_FUSED_ADAM=1
export CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Path config
MODEL_NAME_OR_PATH="./data/Qwen3/Qwen3-0.6B"  
TRAIN_DATASETS="./playground/reward_model/HH-RLHF" 
TRAIN_TEMPLATE="PKUSafeRLHF"
TRAIN_SPLIT="train"
OUTPUT_DIR="./outputs/Qwen3-0.6B-hh-rlhf-RM"  
# HF env
export HF_HUB_DISABLE_REPO_ID_VALIDATION=1
export HF_HUB_DISABLE_VALIDATION=1

# Source setup.sh
source ./playground/reward_model/setup.sh || true

# Master port
MASTER_PORT=$((RANDOM + 10000))

mkdir -p ${OUTPUT_DIR}
[[ ! -f ${OUTPUT_DIR}/.gitignore ]] && cp -f $0 ${OUTPUT_DIR}/script.sh

deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --learning_rate 2e-5 \
     --epochs 1
