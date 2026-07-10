#!/bin/bash
# 启动 Amo 奖励服务, 锁定在 GPU 0,1 (训练用 GPU 2,3)
# 用法: bash serve_rewards.sh <safe|news|stop>
set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
PY=/home/rihongqiu/data/miniconda3/envs/amo/bin/python
LOGDIR=/tmp/amo_reward_logs
mkdir -p $LOGDIR

# align_anything 通过 diffusers 触发 xformers 的 flash-attn 版本检查; flash-attn 2.8.3
# 超出 xformers 允许范围. 这是 xformers 官方提供的跳过开关 (见 fmha/flash.py).
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

MODE=${1:?need safe|news|stop}

case "$MODE" in
  safe)
    RM_PATH=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-RM
    CM_PATH=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-CM
    # helpful (RM) -> GPU 0, port 50051
    CUDA_VISIBLE_DEVICES=0 setsid bash -c "XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 $PY $WORKSPACE/recipe/amo_safe/reward_server.py --model_path $RM_PATH --port 50051 > $LOGDIR/helpful.log 2>&1" &
    echo "helpful(RM) launched on GPU0:50051"
    # harmless (CM) -> GPU 1, port 50052
    CUDA_VISIBLE_DEVICES=1 setsid bash -c "XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 $PY $WORKSPACE/recipe/amo_safe/reward_server.py --model_path $CM_PATH --port 50052 > $LOGDIR/harmless.log 2>&1" &
    echo "harmless(CM) launched on GPU1:50052"
    ;;
  news)
    cd $WORKSPACE/recipe/amo_news
    CUDA_VISIBLE_DEVICES=0,1 setsid bash -c "XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 $PY summarization_server.py --port 50053 > $LOGDIR/news.log 2>&1" &
    echo "news(UniEval) launched on GPU0,1:50053"
    ;;
  stop)
    pkill -9 -f "recipe/amo_safe/reward_server.py" && echo "stopped safe servers" || echo "no safe servers"
    pkill -9 -f "summarization_server.py" && echo "stopped news server" || echo "no news server"
    ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac
