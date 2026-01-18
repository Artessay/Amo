#!/bin/bash

NUM_GPUS=4  # Number of GPUs available

MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct"
MODEL_NAME="qwen2.5-1.5b-instruct"

PID_FILE="vllm_pids.txt"

# Clear previous PID file
> "$PID_FILE"

# start vLLM instances on GPUs 0 to N-1
for ((i=0; i<NUM_GPUS; i++));
do
    PORT=$((6180 + i + 1)) 
    echo "Starting vLLM on GPU $i at port $PORT..."
    
    # Launch vLLM server in the background
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
		--model "$MODEL_PATH" \
		--served-model-name "$MODEL_NAME" \
		--port $PORT \
		--tensor-parallel-size 1 \
		--gpu-memory-utilization 0.95 \
		> "vllm_gpu_$i.log" 2>&1 &

    # Store PID in file
    echo $! >> "$PID_FILE"
done

echo "All vLLM instances started. PIDs stored in $PID_FILE."