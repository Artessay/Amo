#!/bin/bash

# End all vLLM server processes
pkill -f "vllm.entrypoints.openai.api_server"

# PID_FILE="vllm_pids.txt"

# if [ -f "$PID_FILE" ]; then
#     echo "Stopping vLLM instances listed in $PID_FILE..."

#     while read -r pid; do
#         echo "Stopping PID $pid..."
#         kill -SIGTERM "$pid"
#     done < "$PID_FILE"

#     echo "All vLLM instances stopped."
# else
#     echo "No PID file found. No vLLM instances to stop."
# fi