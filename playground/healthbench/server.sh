# Launches a server at default port 8000
python -m vllm.entrypoints.openai.api_server \
        --model m42-health/Llama3-Med42-70B \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --enable_prefix_caching \
        --served-model-name med42-v2-70b