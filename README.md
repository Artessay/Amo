# Amo

Aligning LLMs with Multiple Objects

## ⚙️ Environment

```bash
conda create -n amo python=3.13 -y
conda activate amo

pip install -e .[vllm]

# Install flash attention 2, you can download it from https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```

Download `nltk` punkt tokenizer.

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
```

Install `python-dotenv` to load environment variables from `.env` file.

```bash
pip install python-dotenv
```

Login to `wandb`

```bash
wandb login
```


## Download LLMs from ModelScope

```bash
pip install modelscope
```

```bash
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct  --local_dir /data/Qwen/Qwen2.5-0.5B-Instruct
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct  --local_dir /data/Qwen/Qwen2.5-1.5B-Instruct
modelscope download --model Qwen/Qwen2.5-3B-Instruct  --local_dir /data/Qwen/Qwen2.5-3B-Instruct
modelscope download --model Qwen/Qwen2.5-7B-Instruct  --local_dir /data/Qwen/Qwen2.5-7B-Instruct

modelscope download --model Qwen/Qwen3-0.6B  --local_dir /data/Qwen/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-1.7B  --local_dir /data/Qwen/Qwen3-1.7B
modelscope download --model Qwen/Qwen3-4B  --local_dir /data/Qwen/Qwen3-4B
modelscope download --model Qwen/Qwen3-8B  --local_dir /data/Qwen/Qwen3-8B

modelscope download --model LLM-Research/Llama-3.2-3B-Instruct  --local_dir /data/meta-llama/Llama-3.2-3B-Instruct
```
