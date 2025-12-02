# Amo

Aligning LLMs with Multiple Objects

## ⚙️ Environment

```bash
conda create -n amo python=3.13 -y
conda activate amo

pip install torch==2.8.0
pip install -e .[vllm]

# Install flash attention 2, you can download it from https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```

Download `nltk` punkt tokenizer.

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
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
modelscope download --model Qwen/Qwen3-0.6B  --local_dir /data/Qwen/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-4B  --local_dir /data/Qwen/Qwen3-4B
modelscope download --model Qwen/Qwen3-8B  --local_dir /data/Qwen/Qwen3-8B

modelscope download --model google/gemma-3-4b-it  --local_dir /data/google/gemma-3-4b-it
modelscope download --model google/gemma-3-12b-it  --local_dir /data/google/gemma-3-12b-it
```

Reward model and cost model for `PKU-SafeRLHF` dataset.

```bash
modelscope download --model PKU-Alignment/beaver-7b-v3.0-reward --local_dir /data/PKU-Alignment/beaver-7b-v3.0-reward
modelscope download --model PKU-Alignment/beaver-7b-v3.0-cost --local_dir /data/PKU-Alignment/beaver-7b-v3.0-cost
```
