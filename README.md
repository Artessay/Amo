# Amo

Aligning LLMs with Multiple Objects

## ⚙️ Environment

```bash
conda create -n amo python=3.13 -y
conda activate amo

pip install -e .
```


## Download LLMs from ModelScope

```bash
pip install modelscope
```

```bash
modelscope download --model Qwen/Qwen3-0.6B  --local_dir ~/data/model/Qwen/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-4B  --local_dir ~/data/model/Qwen/Qwen3-4B
modelscope download --model Qwen/Qwen3-8B  --local_dir ~/data/model/Qwen/Qwen3-8B

modelscope download --model google/gemma-3-4b-it  --local_dir ~/data/model/google/gemma-3-4b-it
modelscope download --model google/gemma-3-12b-it  --local_dir ~/data/model/google/gemma-3-12b-it
```