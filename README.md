# Amo

Aligning LLMs with Multiple Objects (Amo) is a project focused on developing and evaluating methods for aligning large language models with multiple objectives. The project implements various reinforcement learning techniques to improve LLM performance across different tasks and metrics.

## 📁 Project Structure

The Amo project is organized into the following main folders:

- **data/**: Datasets used for training and evaluation
- **recipe/**: Evaluation metrics and scoring recipes for different tasks
- **scripts/**: Training and evaluation scripts
- **verl/**: Reinforcement learning framework and utilities

## 📋 Scripts Folder

The `scripts/` folder contains training and evaluation scripts for different alignment methods:

### Trainer Scripts

- **gdpo_trainer/**: Scripts for Generative Direct Preference Optimization
- **grpo_trainer/**: Scripts for Group Relative Policy Optimization
- **hvpo_trainer/**: Scripts for Hypervolume-Guided Policy Optimization

### Evaluation Scripts

- **eval_math.sh**: Evaluates mathematical reasoning capabilities
- **eval_news.sh**: Evaluates news summarization quality
- **eval_safe.sh**: Evaluates safety and harmlessness
- **eval_tool.sh**: Evaluates tool use and execution

### Utility Scripts

- **inference.sh**: Performs model inference
- **merge_model.sh**: Merges FSDP checkpoints or LoRA adapters into base models

### Group Relative Policy Optimization (GRPO)

GRPO is a reinforcement learning algorithm that eliminates the need for a separate critic model by:
- Generating multiple solutions for each problem (group sampling)
- Assigning rewards based on solution quality
- Using the average group reward as a baseline
- Updating the model by comparing each solution's reward to the group baseline

## ⚙️ Environment

### Install Dependencies for Reinforcement Learning

```bash
conda create -n amo python=3.13 -y
conda activate amo

pip install -e .[amo,vllm]

# Install flash attention 2, you can download it from https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```


Install `grpcio-tools` to generate gRPC code.

```bash
pip install grpcio-tools
```

Install `python-dotenv` to load environment variables from `.env` file.

```bash
pip install python-dotenv
```

### Login to SwanLab for Experiment Tracking (Optional)

Login to `swanlab` to track experiments and visualize results.

```bash
swanlab login
```


## Download LLMs from ModelScope

Install `modelscope` to download pre-trained models from ModelScope.

```bash
pip install modelscope
```

Download necessary models from ModelScope for training and evaluation.

```bash
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct  --local_dir /data/Qwen/Qwen2.5-1.5B-Instruct
modelscope download --model Qwen/Qwen2.5-3B-Instruct  --local_dir /data/Qwen/Qwen2.5-3B-Instruct
modelscope download --model LLM-Research/Llama-3.2-3B-Instruct  --local_dir /data/meta-llama/Llama-3.2-3B-Instruct
```

Download the reward models from ModelScope for PKU-SafeRLHF.

```bash
modelscope download --model PKU-Alignment/beaver-7b-v3.0-reward  --local_dir /data/PKU-Alignment/beaver-7b-v3.0-reward
modelscope download --model PKU-Alignment/beaver-7b-v3.0-cost  --local_dir /data/PKU-Alignment/beaver-7b-v3.0-cost
```

## 🚀 Usage Examples

### Training Models

To train a model using GRPO on the math dataset:

```bash
bash scripts/grpo_trainer/run_qwen2.5-3b_math-lighteval.sh
```

To train a model using HVPO on the news dataset:

```bash
bash scripts/hvpo_trainer/run_qwen2.5-3b_news.sh
```

### Evaluating Models

To evaluate a model's mathematical reasoning capabilities:

```bash
bash scripts/eval_math.sh
```

To evaluate a model's news summarization quality:

```bash
bash scripts/eval_news.sh
```

### Running Inference

To run inference with a trained model:

```bash
bash scripts/inference.sh
```

### Merging LoRA Adapters

To merge LoRA adapters into a base model:

```bash
bash scripts/merge_model.sh
```
