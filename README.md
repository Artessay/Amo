# Amo

Aligning LLMs with Multiple Objects (Amo) is a project focused on developing and evaluating methods for aligning large language models with multiple objectives. The project implements various reinforcement learning techniques to improve LLM performance across different tasks and metrics.

## 📁 Project Structure

The Amo project is organized into the following main folders:

- **data/**: Datasets used for training and evaluation
- **recipe/**: Evaluation metrics and scoring recipes for different tasks
- **scripts/**: Training and evaluation scripts
- **checkpoints/**: Resumable trainer state and merged model weights
- **results/**: Generated test-set responses and machine-readable evaluation results
- **verl/**: Reinforcement learning framework and utilities

## Required Experiment Contract

The following rules are **mandatory for every new method, task, and experiment**.
An experiment is not complete when training stops or a checkpoint exists. It is
complete only after the selected checkpoint has been merged, used for test-set
inference, evaluated, and its standard result files have been written under
`results/`.

### Scripts

- Every method MUST have its own `scripts/trainers/<method>/` directory with a
  separately runnable, executable `run_<dataset>.sh` for each supported
  dataset. The public launcher contract is
  `scripts/trainers/<method>/run_<dataset>.sh [MODEL] [EPOCH]`; for example,
  `bash scripts/trainers/hvpo/run_pku-saferlhf.sh 3b 1`. Shared implementation
  belongs under `scripts/trainers/_common/`, but a generic dispatcher alone is
  not a compliant per-method launcher.
- Every task MUST have `scripts/eval_<task>.sh`; for example,
  `scripts/eval_safe.sh` and `scripts/eval_detox.sh`. A new task must also be
  supported by `scripts/merge_model.sh` and `scripts/inference.sh`, or provide
  thin wrappers with the same inputs and output layout.
- Launchers MUST work from the repository root, derive the workspace from their
  own location, use repository-relative data/output paths, and keep one
  experiment name unchanged across training, merge, inference, and evaluation.
  Machine-specific model paths and GPU selection may be configurable, but must
  not change the artifact names.

### Names and artifact layout

Use a stable experiment name such as `<model-tag>_<method>`; append an explicit
suffix for a real experimental dimension such as `_seed42` or a named ablation.
Do not encode a temporary machine, GPU, or rerun name in it.

Given dataset directory name `<Dataset>`, lower-case task slug `<task-slug>`,
and experiment name `<experiment>`, the canonical layout is:

```text
checkpoints/amo_<task-slug>/<experiment>/
  latest_checkpointed_iteration.txt
  global_step_<N>/
    actor/
      merge/                         # merged model, when required

results/<Dataset>/
  <experiment>.parquet              # test prompts and generated responses
  <experiment>.json                 # aggregate evaluation metrics
```

For example, PKU-SafeRLHF uses project `amo_pku-saferlhf` and produces
`results/PKU-SafeRLHF/qwen2.5-1.5b_grpo.{parquet,json}`. The Parquet and JSON
files MUST have the same experiment stem as the checkpoint directory.

`checkpoints/` is only for model/trainer state, including LoRA/FSDP shards,
optimizer state, and merged weights. `results/` is only for auditable generated
responses, metrics, tables, plots, and small diagnostics. **Never place a
checkpoint or merged model below `results/`.** Optional diagnostics may be
stored below `results/<Dataset>/diagnostics/`, but they do not replace the
required top-level `<experiment>.parquet` and `<experiment>.json` pair.

### Required completion sequence

For every trained experiment, all four stages MUST succeed:

1. Train and save a resumable checkpoint under `checkpoints/amo_<task-slug>/`.
2. Merge the selected/latest actor checkpoint with `scripts/merge_model.sh` (or
   an equivalent task wrapper).
3. Run test-set generation with `scripts/inference.sh`, producing
   `results/<Dataset>/<experiment>.parquet`.
4. Run `scripts/eval_<task>.sh`, producing the sibling
   `results/<Dataset>/<experiment>.json`.

Before reporting a run as done, verify that both result files exist, are
non-empty, and correspond to the intended checkpoint. A checkpoint-only run is
unfinished and must not be included as a completed comparison. Method/task
changes should update their trainer launcher, merge/inference selection, and
evaluation script in the same change.

## 📋 Scripts Folder

The `scripts/` folder contains training and evaluation scripts for different alignment methods:

### Trainer Scripts

- **trainers/**: Unified, method-oriented training launchers
  - **_common/**: Shared launch logic plus dataset and model profiles
  - **&lt;method&gt;/**: One `method.sh` and one independently runnable
    `run_<dataset>.sh` per supported dataset (including `grpo/`, `gdpo/`,
    `hvpo/`, and baseline methods)
  - **orchestration/**: Sequential and matrix experiment runners
  - **tools/**: Trainer-specific calibration, evaluation, and aggregation tools

All public method launchers use the same interface:

```bash
bash scripts/trainers/<method>/run_<dataset>.sh [MODEL] [EPOCH]
```

When omitted, `MODEL` and `EPOCH` come from the shared model and dataset
profiles, respectively.

### Evaluation Scripts

- **eval_math.sh**: Evaluates mathematical reasoning capabilities
- **eval_news.sh**: Evaluates news summarization quality
- **eval_safe.sh**: Evaluates safety and harmlessness
- **eval_tool.sh**: Evaluates tool use and execution
- **eval_detox.sh**: Evaluates text detoxification (ParaDetox) across three
  conflicting objectives: style transfer accuracy (toxicity removed), content
  preservation (semantic similarity), and fluency

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

pip install -e .[vllm]

# Install flash attention 2, you can download it from https://github.com/Dao-AILab/flash-attention/releases
# For example, `wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl`
pip install flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```

Download `nltk` punkt tokenizer for math evaluator.

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
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

## 🚀 Usage Examples

### Training Models

To train a model using GRPO on the math dataset:

```bash
bash scripts/trainers/grpo/run_math-lighteval.sh 3b 50
```

To train a model using HVPO on the news dataset:

```bash
bash scripts/trainers/hvpo/run_news.sh 3b 15
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

To evaluate a model's text detoxification quality (ParaDetox), first start the
three reward servers (STA / SIM / FL), then run the eval script:

```bash
bash recipe/amo_detox/start_sta.sh   # port 50060, toxicity classifier
bash recipe/amo_detox/start_sim.sh   # port 50061, LaBSE similarity
bash recipe/amo_detox/start_fl.sh    # port 50062, CoLA fluency
bash scripts/eval_detox.sh
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
