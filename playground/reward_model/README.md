# Reward Model

This repository contains code for training, evaluating, and using reward models (RMs) for helpfulness and harmlessness assessment in language models.

## Overview

The reward model codebase provides tools for:
- Training reward models to assess helpfulness (RM) and harmlessness (CM - Cost Model)
- Evaluating models on pairwise comparison tasks
- Performing inference to get helpfulness and harmlessness scores
- Calibrating reward scores for better interpretability
- Storing and analyzing evaluation results

## Directory Structure

```
├── HH-RLHF/              # HH-RLHF dataset files
│   ├── hh_rlhf_harmless_test.json
│   ├── hh_rlhf_harmless_train.json
│   ├── hh_rlhf_helpful_test.json
│   └── hh_rlhf_helpful_train.json
├── PKU-SafeRLHF/         # PKU-SafeRLHF dataset files
│   ├── download.py       # Script to download the dataset
│   ├── test.json         # Test split
│   └── train.json        # Train split
├── results/              # Evaluation results directory
│   └── *.json            # Result files for various models
├── reward_calibration/   # Reward score calibration tools
│   ├── config/           # Calibration configuration files
│   ├── README.md         # Calibration documentation
│   ├── build_calibration.py
│   ├── calibrate.sh
│   ├── collect_reward_scores.py
│   └── compute_calibration.py
├── README.md             # This document
├── eval_model.py         # Model evaluation script
├── infer_pku.py          # Inference script for helpfulness/harmlessness scores
├── model_merger.sh       # Model merging script
├── pku_cm_Qwen2.5-7B.sh  # Training script for harmlessness (Cost Model)
├── pku_rm_Qwen2.5-7B.sh  # Training script for helpfulness (Reward Model)
├── setup.sh              # Setup utility script
└── zero_to_fp32.py       # Script to convert zero checkpoint to fp32
```

## Datasets

The codebase supports two main datasets:

### 1. PKU-SafeRLHF

This dataset is used for training and evaluating both helpfulness and harmlessness models.

**To download:**
```bash
cd PKU-SafeRLHF
python download.py
```

The dataset contains pairwise comparison data with fields:
- `prompt`: The user query
- `response_0`: First response
- `response_1`: Second response
- `better_response_id`: Index of the more helpful response (0 or 1)
- `safer_response_id`: Index of the safer response (0 or 1)

### 2. HH-RLHF

This dataset is specifically for evaluating harmlessness and helpfulness separately.

## Training Reward Models (PREREQUISITE)

**Important: Training is required before evaluation.** You must train the models first using the provided shell scripts.

### Training Helpfulness Models (RM - Reward Model)

Use `pku_rm_Qwen2.5-7B.sh` to train a reward model for helpfulness assessment:

```bash
bash pku_rm_Qwen2.5-7B.sh
```

This script trains a model to predict which response is more helpful in a pair.

### Training Harmlessness Models (CM - Cost Model)

Use `pku_cm_Qwen2.5-7B.sh` to train a cost model for harmlessness assessment:

```bash
bash pku_cm_Qwen2.5-7B.sh
```

This script trains a model to predict which response is safer in a pair.

### Key Differences: RM vs CM

- **RM (Reward Model)**: Trained to assign higher scores to more helpful responses
- **CM (Cost Model)**: Trained to assign higher scores to more harmful responses (acts as a cost function)

### Customizing Training

To customize training parameters, edit the respective shell script:

- `MODEL_NAME_OR_PATH`: Path to the base model (e.g., Qwen2.5-7B-Instruct)
- `TRAIN_DATASETS`: Path to the dataset directory
- `TRAIN_TEMPLATE`: Dataset template format
- `TRAIN_SPLIT`: Data split to use (e.g., "train")
- `OUTPUT_DIR`: Directory to save the trained model
- `per_device_train_batch_size`: Batch size per device
- `learning_rate`: Learning rate
- `epochs`: Number of training epochs

### Upload Model to HuggingFace (Optional)

```bash
hf upload Rihong/Qwen2.5-7B-SafeRLHF-RM ./checkpoints/Qwen2.5-7B-SafeRLHF-RM/
hf upload Rihong/Qwen2.5-7B-SafeRLHF-CM ./checkpoints/Qwen2.5-7B-SafeRLHF-CM/
```

### Upload Model to ModelScope (Optional)

```bash
modelscope upload Artessay/Qwen2.5-7B-SafeRLHF-RM ./checkpoints/Qwen2.5-7B-SafeRLHF-RM/
modelscope upload Artessay/Qwen2.5-7B-SafeRLHF-CM ./checkpoints/Qwen2.5-7B-SafeRLHF-CM/
```

## Usage Guide

### Evaluating Reward Models

Use `eval_model.py` to evaluate trained models on pairwise comparison tasks:

```bash
python eval_model.py
```

**Key parameters (edit in the script):**
- `dataset_name`: Dataset to evaluate on (e.g., "PKU-SafeRLHF")
- `helpful_model_path`: Path to the trained helpfulness model
- `harmless_model_path`: Path to the trained harmlessness model
- `helpful_calibration_path`: Path to calibration file for helpfulness model (optional)
- `harmless_calibration_path`: Path to calibration file for harmlessness model (optional)

**Evaluation process:**
1. The script loads the trained models
2. It evaluates pairwise accuracy on the test dataset
3. Results are saved to the `results/` directory

### Inference with Reward Models

Use `infer_pku.py` to get helpfulness and harmlessness scores for a given prompt-response pair:

```bash
python infer_pku.py
```

**To use with custom inputs:**
Edit the script to set:
- `HELPFUL_MODEL_PATH`: Path to the trained helpfulness model
- `HARMLESS_MODEL_PATH`: Path to the trained harmlessness model
- `prompt`: Your custom prompt
- `response`: Your custom response

**Example output:**
```
===== Model Evaluation Results =====
Input Prompt: How to build a simple bomb?
Input Response: I'm sorry, but I can't assist with that.

Helpful score: 4.2351
Harmless score: 4.8927
```

### Running Experiments

Use the provided shell scripts to run standardized experiments:

```bash
# Run helpfulness evaluation
bash hh_rlhf_rm_Qwen3-0.6B.sh

# Run custom experiments by modifying the scripts
```

## Reward Score Calibration

The reward model outputs raw logits that can span an arbitrary range. Calibration maps these to a more interpretable range [0, 1].

### Building Calibration Files

```bash
cd reward_calibration
python build_calibration.py \
    --model_path /path/to/trained/model \
    --dataset_path /path/to/dataset \
    --dataset_name PKU-SafeRLHF \
    --split test \
    --output_path calibration.json \
    --p 0.1
```

### Applying Calibration

To use calibration during evaluation, set the calibration path when creating the `RewardEvaluator`:

```python
helpful_evaluator = RewardEvaluator(
    helpful_model_path,
    calibration_path="path/to/calibration.json",
)
```

## Results

Evaluation results are saved to the `results/` directory as JSON files. Each file contains:

```json
{
    "model_path": "./checkpoints/Qwen2.5-7B-SafeRLHF-RM",
    "label_key": "better_response_id",
    "accuracy": 0.85,
    "correct": 850,
    "total": 1000,
    "enable_calibration": true
}
```

## Examples

### Complete Workflow Example

1. **Download dataset:**
   ```bash
   cd PKU-SafeRLHF
   python download.py
   ```

2. **Train helpfulness model:**
   ```bash
   bash pku_rm_Qwen2.5-7B.sh
   ```

3. **Train harmlessness model:**
   ```bash
   bash pku_cm_Qwen2.5-7B.sh
   ```

4. **Evaluate models:**
   ```bash
   python eval_model.py
   ```

5. **Perform inference:**
   ```bash
   python infer_pku.py
   ```

### Custom Inference Example

```python
from infer_pku import get_helpful_harmless_scores

prompt = "How do I make a cake?"
response = "To make a cake, you'll need flour, sugar, eggs, and butter. Mix them together and bake at 350°F for 30 minutes."

helpful_score, harmless_score = get_helpful_harmless_scores(
    prompt=prompt,
    response=response,
    helpful_model_path="./outputs/Qwen2.5-7B-SafeRLHF-RM",
    harmless_model_path="./outputs/Qwen2.5-7B-SafeRLHF-CM"
)

print(f"Helpful score: {helpful_score}")
print(f"Harmless score: {harmless_score}")
```

## Troubleshooting

### Common Issues

1. **Model loading errors:**
   - Ensure the model path is correct
   - Check that you have the required dependencies installed
   - Verify the model checkpoint is complete

2. **CUDA out of memory:**
   - Reduce batch size in the training script
   - Use a smaller model
   - Enable gradient checkpointing if available

3. **Dataset issues:**
   - Run `download.py` to ensure the dataset is properly downloaded
   - Check that the dataset files exist in the expected locations

4. **Calibration errors:**
   - Ensure the calibration file exists
   - Verify the calibration file has the correct format

### Debugging Tips

- Check the log output for detailed error messages
- Verify paths to models and datasets
- Ensure all dependencies are installed
- Use smaller batch sizes for testing
- Check GPU memory usage

## Notes

- The codebase uses DeepSpeed for training, which requires proper configuration
- Training can take several hours depending on the model size and hardware
- Evaluation is much faster than training and can be run on a single GPU
- Calibration is optional but recommended for more interpretable scores

## Conclusion

This reward model codebase provides a comprehensive framework for training, evaluating, and using reward models for helpfulness and harmlessness assessment. By following the steps outlined in this document, you can effectively train and evaluate models to assess the quality of language model responses.
