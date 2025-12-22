# Reward Model Calibration

This document explains a simple post-training calibration method for the reward model (RM) outputs. Calibration maps the raw reward logits, which can span an arbitrary range (e.g., `[-5, 5]`), to a more interpretable probability-like score in `[0, 1]`.

## 1. Rationale

The RM is trained using a pairwise ranking loss, such as `-log(sigmoid(score_chosen - score_rejected))`. This loss function is sensitive only to the *difference* in scores, not their absolute values. As a result, the raw output logits of a trained RM can have any scale and shift. While this is sufficient for training, it makes the scores difficult to interpret or compare across models.

For inference, analysis, and downstream applications (like some forms of RL), it is often useful to have a reward score that behaves like a probability (e.g., the probability that a response is "good").

**This calibration is an inference-only, post-processing step.** It does not affect the training process, which continues to rely on the raw, uncalibrated logits to preserve the semantics of the ranking loss.

## 2. Calibration Method: Shift & Temperature

We use a simple and effective calibration method based on a location-scale transformation of the raw logits, followed by a sigmoid function. The calibrated score `s` is computed from a raw logit `r` as:

```
s = sigmoid((r - alpha) / beta)
```

Here:
- `alpha` is a **shift** parameter that centers the logit distribution.
- `beta` is a **temperature** parameter that controls the "steepness" of the sigmoid, effectively rescaling the logits. A `beta > 1` "flattens" the sigmoid, making the scores less extreme, while `0 < beta < 1` makes it steeper.

### Estimating `alpha` and `beta` via Quantiles

The parameters `alpha` and `beta` are estimated from the distribution of raw reward scores on a reference dataset (e.g., the training or validation set). We define two quantile-based constraints to solve for them.

Let `r_all` be the tensor of all raw reward scores collected from the dataset. We define:
- `q_lo = quantile(r_all, p)` (the p-th percentile of scores).
- `q_hi = quantile(r_all, 1 - p)` (the (1-p)-th percentile of scores).

We want our calibrated score `s(r)` to satisfy:
- `s(q_lo) ≈ p`
- `s(q_hi) ≈ 1 - p`

For example, with `p = 0.1`, we are setting the 10th percentile of raw scores to map to a calibrated score of `0.1`, and the 90th percentile to `0.9`.

By taking the inverse of the sigmoid (the logit function), we get a system of two linear equations:

1. `(q_lo - alpha) / beta = logit(p)`
2. `(q_hi - alpha) / beta = logit(1 - p)`

Solving this system for `alpha` and `beta` gives us the calibration parameters. These can be pre-computed once and stored in a simple JSON file.

## 3. Usage

The workflow consists of two main steps:
1. **Build Calibration**: Run a script to compute and save the `alpha` and `beta` parameters from a dataset.
2. **Apply Calibration**: Pass the path to the saved calibration file during inference (in the evaluation script or the reward server).

### Step 1: Build the Calibration File

Use the `playground/reward_calibration/build_calibration.py` script to generate the `rm_calibration.json` file. This script loads a trained RM, iterates over a dataset, collects all raw `end_scores`, computes `alpha` and `beta`, and saves them.

**Example Command:**

```bash
python ./playground/reward_model/build_calibration.py \
    --model_path /path/to/your/trained/rm \
    --dataset_path ./Amo/data \
    --dataset_name PKU-SafeRLHF \
    --split test \
    --output_path Amo/playground/reward_model/rm_calibration.json \
    --p 0.1
```

**Arguments:**
- `--model_path`: Path to the reward model checkpoint.
- `--dataset_path`: Root directory containing datasets.
- `--dataset_name`: The specific dataset to use (e.g., `PKU-SafeRLHF`).
- `--split`: The data split to use (e.g., `train`, `test`).
- `--output_path`: Path to save the resulting JSON file.
- `--p`: The quantile probability (default: `0.1`).

**Expected JSON Output (`rm_calibration.json`):**

The script will produce a simple JSON file with the estimated parameters.

```json
{"alpha":2.5134,"beta":1.8729}
```

### Step 2: Use the Calibration File

Once you have the `rm_calibration.json` file, you can provide its path to the evaluation script or the reward server to enable calibrated scoring.

#### Evaluation with `eval_model.py`

The `RewardEvaluator` class in `playground/reward_model/eval_model.py` now accepts an optional `calibration_path`.

To use it, modify the main block of the script to pass the path:

```python
# In Amo/playground/reward_model/eval_model.py

if __name__ == "__main__":
    # ... (other setup)

    # Set the path to your pre-computed calibration file.
    helpful_calibration_path = "Amo/playground/reward_model/rm_calibration.json"

    helpful_evaluator = RewardEvaluator(
        helpful_model_path,
        calibration_path=helpful_calibration_path,
    )
    # ...
```

When calibration is active, `get_reward_score` will return the calibrated score. The evaluation script will also print a notice indicating that calibration is active.

#### Reward Server with `reward_server.py`

The `recipe/amo_safe/reward_server.py` script now accepts an optional `--calibration_path` argument.

**Example Command:**

```bash
python Amo/recipe/amo_safe/reward_server.py \
    --model_path /path/to/your/trained/rm \
    --calibration_path Amo/playground/reward_model/rm_calibration.json
```

When active, the server will log both the raw and calibrated scores for each request, but will return the **calibrated score** in the gRPC response. If the path is not provided, it continues to return the raw logit as before.
