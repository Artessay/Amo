# Running the PKU-SafeRLHF Multi-Objective Baseline Sweep

Complete runbook for the controlled baseline study on **PKU-SafeRLHF**
(2 objectives: `safe_helpfulness`, `safe_harmlessness`) across
**3 models × 11 baseline methods**. Everything below is copy-paste runnable and
is what the `scripts/baseline_trainer/` scripts actually do, so you can carry it
to another machine and run cells in parallel.

- Base env python: `/home/rihongqiu/data/miniconda3/envs/amo/bin/python`
  (conda env `amo`, Python 3.13). Override anywhere with `AMO_PY=...`.
- **GPU rule on the main box:** training + eval use **GPU 0,1 only**
  (`TRAIN_GPUS` / `EVAL_GPUS` env). GPU 2,3 are reserved for other experiments.
- All paths are relative to the repo root `…/Amo` unless absolute.

---

## 0. The matrix

```
models  = { qwen2.5-1.5b , qwen2.5-3b , llama3.2-3b }     # CLI tags: 1.5b 3b llama3b
methods = { ls tchebycheff gdpo_weighted rvpo mgda gapo
            lagrangian fair_stable ctwa dynamic_hv nsga2 smsemoa }   # 11 methods
```

Each cell `<model>_<method>` produces:
- checkpoint `checkpoints/amo_pku-saferlhf/<model_tag>_<method>/global_step_N/`
- result `results/PKU-SafeRLHF/<model_tag>_<method>.{parquet,json}`

`<model_tag>` is `qwen2.5-1.5b` / `qwen2.5-3b` / `llama3.2-3b`.

| method | adv_estimator | reward_manager | isolates |
|---|---|---|---|
| `ls`            | grpo          | amo_scalarize | linear scalarization / MORLHF |
| `tchebycheff`   | grpo          | amo_scalarize | (augmented) Tchebycheff |
| `gdpo_weighted` | gdpo_weighted | amo_vanilla   | per-objective weighted group z-score |
| `rvpo`          | rvpo          | amo_vanilla   | soft-min over objectives |
| `mgda`          | mgda          | amo_vanilla   | min-norm advantage aggregation |
| `gapo`          | gapo          | amo_vanilla   | grad-norm rescaled MGDA |
| `lagrangian`    | grpo          | amo_adaptive  | Safe-RLHF hard constraint (dual ascent) |
| `fair_stable`   | grpo          | amo_adaptive  | Fair-and-Stable mirror-descent weights |
| `ctwa`          | grpo          | amo_adaptive  | covariance-targeted weight adaptation |
| `dynamic_hv`    | grpo          | amo_adaptive  | HV-guided group-level dynamic weighting |
| `nsga2`         | grpo          | amo_pareto    | NSGA-II-style rank+crowding credit |
| `smsemoa`       | grpo          | amo_pareto    | SMS-EMOA-style exclusive-HV credit |

---

## 1. Prerequisites (per machine)

### 1.1 Reward-model gRPC servers (REQUIRED)

Both objectives are served by two 7B reward models. Training and eval score
every rollout by calling them. **Every machine that trains/evals must be able to
reach a pair of servers.**

Start them locally (they occupy 2 GPUs, ~14.6 GB each):

```bash
bash scripts/amo_exp/serve_rewards.sh safe
# helpful (RM) -> GPU0 : port 50051
# harmless (CM)-> GPU1 : port 50052
# logs: /tmp/amo_reward_logs/{helpful,harmless}.log
# stop: bash scripts/amo_exp/serve_rewards.sh stop
```

Checkpoints needed by the servers:
`playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-{RM,CM}` (~27 GB each).

**Pointing training/eval at the servers** — reward functions read these env vars
(default `localhost`):

```bash
export HELPFUL_TARGET_HOST=localhost  HELPFUL_TARGET_PORT=50051
export HARMLESS_TARGET_HOST=localhost HARMLESS_TARGET_PORT=50052
```

To share one machine's servers across several trainer boxes, run the servers on
one host and set `*_TARGET_HOST` to that host's IP on the others (the gRPC
servers already bind `0.0.0.0`). Note: throughput is shared, so co-locating
servers with each machine's own training is usually faster.

### 1.2 Models

```
/data/Qwen/Qwen2.5-1.5B-Instruct
/data/Qwen/Qwen2.5-3B-Instruct
/data/meta-llama/Llama-3.2-3B-Instruct      # gated repo; see below
```

Download LLaMA if missing (needs an HF token with Llama access):

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
/home/rihongqiu/data/miniconda3/envs/amo/bin/hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir /data/meta-llama/Llama-3.2-3B-Instruct \
  --exclude "original/*" "*.pth" --token "$HF_TOKEN"
```

### 1.3 Data + frozen calibration

- Data: `data/PKU-SafeRLHF/{train,test}.parquet` (already in the repo).
- Calibration: `results/PKU-SafeRLHF/safe_calibration.json`.

The calibration is a **frozen, shared** artifact (affine bounds / HV ref /
Tchebycheff ideal / Lagrangian budget for the unbounded reward scale). It must be
**identical across every machine and method** — generate it ONCE and copy the
JSON to the other machines. Do NOT regenerate it per machine.

```bash
# run ONCE (reward servers must be up), then scp the json to other machines
/home/rihongqiu/data/miniconda3/envs/amo/bin/python \
  scripts/baseline_trainer/calibrate_safe.py --n 512 --seed 0
```

---

## 2. Run one cell (train + eval)

```bash
# TRAIN one method on one model (GPU 0,1; 1 full epoch ≈ 144 steps @ batch 512)
bash scripts/baseline_trainer/run_baseline_safe.sh <METHOD> <MODEL> [EPOCH] [hydra overrides...]
#   MODEL  = 1.5b | 3b | llama3b
#   METHOD = ls | tchebycheff | gdpo_weighted | rvpo | mgda | gapo |
#            lagrangian | fair_stable | ctwa | dynamic_hv | nsga2 | smsemoa

# EVAL the trained cell -> results/PKU-SafeRLHF/<model_tag>_<method>.{parquet,json}
bash scripts/baseline_trainer/eval_safe_baseline.sh <model_tag>_<method> <MODEL>
```

Example:

```bash
bash scripts/baseline_trainer/run_baseline_safe.sh rvpo 1.5b
bash scripts/baseline_trainer/eval_safe_baseline.sh qwen2.5-1.5b_rvpo 1.5b
```

Useful env overrides for `run_baseline_safe.sh`:

| var | default | meaning |
|---|---|---|
| `TRAIN_GPUS` | `0,1` | GPUs for training |
| `GPU_MEM_UTIL` | `0.5` | vLLM mem frac (share GPU with reward servers) |
| `MICRO_BATCH_SIZE_PER_GPU` | `16` | lower if OOM |
| `SAVE_FREQ` / `TEST_FREQ` | `50` | checkpoint / periodic-val interval (steps) |
| `RESUME_MODE` | `auto` | `auto` continues from last checkpoint |
| `VAL_BEFORE_TRAIN` | `False` | skip the expensive 8211-prompt pre-val |

For `eval_safe_baseline.sh`: `EVAL_GPUS` (default `0,1`), `GPU_MEM_UTIL`
(default `0.35`), `EVAL_DATA` (default full `test.parquet`).

Smoke test (cheap, verifies the whole path):

```bash
GPU_MEM_UTIL=0.35 MICRO_BATCH_SIZE_PER_GPU=4 RESUME_MODE=disable \
bash scripts/baseline_trainer/run_baseline_safe.sh ls 1.5b 1 \
  trainer.total_training_steps=2 data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16
```

---

## 3. Run the whole matrix (one machine)

```bash
nohup bash scripts/baseline_trainer/run_safe_matrix.sh \
  > train_logs/safe_baselines/matrix.log 2>&1 &
```

The driver is **serial, idempotent, resumable**:
- skips any cell whose `results/PKU-SafeRLHF/<exp>.json` already exists;
- interrupted training resumes from its last checkpoint (`resume_mode=auto`);
- a failing cell is logged and skipped; the run continues;
- refreshes `results/PKU-SafeRLHF/baselines_table.md` after every cell.

Progress:

```bash
tail -f train_logs/safe_baselines/matrix_progress.log     # ledger
cat results/PKU-SafeRLHF/baselines_table.md               # current table
ls train_logs/safe_baselines/*.train.log                  # per-cell logs
```

---

## 4. Parallelize across machines

Because the driver **skips cells that already have a result JSON**, splitting
work = giving each machine a **disjoint subset** of models/methods. The signature
is:

```bash
bash scripts/baseline_trainer/run_safe_matrix.sh "<MODELS>" "<METHODS>" [EPOCH]
#   MODELS/METHODS: space- or comma-separated lists (default = full sets)
```

### Split by model (simplest — 3 machines)

```bash
# machine A
bash scripts/baseline_trainer/run_safe_matrix.sh "1.5b"     # all 11 methods, 1.5B
# machine B
bash scripts/baseline_trainer/run_safe_matrix.sh "3b"       # all 11 methods, 3B
# machine C
bash scripts/baseline_trainer/run_safe_matrix.sh "llama3b"  # all 11 methods, LLaMA
```

### Split by method (finer — e.g. 2 machines on the 3B model)

```bash
# machine A
bash scripts/baseline_trainer/run_safe_matrix.sh "3b" "ls tchebycheff gdpo_weighted rvpo mgda gapo"
# machine B
bash scripts/baseline_trainer/run_safe_matrix.sh "3b" "lagrangian fair_stable ctwa dynamic_hv nsga2 smsemoa"
```

### Or just launch individual cells anywhere

`run_baseline_safe.sh` + `eval_safe_baseline.sh` are fully self-contained; run any
`<method> <model>` pair on any machine that has the models, data, calibration and
reachable reward servers.

### Merging results back

Every result is a standalone `results/PKU-SafeRLHF/<exp>.{parquet,json}`. Collect
the JSONs from all machines into one `results/PKU-SafeRLHF/` and build the table:

```bash
/home/rihongqiu/data/miniconda3/envs/amo/bin/python \
  scripts/baseline_trainer/aggregate_safe.py
# -> results/PKU-SafeRLHF/baselines_table.md
```

**Per-machine checklist before starting:** same repo commit · models present ·
`data/PKU-SafeRLHF/*.parquet` present · **same** `safe_calibration.json` copied
in · reward servers up and env vars pointing at them · a free GPU pair for
training (and, if running its own servers, 2 more GPUs for those).

---

## 5. What "done" looks like

- 33 files `results/PKU-SafeRLHF/<model_tag>_<method>.json`, each with
  `safe_helpfulness`, `safe_harmlessness`, `hypervolume`, `num_prompts`.
- `results/PKU-SafeRLHF/baselines_table.md` filled in for all 3 models
  (reference `grpo`/`gdpo`/`hvpo` rows appear too if those results exist).

## 6. Notes / gotchas

- **Reward scale.** Safe-RLHF rewards are unbounded (~[−4,+6]). Scale-sensitive
  methods (`ls`, `tchebycheff`, `lagrangian`, `dynamic_hv`) consume the frozen
  calibration; the rest are scale-invariant and use raw scores (like GRPO/GDPO).
- **Benign shutdown message.** After the last step you may see
  `RuntimeError: DataLoader worker … killed by signal: Killed` right after
  `Training Progress: 100%`. That's verl's noisy dataloader teardown *after* the
  checkpoint is saved — not a training failure.
- **Don't run two vLLM generations on the same GPUs at once** (shm conflict). The
  serial driver already avoids this; if you launch cells by hand, don't overlap
  an eval-generation with a training rollout on the same GPU pair.
- **HV reference point.** `amo_eval` computes hypervolume at ref = origin (matches
  the existing `qwen2.5-3b_grpo.json`). For a fair cross-method HV with negative
  rewards you may want to recompute HV offline with a shared reference below the
  global min; the raw per-objective means are unaffected.
