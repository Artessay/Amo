# Amo Multi-Objective Alignment Baselines

Reproductions of published / widely-used multi-objective LLM-alignment baselines
for comparison with **HVPO**, integrated into the *same* controlled online-RL
pipeline so the only experimental variable is the multi-objective credit rule.

All baselines share the identical base model, dataset, rollout count (`rollout.n`),
reward functions, KL penalty, optimizer and training-token budget as the existing
`grpo` / `gdpo` / `hvpo` scripts. Each method changes exactly one of:

1. **reward scalarization** (reward manager) — how the per-objective score vector
   becomes a scalar reward;
2. **advantage construction** (advantage estimator) — how per-objective signals
   are combined into an advantage;
3. **adaptive state** — weights / duals updated across steps.

## Method map

| Method (`run_baseline_math.sh <METHOD>`) | Track | adv_estimator | reward_manager | What it isolates |
|---|---|---|---|---|
| `ls`            | scalarize | `grpo`          | `amo_scalarize` | Linear scalarization / MORLHF (weight sweep) — the minimum bar |
| `tchebycheff`   | scalarize | `grpo`          | `amo_scalarize` | (Augmented) Tchebycheff — reaches concave front regions |
| `gdpo_weighted` | GDPO-family | `gdpo_weighted` | `amo_vanilla` | Per-objective weighted group z-score |
| `rvpo`          | GDPO-family | `rvpo`          | `amo_vanilla` | Soft-min over objectives (worst-objective emphasis) |
| `mgda`          | grad-agg | `mgda`          | `amo_vanilla` | Min-norm advantage aggregation (advantage-space proxy) |
| `gapo`          | grad-agg | `gapo`          | `amo_vanilla` | Grad-norm rescaled MGDA (advantage-space proxy) |
| `lagrangian`    | constrained | `grpo`       | `amo_adaptive` | Safe-RLHF-style hard-constraint (dual ascent) |
| `fair_stable`   | adaptive-w | `grpo`        | `amo_adaptive` | Fair-and-Stable mirror-descent reward composition |
| `ctwa`          | adaptive-w | `grpo`        | `amo_adaptive` | Covariance-targeted weight adaptation |
| `dynamic_hv`    | adaptive-w | `grpo`        | `amo_adaptive` | HV-guided **group-level** dynamic weighting |
| `nsga2`         | Pareto-credit | `grpo`     | `amo_pareto` | NSGA-II-style rank + crowding response credit |
| `smsemoa`       | Pareto-credit | `grpo`     | `amo_pareto` | SMS-EMOA-style exclusive-HV response credit |

> **Reporting caveats.** `mgda` / `gapo` here operate in *advantage space* (each
> objective's per-sample group-normalized advantage stands in for its gradient),
> a clearly-labelled efficient adaptation of the last-layer MGDA/GAPO trick — not
> a faithful full-parameter-gradient MGDA. `nsga2` / `smsemoa` adapt EA
> survival rules into a per-response scalar credit; call them
> "NSGA-II-style / SMS-EMOA-style response credit", not "NSGA-II training an LLM".
> `dynamic_hv` uses HV at the **group/meta** level, distinct from HVPO's
> per-response exclusive-HV credit.

## Usage

```bash
# Full run (50 epochs), 1.5B model, MATH task (3 local objectives, no reward server):
bash scripts/baseline_trainer/run_baseline_math.sh ls
bash scripts/baseline_trainer/run_baseline_math.sh rvpo
bash scripts/baseline_trainer/run_baseline_math.sh ctwa

# 2-step smoke test (verifies the full pipeline cheaply):
bash scripts/baseline_trainer/run_baseline_math.sh ls 1.5b 1 trainer.total_training_steps=2 \
    data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16

# Smoke ALL methods in one shot:
bash scripts/baseline_trainer/smoke_all_baselines.sh
```

Per-objective weights / ideal points / constraint budgets / covariance targets
are configured under `amo_strategy.{scalarize,adaptive,pareto}_config` (see
`verl/trainer/config/ppo_trainer.yaml`) and the advantage hyperparameters
(`amo_objective_weights`, `rvpo_k`, `gapo_p`) under `algorithm.*`. Choose them on
a frozen calibration split and reuse the same values across every method.

## Fair-comparison notes

* The MATH objectives (accuracy, conciseness, format) are already in `[0,1]`, so
  the default HV reference point (origin) and Tchebycheff ideal (all-ones) are
  correct. For tasks with unbounded reward scales (e.g. Safe-RLHF), set
  `normalize=affine` with frozen calibration bounds and an explicit
  `hv_reference_point` / `ideal`.
* For the LS/MORLHF Pareto front, sweep `amo_strategy.scalarize_config.weights`
  over a fixed simplex design and reuse that weight set for `gdpo_weighted`,
  `mgda`/`gapo` (via `algorithm.amo_objective_weights`) and P-GAPO.

## Tests

CPU-only unit tests (no model/GPU/server) cover every numeric core:

```bash
python -m pytest tests/amo/test_baselines.py -v   # or the plain-python runner in the file
```
