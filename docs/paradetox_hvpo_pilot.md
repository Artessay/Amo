# HVPO vs GRPO: ParaDetox pilot

## Choice

- Benchmark: ParaDetox English text detoxification.
- Base model: Qwen2.5-1.5B-Instruct.
- Objectives: non-toxicity (STA), source-semantic similarity (SIM), and fluency
  (FL), all oriented so higher is better.

STA and SIM provide a direct response-level trade-off: copying more of the toxic
source preserves semantics but also preserves toxic expressions. FL prevents
degenerate rewrites. The 1.5B model and three cached classifiers keep the
iteration loop short.

## Controlled pilot

GRPO uses equal-weight linear scalarization of the three raw rewards. HVPO uses
running objective normalization followed by intra-group exclusive-hypervolume
credit (with its distance fallback). Both then use the same GRPO group
standardization, so this compares scalar credit construction rather than a
different policy-loss formula. All other settings are shared: seed 42, LoRA rank 32,
learning rate 1e-5, KL coefficient 0.001, eight training responses per prompt,
and four held-out responses per prompt.

Start the three offline reward servers on a spare GPU:

```bash
CUDA_VISIBLE_DEVICES=2 bash recipe/amo_detox/start_all_pilot.sh
```

Run the methods sequentially on another GPU. A one-step smoke test is obtained
with `STEPS=1 VAL_SAMPLES=8`; the default is a ten-step, 64-prompt pilot.

```bash
CUDA_VISIBLE_DEVICES=0 RUN_TAG=pilot10 bash scripts/paradetox/run_pilot.sh grpo
CUDA_VISIBLE_DEVICES=0 RUN_TAG=pilot10 bash scripts/paradetox/run_pilot.sh hvpo
```

Use a new `RUN_TAG` for every rerun. The launcher rejects an existing method
directory so stale validation steps cannot be mistaken for the new final step.

Compare final held-out response sets:

```bash
python3 scripts/paradetox/analyze_pilot.py \
  --grpo-dir results/ParaDetox/pilot10/qwen2.5-1.5b_grpo_seed42/validation \
  --hvpo-dir results/ParaDetox/pilot10/qwen2.5-1.5b_hvpo_seed42/validation
```

## Seed-42 pilot result

The paired step-0 controls are identical (64 prompts, four responses per
prompt, response-set HV 0.701920). After ten updates:

| Metric | GRPO | HVPO | HVPO - GRPO |
| --- | ---: | ---: | ---: |
| Response-set HV | 0.685903 | 0.665246 | -0.020656 |
| Joint product | 0.507026 | 0.492682 | -0.014344 |
| Equal-weight linear reward | 0.809601 | 0.802074 | -0.007527 |

The prompt-paired 95% bootstrap interval for the HV difference is
[-0.065595, 0.024667]. Thus this ten-step, one-seed pilot does not show an HVPO
advantage. The HVPO signal itself did not collapse: across 640 training
responses its contribution mean/std are 0.009294/0.037502, and all 80 prompt
groups have at least one positive contribution. Full machine-readable results,
including step-0-to-10 changes and difference-in-differences, are in
`results/ParaDetox/pilot10/summary.json`.

The analyzer refuses to compare mismatched latest steps, verifies paired prompt
groups, de-duplicates objective points for Pareto diagnostics, and reports a
10,000-sample prompt-paired bootstrap interval. The primary pilot metric is mean
per-prompt response-set HV with a fixed origin reference point. Also inspect each raw objective, mean joint product,
non-dominated front size, incomparable-pair rate, and STA-SIM correlation. A
publishable comparison should repeat at least three seeds and evaluate with an
independent toxicity model to detect reward hacking. For multi-seed runs, change
`SEED` (actor and rollout randomness) while leaving `DATA_SEED=42` fixed so the
training prompt order and held-out prompt subset do not change between seeds.
