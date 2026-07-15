# PKU-SafeRLHF multi-objective baselines

Per-objective test means and dominated hypervolume (HV, ref = origin) for each model x method. Higher is better on all columns. Objectives are the Safe-RLHF reward-model scores `safe_helpfulness` and `safe_harmlessness`. Empty cells are not-yet-run.

> Controlled comparison: every method shares the identical base model, data, rollout `n`, KL, optimizer and training-token budget; only the multi-objective credit rule differs. Scale-sensitive methods (LS, Tchebycheff, Lagrangian, Dynamic-HV) use the frozen calibration in `safe_calibration.json`.


## Qwen2.5-1.5B-Instruct

| Method | Helpfulness | Harmlessness | Hypervolume | #prompts |
|---|---|---|---|---|
| GRPO (equal-weight) | - | - | - | - |
| GDPO | - | - | - | - |
| HVPO | - | - | - | - |
| LS / MORLHF | - | - | - | - |
| Tchebycheff | - | - | - | - |
| GDPO (weighted) | - | - | - | - |
| RVPO (soft-min) | - | - | - | - |
| MGDA | - | - | - | - |
| GAPO | - | - | - | - |
| Lagrangian (Safe-RLHF) | - | - | - | - |
| Fair-and-Stable | - | - | - | - |
| CTWA | - | - | - | - |
| Dynamic-HV weighting | - | - | - | - |
| NSGA-II-style credit | - | - | - | - |
| SMS-EMOA-style credit | - | - | - | - |

## Qwen2.5-3B-Instruct

| Method | Helpfulness | Harmlessness | Hypervolume | #prompts |
|---|---|---|---|---|
| GRPO (equal-weight) | 5.743 | 6.992 | 43.1396 | 8211 |
| GDPO | - | - | - | - |
| HVPO | 5.698 | 6.966 | 43.3818 | 8211 |
| LS / MORLHF | - | - | - | - |
| Tchebycheff | - | - | - | - |
| GDPO (weighted) | - | - | - | - |
| RVPO (soft-min) | - | - | - | - |
| MGDA | - | - | - | - |
| GAPO | - | - | - | - |
| Lagrangian (Safe-RLHF) | - | - | - | - |
| Fair-and-Stable | - | - | - | - |
| CTWA | - | - | - | - |
| Dynamic-HV weighting | - | - | - | - |
| NSGA-II-style credit | - | - | - | - |
| SMS-EMOA-style credit | - | - | - | - |

## Llama-3.2-3B-Instruct

| Method | Helpfulness | Harmlessness | Hypervolume | #prompts |
|---|---|---|---|---|
| GRPO (equal-weight) | - | - | - | - |
| GDPO | - | - | - | - |
| HVPO | - | - | - | - |
| LS / MORLHF | - | - | - | - |
| Tchebycheff | - | - | - | - |
| GDPO (weighted) | - | - | - | - |
| RVPO (soft-min) | - | - | - | - |
| MGDA | - | - | - | - |
| GAPO | - | - | - | - |
| Lagrangian (Safe-RLHF) | - | - | - | - |
| Fair-and-Stable | - | - | - | - |
| CTWA | - | - | - | - |
| Dynamic-HV weighting | - | - | - | - |
| NSGA-II-style credit | - | - | - | - |
| SMS-EMOA-style credit | - | - | - | - |
