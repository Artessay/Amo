# Experiment Ledger

Append-only record of every training/eval run. Generated from `ledger.jsonl` by `playground/log_experiment.py` — do not edit by hand.

| timestamp | method | model | dataset | final_step | reward | safe_helpfulness | safe_harmlessness | hybrid_rewards | hv_contribution | distance_penalty | num_best_HV | total_problems | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-13 21:48 | grpo | qwen2.5-3b | PKU-SafeRLHF | 150 | 6.4621 | 5.8665 | 7.0577 |  |  |  |  |  | 1 epoch (150 steps), linear weighted-sum reward, lr1e-5 LoRA r32 |
| 2026-07-13 21:48 | hvpo | qwen2.5-3b | PKU-SafeRLHF | 150 | 6.4359 | 5.8360 | 7.0359 | 6.4359 | 0.0000 | -0.2190 |  |  | 1 epoch (150 steps), exclusive-HV reward, static ref [0,0], chebyshev fallback |
| 2026-07-13 21:48 | hvpo |  | moo_suite_22probs | None |  |  |  |  |  |  | 10.0000 | 22.0000 | GA-harness synthetic MOO (ZDT/DTLZ/WFG); HVPO wins 10/22 on HV, beats nsga3(5)/moead(4) |
