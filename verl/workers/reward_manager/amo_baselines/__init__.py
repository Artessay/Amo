# Copyright 2025 Rihong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-objective alignment *baseline* reward managers for Amo.

This subpackage collects reward managers that reproduce published (or
widely-used) multi-objective LLM-alignment baselines so they can be compared
against HVPO under an identical online-RL pipeline (same base model, data,
rollout count, reward functions, KL and training budget). Only the
multi-objective *credit* differs.

The managers here deliberately reuse :class:`AmoVanillaRewardManager` for the
per-sample multi-objective scoring loop (see :mod:`amo_baselines.common`) and
only change how the per-objective score vector is turned into the scalar
per-response reward that the (GRPO-form) advantage estimator consumes.

Registered managers:

* ``amo_scalarize`` -- fixed linear scalarization (LS/MORLHF weight sweep) and
  (augmented) Tchebycheff scalarization.
* ``amo_adaptive``  -- adaptive-weight / constrained baselines that hold state
  across steps: Lagrangian (Safe-RLHF style), Fair-and-Stable mirror-descent
  reward composition, CTWA (covariance-targeted weight adaptation) and
  hypervolume-guided dynamic reward weighting.
* ``amo_pareto``    -- Pareto-selection credit: NSGA-II-style
  (rank + crowding) and SMS-EMOA-style (exclusive-HV) response credit.
"""

# Importing the modules triggers the @register(...) decorators.
from verl.workers.reward_manager.amo_baselines.scalarize import AmoScalarizeRewardManager
from verl.workers.reward_manager.amo_baselines.adaptive import AmoAdaptiveRewardManager
from verl.workers.reward_manager.amo_baselines.pareto import AmoParetoRewardManager

__all__ = [
    "AmoScalarizeRewardManager",
    "AmoAdaptiveRewardManager",
    "AmoParetoRewardManager",
]
