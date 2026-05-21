"""GRPO advantage function with the ISP-KGR sanity clamp.

slime already does:
- group-wise mean/std reward normalization (slime/ray/rollout.py:611-636
  `_post_process_rewards`, gated on `--rewards-normalization` +
  `--grpo-std-normalization`)
- token-broadcast advantage = scalar reward repeated per response token
  (`slime/utils/ppo_utils.py:201 get_grpo_returns`)
- asymmetric PPO clip via `--eps-clip` / `--eps-clip-high`
  (`slime/utils/ppo_utils.py:125 compute_policy_loss`)

The ONLY behaviour we need on top is the [-10, 10] advantage clamp from
ISP-KGR/grpo.py:423. So this custom advantage function reuses slime's
`get_grpo_returns` and just adds the clamp.
"""

from __future__ import annotations

from argparse import Namespace

import torch

from slime.utils.ppo_utils import get_grpo_returns
from slime.utils.types import RolloutBatch


ADV_CLAMP_MIN = -10.0
ADV_CLAMP_MAX = 10.0


def dapo_grpo_advantage(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute GRPO advantages with the ISP-KGR clamp, in-place.

    Mirrors the GRPO branch in `compute_advantages_and_returns`
    (slime/backends/megatron_utils/loss.py:632-636), then clamps.
    """
    rewards = rollout_data.get("rewards")
    kl = rollout_data.get("kl")
    assert rewards is not None and kl is not None, (
        "rollout_data must contain 'rewards' and 'kl' before advantage_fn runs"
    )

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
    rewards_t = torch.clamp(rewards_t, ADV_CLAMP_MIN, ADV_CLAMP_MAX)

    returns = get_grpo_returns(rewards_t, kl)
    rollout_data["returns"] = returns
    rollout_data["advantages"] = [r for r in returns]
