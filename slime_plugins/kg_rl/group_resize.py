"""Resize a variable-sized GRPO group to a fixed K.

Ports the cyclic-copy + reward-stratified-sampling logic from
ISP-KGR/grpo.py:119-158 (`GRPODataset._process_item`), generalized to operate
on any list of objects paired with a scalar reward.

Why this exists: slime requires `len(group) == n_samples_per_prompt`, but the
tree rollout produces a variable number of kept children per parent (after
dedup). We pad small groups by cyclic repeat (to preserve reward distribution)
and shrink large groups by reward-stratified round-robin (to keep intra-group
variance, which GRPO needs as learning signal).
"""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def resize_to_k(
    items: list[T],
    rewards: list[float],
    k: int,
    rng: random.Random | None = None,
) -> tuple[list[T], list[float]]:
    """Resize (items, rewards) of arbitrary length to exactly k.

    - len < k: cyclic-repeat-then-truncate. Preserves the reward distribution
      better than random oversampling, so GRPO's mean/std stay stable.
    - len > k: bucket by reward, round-robin pick from each bucket so distinct
      reward values are preserved (variance > 0 is what gives GRPO a signal),
      then shuffle to break round-robin order.
    - len == k: return as-is.
    """
    assert len(items) == len(rewards), "items and rewards length mismatch"
    assert len(items) >= 1, "cannot resize an empty group"
    assert k >= 1

    rng = rng or random

    n = len(items)
    if n == k:
        return list(items), list(rewards)

    if n < k:
        multiplier = (k // n) + 1
        idx = (list(range(n)) * multiplier)[:k]
        return [items[i] for i in idx], [rewards[i] for i in idx]

    # n > k: reward-stratified shrink
    reward_buckets: dict[float, list[int]] = {}
    for i, r in enumerate(rewards):
        reward_buckets.setdefault(r, []).append(i)

    unique_rewards = sorted(reward_buckets.keys())
    for r in unique_rewards:
        rng.shuffle(reward_buckets[r])

    selected: list[int] = []
    while len(selected) < k:
        added = False
        for r in unique_rewards:
            if len(selected) >= k:
                break
            if reward_buckets[r]:
                selected.append(reward_buckets[r].pop())
                added = True
        if not added:
            break

    rng.shuffle(selected)
    return [items[i] for i in selected], [rewards[i] for i in selected]
