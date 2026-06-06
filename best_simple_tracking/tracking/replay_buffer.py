"""Replay buffer for off-policy critic update.

NOT in original RL_Active project. This module is NEW (added by us) to address
the high-variance, sample-inefficient online TD learning of the original.
"""

from __future__ import annotations

import torch
from collections import deque
from typing import NamedTuple


class Transition(NamedTuple):
    obs: torch.Tensor       # (obs_dim,)
    action: torch.Tensor    # (action_dim,) per robot
    reward_heads: torch.Tensor  # (num_heads,)  e.g. (persistence, info_gain)
    next_obs: torch.Tensor
    done: bool
    weights: torch.Tensor   # (num_heads,) — dynamic head weights at this step


class ReplayBuffer:
    """Simple FIFO replay buffer (no prioritization)."""

    def __init__(self, capacity: int = 50_000):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self._buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        n = min(batch_size, len(self._buffer))
        idx = torch.randint(0, len(self._buffer), (n,))
        return [self._buffer[i.item()] for i in idx]

    def __len__(self):
        return len(self._buffer)

    def is_ready(self, min_size: int = 256) -> bool:
        return len(self._buffer) >= min_size
