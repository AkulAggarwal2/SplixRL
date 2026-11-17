from __future__ import annotations

import numpy as np

from ..game import SplixGame
from ..obs import build_obs
from .base import Opponent

_CACHE: dict[tuple[str, str], object] = {}


class FrozenOpponent(Opponent):
    """A saved RL agent playing as an opponent. Its weights are never updated.

    It sees the board from its own seat, using the observation settings it
    was trained with (stored in its checkpoint).
    """

    def __init__(self, path: str, deterministic: bool = True, device: str = "cpu"):
        from ..agents import load_agent

        key = (path, device)
        if key not in _CACHE:  # load each checkpoint once per process
            _CACHE[key] = load_agent(path, device=device)
        self.agent = _CACHE[key]
        self.deterministic = deterministic
        self.name = f"frozen:{path}"

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)

    def act(self, game: SplixGame, pid: int) -> int:
        obs = build_obs(game, pid, self.agent.obs_config)
        return int(self.agent.act(obs, deterministic=self.deterministic))
