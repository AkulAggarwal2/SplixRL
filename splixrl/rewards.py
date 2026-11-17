"""Reward settings. Every algorithm uses the same RewardConfig for a comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardConfig:
    capture_per_cell: float = 0.1  # per cell gained when a loop closes (incl. stolen cells)
    lost_per_cell: float = 0.0  # per own cell taken by an opponent's capture
    kill: float = 3.0  # per opponent eliminated by this agent
    death: float = -3.0  # once, when the agent dies
    # per cell the agent owned when it died (all of it is lost). With this equal
    # to capture_per_cell (and lost_per_cell), the return adds up to the
    # territory held at the end, like the real game's score, so big risky
    # loops are no longer free. 0 = flat death penalty only (reward v1).
    death_per_cell: float = 0.0
    win: float = 0.0  # once, if the episode ends in a win
    step: float = 0.0  # every step (negative = time pressure)
    # Potential-based shaping (Ng, Harada & Russell 1999). The potential is
    # shaping * capture_per_cell * trail_length: it pays a little for
    # extending a trail and takes it back when the loop closes or the agent
    # dies, which leaves the optimal policy unchanged. 0 disables it.
    shaping: float = 0.0
    gamma: float = 0.99  # must equal the learner's discount for shaping to be exact

    @classmethod
    def preset(cls, name: str) -> "RewardConfig":
        """Named settings.

        v1:    +0.1 per cell, +3 per kill, -3 on death.
        v2:    v1, plus dying or having land stolen costs 0.1 per cell.
        score: the real game's score -- 1 point per tile, 500 per kill (see the
               in-game "Kills: 1, Blocks: 3748, Score: 4248") -- scaled by 0.01,
               and dying loses the land you hold. Kills are worth 50 cells of
               land here instead of 30 under v1/v2.
        """
        presets = {
            "v1": {},
            "v2": {"death_per_cell": 0.1, "lost_per_cell": 0.1},
            "score": {"capture_per_cell": 0.01, "kill": 5.0, "death": 0.0,
                      "death_per_cell": 0.01, "lost_per_cell": 0.01},
        }
        if name not in presets:
            raise ValueError(f"unknown reward preset {name!r}; choose from {sorted(presets)}")
        return cls(**presets[name])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        return cls(**(d or {}))
