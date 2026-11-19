from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ..obs import ObsConfig


class Agent(ABC):
    """What evaluation, watching and freezing need from a trained agent.

    Training loops are each algorithm's own business; this is only the
    interface to a finished (or mid-training) policy.
    """

    algo: str = "base"
    obs_config: ObsConfig

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> int:
        """One action (0 straight, 1 left, 2 right) for one observation."""

    @abstractmethod
    def save(self, path: str) -> None:
        """torch.save a dict that includes "algo" and "obs_config"."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = "cpu") -> "Agent":
        ...


def obs_to_torch(obs: dict[str, np.ndarray], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Single observation or a batch (leading dim) -> (grid, vec) float tensors with a batch dim."""
    grid = torch.as_tensor(obs["grid"], dtype=torch.float32, device=device)
    vec = torch.as_tensor(obs["vec"], dtype=torch.float32, device=device)
    if grid.dim() == 3:
        grid, vec = grid.unsqueeze(0), vec.unsqueeze(0)
    return grid, vec
