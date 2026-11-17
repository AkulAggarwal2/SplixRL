"""Agents and the checkpoint format every algorithm shares.

A checkpoint is a ``torch.save`` dict with at least:

    {"algo": "dyna" | "ddqn" | "ppo", "obs_config": ObsConfig.to_dict(), ...}

``load_agent(path)`` reads ``algo`` and calls that class's ``load``. Anything
loaded this way can be evaluated, watched, or frozen as an opponent
(``"frozen:path"``), whichever algorithm made it.
"""

from __future__ import annotations

import importlib

import torch

from .base import Agent

REGISTRY = {
    "dyna": "splixrl.agents.dyna:DynaAgent",
    "ddqn": "splixrl.agents.ddqn:DDQNAgent",  # Akul
    "ppo": "splixrl.agents.ppo:PPOAgent",  # Sai
}


def agent_class(algo: str) -> type[Agent]:
    if algo not in REGISTRY:
        raise ValueError(f"unknown algo {algo!r}; known: {sorted(REGISTRY)}")
    module, _, name = REGISTRY[algo].partition(":")
    return getattr(importlib.import_module(module), name)


def load_agent(path: str, device: str = "cpu") -> Agent:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return agent_class(ckpt["algo"]).load(path, device=device)


__all__ = ["Agent", "REGISTRY", "agent_class", "load_agent"]
