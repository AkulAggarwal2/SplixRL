"""Double DQN -- Akul's part. Only the interface is here.

What the rest of the project needs from this file:

1. ``DDQNAgent(Agent)`` with ``act``, ``save`` and ``load`` (see ``base.py``).
   ``save`` must ``torch.save`` a dict with ``"algo": "ddqn"`` and
   ``"obs_config": self.obs_config.to_dict()``; then ``load_agent(path)``,
   ``"frozen:<path>"`` opponents, ``python -m splixrl.evaluate`` and
   ``python -m splixrl.watch`` all work with your checkpoints.
2. A training loop (e.g. ``splixrl/train_ddqn.py``) that builds
   ``SplixEnv(**env_kwargs)`` with the same env settings as the other
   algorithms and calls ``TrainingEvaluator(run_dir, env_kwargs,
   DEFAULT_POOL, episodes)(agent, step)`` every N steps, so its eval.csv
   plots next to the others with ``python -m splixrl.plot``.

Observations are dicts: ``obs["grid"]`` float32 (6, 15, 15) and
``obs["vec"]`` float32 (11,). Actions: 0 straight, 1 left, 2 right.
``splixrl.agents.base.obs_to_torch`` batches observations for a network.
"""

from __future__ import annotations

from ..obs import ObsConfig
from .base import Agent


class DDQNAgent(Agent):
    algo = "ddqn"

    def __init__(self, obs_config: ObsConfig, *args, **kwargs):
        self.obs_config = obs_config
        raise NotImplementedError("Double DQN is Akul's part of the project")

    def act(self, obs, deterministic: bool = True) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DDQNAgent":
        raise NotImplementedError
