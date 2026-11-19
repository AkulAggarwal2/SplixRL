"""Double DQN -- Akul's part. Only the interface is here.

What the rest of the project needs from this file:

1. ``DDQNAgent(Agent)`` with ``act``, ``save`` and ``load`` (see ``base.py``).
   ``save`` must ``torch.save`` a dict with ``"algo": "ddqn"`` and
   ``"obs_config": self.obs_config.to_dict()``; then ``load_agent(path)``,
   ``"frozen:<path>"`` opponents, ``python -m splixrl.evaluate`` and
   ``python -m splixrl.watch`` all work with your checkpoints.
2. Use the exact Round 0 settings from ``splixrl/experiment.py``
   (``round0_env_kwargs()``, ``ROUND0``; reward v2), see docs/TEAM_SETUP.md.
3. A training loop (e.g. ``splixrl/train_ddqn.py``) that builds
   ``SplixEnv(**env_kwargs)`` with the same env settings as the other
   algorithms and calls ``TrainingEvaluator(run_dir, env_kwargs,
   DEFAULT_POOL, episodes)(agent, step)`` every N steps, so its eval.csv
   plots next to the others with ``python -m splixrl.plot``.

Observations are dicts: ``obs["grid"]`` float32 (6, 15, 15) and
``obs["vec"]`` float32 (11,). Actions: 0 straight, 1 left, 2 right.
``splixrl.agents.base.obs_to_torch`` batches observations for a network.

Experiment 2 (optional, after the per-tile version works): the same algorithm
over options -- 17 plans (``N_OPTIONS``: rectangular loops, home, reclaim,
attack, ambush, adaptive loops, steal-edge) carried out by scripted controllers (``splixrl/options.py``).

* Env: ``splixrl.options.make_options_env(gamma=..., **env_kwargs)``, a normal
  Gymnasium env with ``Discrete(N_OPTIONS)`` actions. One ``step`` runs a whole
  plan and returns the discounted reward collected during it.
* Plans take different numbers of tiles. ``info["option_disc"]`` is gamma ** k
  for a k-tile plan: use it instead of gamma when bootstrapping
  (target = r + option_disc * (1 - done) * Q(s', a')).
* Observations at decision points are the same dicts as above (use the same
  observation settings as the Dyna options run, see README "Experiment 2").
* Save ``"options": {"safety": 6.0, "lag_slack": 30.0}`` in the checkpoint;
  ``load_agent`` then marks the agent as an options agent and every tool plays
  it plan by plan. ``act`` returns an option index in ``range(N_OPTIONS)``.
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
