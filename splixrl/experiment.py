"""The exact settings of the main comparison (Round 0), in one place.

Every algorithm (Dyna, Double DQN, PPO) must train and be evaluated with these,
or the results can't be compared (Misha's feedback: only agents that trained
and were evaluated against exactly the same opponents can be compared)::

    from splixrl.experiment import ROUND0, round0_env_kwargs
    env = SplixEnv(**round0_env_kwargs())
    evaluator = TrainingEvaluator(run_dir, round0_env_kwargs(), ROUND0["eval_opponents"],
                                  ROUND0["eval_episodes"])
    ...
    if step % ROUND0["eval_every"] == 0:
        evaluator(agent, step)

These are the settings of the reported Dyna runs (``runs/round0_reward2``):
``python -m splixrl.train_dyna ... --reward-preset v2`` with the defaults.
"""

from __future__ import annotations

from .opponents import DEFAULT_POOL
from .rewards import RewardConfig

ROUND0 = {
    "env": {
        "size": 20,
        "opponents": list(DEFAULT_POOL),  # random_safe, loop, chaser: one drawn per episode
        "n_opponents": 1,
        "max_steps": 500,
        "spawn_radius": 2,
        "obs_mode": "local",
        "view_radius": 7,
        "lag_prob": 0.0,
    },
    "reward_preset": "v2",  # +0.1 per cell, +3 per kill, -3 on death, -0.1 per cell lost or held at death
    "gamma": 0.99,
    "steps": 200_000,  # real env steps per run
    "seeds": (0, 1, 2),
    "eval_opponents": list(DEFAULT_POOL),
    "eval_episodes": 20,  # per opponent, during training
    "eval_every": 10_000,
    "final_eval_episodes": 100,  # per opponent, for the results table
}


def round0_env_kwargs() -> dict:
    """``SplixEnv(**round0_env_kwargs())`` is the Round 0 environment."""
    reward = RewardConfig.preset(ROUND0["reward_preset"])
    reward.gamma = ROUND0["gamma"]
    return {**ROUND0["env"], "opponents": list(ROUND0["env"]["opponents"]), "reward": reward.to_dict()}
