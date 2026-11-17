"""SplixRL: a Splix.io-style territory game for comparing RL algorithms."""

from gymnasium.envs.registration import register

from .env import SplixEnv
from .game import SplixGame
from .obs import ObsConfig, build_obs
from .rewards import RewardConfig

register(id="SplixRL-v0", entry_point="splixrl.env:SplixEnv")

__all__ = ["SplixEnv", "SplixGame", "ObsConfig", "RewardConfig", "build_obs"]
