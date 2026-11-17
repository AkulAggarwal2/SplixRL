"""Gymnasium environment: one learning agent (player 0) against opponents.

Opponents come from a pool of specs (see ``splixrl.opponents``). Each episode
the env draws ``n_opponents`` specs from the pool and builds fresh opponent
objects, so bots never share state between episodes or between seats. Pass
``options={"opponents": [...]}`` to ``reset`` to choose them for one episode
(the evaluation code does this so every algorithm meets the same opponents).

Episode end:
* terminated: the agent died, or every opponent is dead;
* truncated: ``max_steps`` reached.

``info["outcome"]`` on the last step is "win", "loss" or "draw":
win  = agent alive and all opponents dead, or alive at the step limit with
       strictly more territory than every living opponent;
loss = agent dead (unless everyone died on the same step: draw), or behind
       on territory at the step limit; draw = tied.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from .game import N_ACTIONS, STRAIGHT, SplixGame
from .obs import ObsConfig, build_obs
from .opponents import Opponent, make_opponent, opponent_label
from .render import render_rgb
from .rewards import RewardConfig


class SplixEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        size: int = 20,
        opponents: Sequence[Any] = ("loop",),
        n_opponents: int = 1,
        max_steps: int = 500,
        spawn_radius: int = 2,
        obs_mode: str = "local",
        view_radius: int = 7,
        reward: RewardConfig | dict | None = None,
        render_mode: str | None = None,
        lag_prob: float = 0.0,
    ):
        if not opponents and n_opponents:
            raise ValueError("opponent pool is empty")
        self.size = size
        self.pool = list(opponents)
        self.n_opponents = n_opponents
        self.max_steps = max_steps
        self.obs_config = ObsConfig(mode=obs_mode, view_radius=view_radius)
        self.reward_config = reward if isinstance(reward, RewardConfig) else RewardConfig.from_dict(reward)
        self.render_mode = render_mode
        # Network lag, as on the real server: with this probability a turn the
        # agent asks for is applied one tile late (it goes straight first).
        # 0 for the main comparison; > 0 to train agents for the live server.
        self.lag_prob = lag_prob
        self._delayed: int | None = None
        self.game = SplixGame(
            size=size, n_players=1 + n_opponents, spawn_radius=spawn_radius, max_steps=max_steps
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = self.obs_config.space(size)
        self.opponents: list[Opponent] = []
        self.opponent_names: list[str] = []

    # kwargs that rebuild an identical env (used by evaluation / checkpoints)
    def config(self) -> dict:
        return dict(
            size=self.size,
            opponents=[s for s in self.pool if isinstance(s, str)],
            n_opponents=self.n_opponents,
            max_steps=self.max_steps,
            spawn_radius=self.game.spawn_radius,
            obs_mode=self.obs_config.mode,
            view_radius=self.obs_config.view_radius,
            reward=self.reward_config.to_dict(),
            lag_prob=self.lag_prob,
        )

    def _obs(self) -> dict[str, np.ndarray]:
        return build_obs(self.game, 0, self.obs_config)

    def _potential(self) -> float:
        rc = self.reward_config
        if rc.shaping == 0.0 or not self.game.players[0].alive:
            return 0.0
        return rc.shaping * rc.capture_per_cell * len(self.game.players[0].trail)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.game.reset(rng)
        specs = (options or {}).get("opponents")
        if specs is None:
            idx = rng.integers(len(self.pool), size=self.n_opponents)
            specs = [self.pool[i] for i in idx]
        if len(specs) != self.n_opponents:
            raise ValueError(f"expected {self.n_opponents} opponents, got {len(specs)}")
        self.opponents = [make_opponent(s) for s in specs]
        self.opponent_names = [opponent_label(s) for s in specs]
        for i, opp in enumerate(self.opponents, start=1):
            opp.reset(self.game, i, np.random.default_rng(int(rng.integers(2**63))))
        self._phi = self._potential()
        self._delayed = None
        self._total_captured = 0
        self._peak_territory = self.game.territory(0)
        return self._obs(), self._info()

    def step(self, action):
        g = self.game
        rc = self.reward_config
        action, lagged = int(action), False
        if self._delayed is not None:  # last step's turn finally lands; this step's choice is lost
            action, self._delayed, lagged = self._delayed, None, True
        elif action != STRAIGHT and self.lag_prob > 0 and self.np_random.random() < self.lag_prob:
            self._delayed, action, lagged = action, STRAIGHT, True
        actions = [action]
        self._territory_before = g.territory(0)
        for i, opp in enumerate(self.opponents, start=1):
            actions.append(int(opp.act(g, i)) if g.players[i].alive else 0)
        ev = g.step(actions, self.np_random)

        captured = ev.captured.get(0, 0)
        self._total_captured += captured
        r = rc.capture_per_cell * captured
        r -= rc.lost_per_cell * ev.lost.get(0, 0)
        r += rc.kill * ev.kills.get(0, 0)
        r += rc.step

        agent_dead = not g.players[0].alive
        opps_dead = all(not p.alive for p in g.players[1:])
        if agent_dead:
            r += rc.death - rc.death_per_cell * self._territory_before
        terminated = agent_dead or (opps_dead and self.n_opponents > 0)
        truncated = not terminated and g.t >= self.max_steps
        self._peak_territory = max(self._peak_territory, g.territory(0))

        phi = 0.0 if terminated else self._potential()
        r += rc.gamma * phi - self._phi
        self._phi = phi

        info = self._info(captured=captured)
        info["lagged"] = lagged
        if terminated or truncated:
            info["outcome"] = self._outcome(agent_dead, opps_dead)
            info["won"] = info["outcome"] == "win"
            if info["won"]:
                r += rc.win
        return self._obs(), float(r), terminated, truncated, info

    def _outcome(self, agent_dead: bool, opps_dead: bool) -> str:
        g = self.game
        if agent_dead:
            # the episode stops as soon as all opponents die, so both being dead
            # here means everyone died on the same step
            return "draw" if opps_dead else "loss"
        if opps_dead:
            return "win"
        mine = g.territory(0)
        best = max(g.territory(p.pid) for p in g.players[1:] if p.alive)
        return "win" if mine > best else "draw" if mine == best else "loss"

    def _info(self, captured: int = 0) -> dict:
        g = self.game
        me = g.players[0]
        return {
            "steps": g.t,
            "territory": g.territory(0),
            "territory_frac": g.territory(0) / (g.size * g.size),
            "peak_territory": self._peak_territory,
            "total_captured": self._total_captured,
            "captured": captured,
            "trail_len": len(me.trail),
            "kills": me.kills,
            "alive": me.alive,
            "death_cause": me.death_cause,
            "opponents": list(self.opponent_names),
            "opp_territory": [g.territory(p.pid) for p in g.players[1:]],
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb(self.game)
        if self.render_mode == "ansi":
            return self.game.ansi()
        return None
