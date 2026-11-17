"""Gymnasium environment: one learning agent (player 0) against opponents.

Opponents come from a pool of specs (see ``splixrl.opponents``). Each episode
the env draws ``n_opponents`` specs from the pool and builds fresh opponent
objects, so bots never share state between episodes or between seats. Pass
``options={"opponents": [...]}`` to ``reset`` to choose them for one episode
(the evaluation code does this so every algorithm meets the same opponents).

Episode end (default "duel" rules):
* terminated: the agent died, or every opponent is dead;
* truncated: ``max_steps`` reached.

``info["outcome"]`` on the last step is "win", "loss" or "draw":
win  = agent alive and all opponents dead, or alive at the step limit with
       strictly more territory than every living opponent;
loss = agent dead (unless everyone died on the same step: draw), or behind
       on territory at the step limit; draw = tied.

"Territory" rules, closer to the real game, where the goal is to take the map
without dying: ``respawn=True`` brings dead opponents back after
``respawn_delay`` steps (so killing them never ends the game), and
``target_frac`` ends the episode with a win once the agent owns that share of
the board. Then win = reached the target, loss = died, "timeout" = neither
by ``max_steps``. With ``respawn`` but no target, the step-limit territory
comparison above decides.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from .game import N_ACTIONS, N_ACTIONS_WITH_PAUSE, PAUSE, STRAIGHT, SplixGame
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
        respawn: bool = False,
        respawn_delay: int = 10,
        target_frac: float | None = None,
        coarse_factor: int = 0,
        visible_radius: int | None = None,
        threat_features: bool = False,
        minimap: bool = False,
        enemy_heading: bool = False,
        size_features: bool = False,
        reclaim_features: bool = False,
        escape_features: bool = False,
        allow_pause: bool = False,
        max_pause: int = 20,
        size_range: tuple[int, int] | None = None,
        n_opponents_range: tuple[int, int] | None = None,
        lag_range: tuple[float, float] | None = None,
    ):
        """``size_range`` / ``n_opponents_range`` / ``lag_range`` (inclusive)
        draw a new board size, opponent count and lag for every episode, so an
        agent sees many layouts and crowd sizes (local observations only: the
        input shape doesn't depend on the board size). The fixed values are
        used until the first reset and when no range is given."""
        if not opponents and (n_opponents or n_opponents_range):
            raise ValueError("opponent pool is empty")
        if obs_mode != "local" and size_range:
            raise ValueError("size_range needs local observations")
        self.size_range = tuple(size_range) if size_range else None
        self.n_opponents_range = tuple(n_opponents_range) if n_opponents_range else None
        self.lag_range = tuple(lag_range) if lag_range else None
        self.spawn_radius = spawn_radius
        self.size = size
        self.pool = list(opponents)
        self.n_opponents = n_opponents
        self.max_steps = max_steps
        self.obs_config = ObsConfig(
            mode=obs_mode, view_radius=view_radius, coarse_factor=coarse_factor,
            visible_radius=visible_radius, threat_features=threat_features, minimap=minimap,
            enemy_heading=enemy_heading, size_features=size_features, reclaim_features=reclaim_features,
            escape_features=escape_features, pause_feature=allow_pause,
        )
        # Pausing (action 3), as the real server allows. max_pause caps how many
        # steps in a row the agent may stay paused (then it goes straight), so
        # "stand still forever" can't become a way to avoid all risk.
        self.allow_pause = allow_pause
        self.max_pause = max_pause
        self._paused_for = 0
        self.reward_config = reward if isinstance(reward, RewardConfig) else RewardConfig.from_dict(reward)
        self.render_mode = render_mode
        # Network lag, as on the real server: with this probability a turn the
        # agent asks for is applied one tile late (it goes straight first).
        # 0 for the main comparison; > 0 to train agents for the live server.
        self.lag_prob = lag_prob
        self._delayed: int | None = None
        self.respawn = respawn
        self.respawn_delay = respawn_delay
        self.target_frac = target_frac
        self.game = SplixGame(
            size=size, n_players=1 + n_opponents, spawn_radius=spawn_radius, max_steps=max_steps
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS_WITH_PAUSE if allow_pause else N_ACTIONS)
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
            coarse_factor=self.obs_config.coarse_factor,
            visible_radius=self.obs_config.visible_radius,
            threat_features=self.obs_config.threat_features,
            minimap=self.obs_config.minimap,
            enemy_heading=self.obs_config.enemy_heading,
            size_features=self.obs_config.size_features,
            reclaim_features=self.obs_config.reclaim_features,
            escape_features=self.obs_config.escape_features,
            allow_pause=self.allow_pause,
            max_pause=self.max_pause,
            reward=self.reward_config.to_dict(),
            lag_prob=self.lag_prob,
            respawn=self.respawn,
            respawn_delay=self.respawn_delay,
            target_frac=self.target_frac,
            size_range=self.size_range,
            n_opponents_range=self.n_opponents_range,
            lag_range=self.lag_range,
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
        if self.size_range or self.n_opponents_range:
            size = int(rng.integers(self.size_range[0], self.size_range[1] + 1)) if self.size_range else self.size
            n = int(rng.integers(self.n_opponents_range[0], self.n_opponents_range[1] + 1)) if self.n_opponents_range else self.n_opponents
            if size != self.game.size or n + 1 != self.game.n_players:
                self.size, self.n_opponents = size, n
                self.game = SplixGame(size=size, n_players=1 + n, spawn_radius=self.spawn_radius, max_steps=self.max_steps)
        if self.lag_range:
            self.lag_prob = float(rng.uniform(*self.lag_range))
        while True:
            try:
                self.game.reset(rng)
                break
            except ValueError:  # too many players for this board: drop one and try again
                if not self.n_opponents_range or self.n_opponents <= 1:
                    raise
                self.n_opponents -= 1
                self.game = SplixGame(size=self.size, n_players=1 + self.n_opponents,
                                      spawn_radius=self.spawn_radius, max_steps=self.max_steps)
        specs = (options or {}).get("opponents")
        if specs is None:
            idx = rng.integers(len(self.pool), size=self.n_opponents)
            specs = [self.pool[i] for i in idx]
        if len(specs) != self.n_opponents:
            if len(set(map(str, specs))) == 1:  # one spec for every seat
                specs = [specs[0]] * self.n_opponents
            else:
                raise ValueError(f"expected {self.n_opponents} opponents, got {len(specs)}")
        self._specs = list(specs)
        self._dead_at: dict[int, int] = {}
        self.opponent_respawns = 0
        self.opponents = [make_opponent(s) for s in specs]
        self.opponent_names = [opponent_label(s) for s in specs]
        for i, opp in enumerate(self.opponents, start=1):
            opp.reset(self.game, i, np.random.default_rng(int(rng.integers(2**63))))
        self._phi = self._potential()
        self._delayed = None
        self._paused_for = 0
        self._total_captured = 0
        self._peak_territory = self.game.territory(0)
        return self._obs(), self._info()

    def step(self, action):
        g = self.game
        rc = self.reward_config
        action, lagged = int(action), False
        if action == PAUSE:
            if not self.allow_pause:
                raise ValueError("pausing is off for this env (allow_pause=False)")
            self._paused_for += 1
            if self._paused_for > self.max_pause:
                action = STRAIGHT
        if action != PAUSE:
            self._paused_for = 0
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
        reached = (
            not agent_dead and self.target_frac is not None
            and g.territory(0) >= self.target_frac * g.size * g.size
        )
        if self.respawn and not agent_dead and not reached:
            self._respawn_opponents()
        ends_on_kills = not self.respawn and self.n_opponents > 0
        terminated = agent_dead or reached or (opps_dead and ends_on_kills)
        truncated = not terminated and g.t >= self.max_steps
        self._peak_territory = max(self._peak_territory, g.territory(0))

        phi = 0.0 if terminated else self._potential()
        r += rc.gamma * phi - self._phi
        self._phi = phi

        info = self._info(captured=captured)
        info["lagged"] = lagged
        if terminated or truncated:
            info["outcome"] = self._outcome(agent_dead, opps_dead and not self.respawn, reached)
            info["won"] = info["outcome"] == "win"
            if info["won"]:
                r += rc.win
            if rc.rank_bonus and not agent_dead:
                mine = g.territory(0)
                others = [g.territory(p.pid) for p in g.players[1:]]
                share = sum(o < mine for o in others) / max(1, len(others))
                info["rank_share"] = share
                r += rc.rank_bonus * share
        return self._obs(), float(r), terminated, truncated, info

    def _respawn_opponents(self) -> None:
        g = self.game
        for i in range(1, len(g.players)):
            if g.players[i].alive:
                continue
            self._dead_at.setdefault(i, g.t)
            if g.t - self._dead_at[i] >= self.respawn_delay and g.respawn(i, self.np_random):
                del self._dead_at[i]
                self.opponents[i - 1] = make_opponent(self._specs[i - 1])  # fresh bot for the new life
                self.opponents[i - 1].reset(g, i, np.random.default_rng(int(self.np_random.integers(2**63))))
                self.opponent_respawns += 1

    def _outcome(self, agent_dead: bool, opps_dead: bool, reached: bool = False) -> str:
        g = self.game
        if reached:
            return "win"
        if self.target_frac is not None and not agent_dead:
            return "win" if opps_dead else "timeout"
        if agent_dead:
            # the episode stops as soon as all opponents die, so both being dead
            # here means everyone died on the same step
            return "draw" if opps_dead else "loss"
        if opps_dead:
            return "win"
        mine = g.territory(0)
        alive = [g.territory(p.pid) for p in g.players[1:] if p.alive]
        if not alive:
            return "win"
        best = max(alive)
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
            "opp_respawns": getattr(self, "opponent_respawns", 0),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb(self.game)
        if self.render_mode == "ansi":
            return self.game.ansi()
        return None
