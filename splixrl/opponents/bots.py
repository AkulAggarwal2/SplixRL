"""Hand-written opponents. They never learn, so they are the fixed yardstick
every algorithm trains and is evaluated against."""

from __future__ import annotations

import numpy as np

from ..game import N_ACTIONS, STRAIGHT, TURN_LEFT, TURN_RIGHT, SplixGame
from .base import Opponent, bfs_first_action, safe_actions


class RandomBot(Opponent):
    """Uniformly random actions. With ``safe=True`` it never drives into a
    wall or its own trail when it has another choice."""

    def __init__(self, safe: bool = False):
        self.safe = safe
        self.name = "random_safe" if safe else "random"

    def act(self, game: SplixGame, pid: int) -> int:
        if self.safe:
            options = safe_actions(game, pid)
            if options:
                return int(self.rng.choice(options))
        return int(self.rng.integers(N_ACTIONS))


class LoopBot(Opponent):
    """Captures land with small rectangular loops and heads home when threatened.

    From its territory it takes the shortest way out (away from enemy heads),
    then draws a rectangle: ``out`` cells straight, a turn, ``side`` cells, a
    turn, then the shortest safe path home (which closes the loop). Loop sizes
    are random within the given ranges. It cuts the loop short and goes home
    when an opponent's head comes within ``danger_dist`` of its head or trail,
    or when its trail reaches ``max_trail``. Paths outside its own land keep
    ``keep_away`` cells from enemy heads when they can, so it doesn't drive
    its trail past an opponent that is waiting at home.
    """

    name = "loop"

    def __init__(
        self,
        min_out: int = 2,
        max_out: int = 5,
        min_side: int = 2,
        max_side: int = 6,
        danger_dist: int = 4,
        max_trail: int = 24,
        wander: float = 0.2,
        keep_away: int = 2,
    ):
        self.min_out, self.max_out = min_out, max_out
        self.min_side, self.max_side = min_side, max_side
        self.danger_dist = danger_dist
        self.max_trail = max_trail
        self.wander = wander  # chance of a random turn while inside its territory
        self.keep_away = keep_away

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.phase = "home"  # home -> out -> return -> home

    # -- helpers -----------------------------------------------------------
    def _order(self) -> list[int]:
        rest = [TURN_LEFT, TURN_RIGHT]
        self.rng.shuffle(rest)
        return [STRAIGHT, *rest]

    def _enemy_dist(self, game: SplixGame, pid: int) -> np.ndarray:
        """Manhattan distance from every cell to the nearest enemy head (large if none)."""
        n = game.size
        heads = [q.pos for q in game.players if q.alive and q.pid != pid]
        if not heads:
            return np.full((n, n), 10 * n)
        rr, cc = np.mgrid[0:n, 0:n]
        return np.min([np.abs(rr - r) + np.abs(cc - c) for r, c in heads], axis=0)

    def _route(self, game: SplixGame, pid: int, goal: np.ndarray, blocked: np.ndarray, near: np.ndarray) -> int | None:
        """BFS that avoids cells near enemy heads (outside own land), falling back to any path."""
        own = game.owner == pid
        a = bfs_first_action(game, pid, goal & ~near, blocked | (near & ~own), self._order())
        if a is None:
            a = bfs_first_action(game, pid, goal, blocked, self._order())
        return a

    def _danger(self, game: SplixGame, pid: int) -> bool:
        p = game.players[pid]
        heads = np.array([q.pos for q in game.players if q.alive and q.pid != pid]).reshape(-1, 2)
        if not len(heads):
            return False
        mine = np.array([p.pos, *p.trail])
        return int(np.abs(mine[:, None, :] - heads[None, :, :]).sum(2).min()) <= self.danger_dist

    def go_home(self, game: SplixGame, pid: int) -> int:
        own = game.owner == pid
        blocked = game.trail == pid
        near = self._enemy_dist(game, pid) <= self.keep_away
        a = self._route(game, pid, own, blocked, near)
        if a is None:  # boxed in by its own trail: survive as long as possible
            options = safe_actions(game, pid)
            return int(self.rng.choice(options)) if options else STRAIGHT
        return a

    def _leave_home(self, game: SplixGame, pid: int) -> int:
        options = safe_actions(game, pid)
        if options and self.rng.random() < self.wander:
            return int(self.rng.choice(options))
        outside = (game.owner != pid) & (game.trail < 0)
        # leave on a side where no enemy is close, so the loop is drawn away from them
        near = self._enemy_dist(game, pid) <= self.danger_dist + self.keep_away
        a = self._route(game, pid, outside, np.zeros_like(outside), near)
        if a is None or not game.is_safe(pid, a):
            return int(self.rng.choice(options)) if options else STRAIGHT
        return a

    # -- policy ------------------------------------------------------------
    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        if game.owner[p.pos] == pid:
            self.phase = "home"
            return self._leave_home(game, pid)

        if self.phase == "home":  # first step outside: plan the rectangle
            self.phase = "out"
            self.leg = 0
            self.leg_left = int(self.rng.integers(self.min_out, self.max_out + 1)) - 1
            self.side = int(self.rng.integers(self.min_side, self.max_side + 1))
            self.turn_dir = TURN_LEFT if self.rng.random() < 0.5 else TURN_RIGHT

        if self.phase == "out" and (len(p.trail) >= self.max_trail or self._danger(game, pid)):
            self.phase = "return"
        if self.phase == "return":
            return self.go_home(game, pid)

        if self.leg_left > 0:
            a = STRAIGHT
            self.leg_left -= 1
        elif self.leg == 0:
            a = self.turn_dir
            self.leg = 1
            self.leg_left = self.side - 1
        else:
            self.phase = "return"
            return self.go_home(game, pid)
        (r, c), _ = game.next_cell(pid, a)
        if not game.is_safe(pid, a) or self._enemy_dist(game, pid)[r, c] <= self.keep_away:
            self.phase = "return"
            return self.go_home(game, pid)
        return a


class ChaserBot(LoopBot):
    """Aggressive: hunts any opponent trail within ``chase_radius`` and
    otherwise plays like LoopBot. Ignores danger while hunting."""

    name = "chaser"

    def __init__(self, chase_radius: int = 10, chase_max_trail: int = 30, **kw):
        super().__init__(**kw)
        self.chase_radius = chase_radius
        self.chase_max_trail = chase_max_trail

    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        enemy_trail = (game.trail >= 0) & (game.trail != pid)
        if enemy_trail.any() and len(p.trail) < self.chase_max_trail:
            cells = np.argwhere(enemy_trail)
            if int(np.abs(cells - np.array(p.pos)).sum(1).min()) <= self.chase_radius:
                a = bfs_first_action(game, pid, enemy_trail, game.trail == pid, self._order())
                if a is not None and game.is_safe(pid, a):
                    if game.owner[p.pos] != pid:
                        self.phase = "return"  # after the chase, go home
                    return a
        return super().act(game, pid)


class InvaderBot(LoopBot):
    """Steals land: heads for the nearest enemy territory within ``reach``,
    drives ``depth`` tiles into it, then goes home, so the loop closes around
    the stolen cells. Otherwise plays like LoopBot. Gives agents practice at
    noticing an invader, cutting its trail from the safety of their own land,
    and taking the land back."""

    name = "invader"

    def __init__(self, reach: int = 14, depth: int = 4, max_trail: int = 40, **kw):
        super().__init__(max_trail=max_trail, keep_away=1, danger_dist=2, **kw)
        self.reach = reach
        self.depth = depth

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.raid = 0  # tiles driven into enemy land on the current raid

    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        enemy_land = (game.owner >= 0) & (game.owner != pid)
        if game.owner[p.pos] == pid:
            self.raid = 0
        if enemy_land[p.pos]:
            self.raid += 1
        if self.raid >= self.depth or len(p.trail) >= self.max_trail or (self.raid and self._danger(game, pid)):
            self.phase = "return"
            return self.go_home(game, pid)
        if enemy_land.any() and self.raid < self.depth:
            cells = np.argwhere(enemy_land)
            if int(np.abs(cells - np.array(p.pos)).sum(1).min()) <= self.reach:
                a = bfs_first_action(game, pid, enemy_land, game.trail == pid, self._order())
                if a is not None and game.is_safe(pid, a):
                    if game.owner[p.pos] != pid:
                        self.phase = "out"
                        self.leg, self.leg_left = 1, 0
                    return a
        return super().act(game, pid)
