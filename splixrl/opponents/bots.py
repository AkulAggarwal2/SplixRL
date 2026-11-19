"""Hand-written opponents. They never learn, so they are the fixed yardstick
every algorithm trains and is evaluated against."""

from __future__ import annotations

import numpy as np

from ..game import DIRS, N_ACTIONS, PAUSE, STRAIGHT, TURN_LEFT, TURN_RIGHT, SplixGame
from .base import Opponent, bfs_first_action, safe_actions


def _manhattan_core(n, heads):
    out = np.empty((n, n), dtype=np.int64)
    for r in range(n):
        for c in range(n):
            best = 1 << 30
            for k in range(heads.shape[0]):
                d = abs(r - heads[k, 0]) + abs(c - heads[k, 1])
                if d < best:
                    best = d
            out[r, c] = best
    return out


try:
    from numba import njit

    _manhattan_to_nearest = njit(cache=True)(_manhattan_core)
except ImportError:  # pragma: no cover
    def _manhattan_to_nearest(n, heads):
        rr, cc = np.mgrid[0:n, 0:n]
        return np.min([np.abs(rr - r) + np.abs(cc - c) for r, c in heads], axis=0)


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
        jitter: bool = False,
    ):
        self.min_out, self.max_out = min_out, max_out
        self.min_side, self.max_side = min_side, max_side
        self.danger_dist = danger_dist
        self.max_trail = max_trail
        self.wander = wander  # chance of a random turn while inside its territory
        self.keep_away = keep_away
        # jitter: every life draws its own loop sizes and caution around these
        # settings, so a learner can't memorise one bot's habits
        self.jitter = jitter
        self._base = (min_out, max_out, min_side, max_side, danger_dist, max_trail, keep_away)

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.phase = "home"  # home -> out -> return -> home
        if self.jitter:
            mo, xo, ms, xs, dd, mt, ka = self._base
            scale = float(rng.uniform(0.6, 1.8))  # timid ... bold
            self.min_out, self.max_out = max(1, round(mo * scale)), max(2, round(xo * scale))
            self.min_side, self.max_side = max(1, round(ms * scale)), max(2, round(xs * scale))
            self.danger_dist = max(1, round(dd / scale))
            self.max_trail = max(8, round(mt * scale))
            self.keep_away = int(rng.integers(max(0, ka - 1), ka + 2))

    # -- helpers -----------------------------------------------------------
    def _order(self) -> list[int]:
        rest = [TURN_LEFT, TURN_RIGHT]
        self.rng.shuffle(rest)
        return [STRAIGHT, *rest]

    def _enemy_dist(self, game: SplixGame, pid: int) -> np.ndarray:
        """Manhattan distance from every cell to the nearest enemy head (large if none).
        Cached per step and seat: several bot rules ask for it on the same step."""
        cache = getattr(game, "_step_cache", None)
        if cache is None or cache.get("t") != game.t:
            cache = game._step_cache = {"t": game.t}
        key = ("enemy_dist", pid)
        if key not in cache:
            cache[key] = self._enemy_dist_uncached(game, pid)
        return cache[key]

    def _enemy_dist_uncached(self, game: SplixGame, pid: int) -> np.ndarray:
        n = game.size
        heads = np.array([q.pos for q in game.players if q.alive and q.pid != pid], dtype=np.int64).reshape(-1, 2)
        if not len(heads):
            return np.full((n, n), 10 * n)
        return _manhattan_to_nearest(n, heads)

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

    def __init__(self, chase_radius: int = 10, chase_max_trail: int = 30,
                 vision: int | None = None, reaction: int = 0, **kw):
        """``vision``: only trails within this many tiles (square, like the
        screen) are seen; None sees the whole board. ``reaction``: a trail
        must have been in sight this many steps in a row before the chase starts."""
        super().__init__(**kw)
        self.chase_radius = chase_radius
        self.chase_max_trail = chase_max_trail
        self.vision = vision
        self.reaction = reaction
        self.seen_for = 0

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.seen_for = 0

    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        enemy_trail = (game.trail >= 0) & (game.trail != pid)
        if self.vision is not None and enemy_trail.any():
            r, c, v = p.pos[0], p.pos[1], self.vision
            sight = np.zeros_like(enemy_trail)
            sight[max(0, r - v) : r + v + 1, max(0, c - v) : c + v + 1] = True
            enemy_trail &= sight
        if enemy_trail.any() and len(p.trail) < self.chase_max_trail:
            cells = np.argwhere(enemy_trail)
            in_range = int(np.abs(cells - np.array(p.pos)).sum(1).min()) <= self.chase_radius
            self.seen_for = self.seen_for + 1 if in_range else 0
            if in_range and self.seen_for > self.reaction:
                a = bfs_first_action(game, pid, enemy_trail, game.trail == pid, self._order())
                if a is not None and game.is_safe(pid, a):
                    if game.owner[p.pos] != pid:
                        self.phase = "return"  # after the chase, go home
                    return a
        else:
            self.seen_for = 0
        return super().act(game, pid)


class InvaderBot(LoopBot):
    """Steals land: heads for the nearest enemy territory within ``reach``,
    drives ``depth`` tiles into it, then goes home, so the loop closes around
    the stolen cells. Otherwise plays like LoopBot. Gives agents practice at
    noticing an invader, cutting its trail from the safety of their own land,
    and taking the land back."""

    name = "invader"

    def __init__(self, reach: int = 14, depth: int = 4, max_trail: int = 40, **kw):
        kw.setdefault("keep_away", 1)
        kw.setdefault("danger_dist", 3)
        super().__init__(max_trail=max_trail, **kw)
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
        # like a sensible player: abort the raid as soon as an enemy head gets close
        if p.trail and (self.raid >= self.depth or len(p.trail) >= self.max_trail or self._danger(game, pid)):
            self.phase = "return"
            return self.go_home(game, pid)
        near_enemy = int(self._enemy_dist(game, pid)[p.pos]) <= self.danger_dist + 3
        if enemy_land.any() and self.raid < self.depth and not near_enemy:
            cells = np.argwhere(enemy_land)
            if int(np.abs(cells - np.array(p.pos)).sum(1).min()) <= self.reach:
                a = bfs_first_action(game, pid, enemy_land, game.trail == pid, self._order())
                if a is not None and game.is_safe(pid, a):
                    if game.owner[p.pos] != pid:
                        self.phase = "out"
                        self.leg, self.leg_left = 1, 0
                    return a
        return super().act(game, pid)


def plan_cut(game: SplixGame, pid: int, reach: int = 30, min_gain: int = 4, ok=None):
    """Best straight cut through enemy land enclosed by mine, as
    (start cell on my land, direction, cells cut off, line cells, owner pid), or None.
    The line runs from my land to my land, at least 2 rows/columns from the
    owner's head; what it cuts off is the patch on the far side from the head.
    ``ok(line cells, owner)`` can veto lines (e.g. ones the owner could reach first)."""
    from scipy.ndimage import binary_fill_holes

    own = game.owner == pid
    if not own.any():
        return None
    closed = binary_fill_holes(own)
    me = np.array(game.players[pid].pos)
    best, best_score = None, min_gain - 1
    for q in game.players:
        if not q.alive or q.pid == pid:
            continue
        patch = closed & (game.owner == q.pid)
        if not patch.any():
            continue
        hr, hc = q.pos
        cells = np.argwhere(patch)
        for axis in (0, 1):  # 0: a horizontal line (fixed row), 1: vertical (fixed column)
            for line in np.unique(cells[:, axis]):
                if abs(int(line) - (hr if axis == 0 else hc)) <= 1:
                    continue  # too close to its head
                span = cells[cells[:, axis] == line][:, 1 - axis]
                a, b = int(span.min()), int(span.max())
                path = [(int(line), k) if axis == 0 else (k, int(line)) for k in range(a, b + 1)]
                ends = [(int(line), a - 1), (int(line), b + 1)] if axis == 0 else [(a - 1, int(line)), (b + 1, int(line))]
                n = game.size
                if not all(0 <= x < n and 0 <= y < n and own[x, y] for x, y in ends):
                    continue
                if any(own[x, y] or (x, y) == (hr, hc) for x, y in path):
                    continue
                # what the line cuts off: patch cells on the far side from the head
                side = cells[:, axis] - int(line)
                head_side = (hr if axis == 0 else hc) - int(line)
                score = int(np.count_nonzero(np.sign(side) == -np.sign(head_side)))
                start = min(ends, key=lambda e: abs(e[0] - me[0]) + abs(e[1] - me[1]))
                if abs(start[0] - me[0]) + abs(start[1] - me[1]) > reach:
                    continue
                other = ends[1] if start == ends[0] else ends[0]
                h = [i for i in range(4) if (int(np.sign(other[0] - start[0])), int(np.sign(other[1] - start[1]))) == (int(DIRS[i, 0]), int(DIRS[i, 1]))][0]
                if score > best_score and (ok is None or ok(path, q)):
                    best, best_score = (start, h, score, path, q.pid), score
    return best


class CutterBot(LoopBot):
    """Cuts enemy land that sits inside or against its own: drives a straight
    line across the enemy patch, from its land to its land, on the side away
    from the owner's head. Under the real server's rule the piece the owner
    is standing in is safe, and everything the line cuts off is captured --
    what people do to a player who spawned or got stuck inside their land.
    Otherwise plays like LoopBot."""

    name = "cutter"

    def __init__(self, reach: int = 30, **kw):
        super().__init__(**kw)
        self.reach = reach

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.cut = None  # (start cell on my land, direction across)

    def _plan(self, game: SplixGame, pid: int):
        plan = plan_cut(game, pid, self.reach)
        return None if plan is None else plan[:2]

    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        own = game.owner == pid
        if self.cut is not None and p.trail:  # crossing: straight on until back on my land
            if game.is_safe(pid, STRAIGHT):
                return STRAIGHT
            self.cut = None
            return self.go_home(game, pid)
        if own[p.pos] and not p.trail:
            if self.cut is None or self.rng.random() < 0.02:
                self.cut = self._plan(game, pid)
            if self.cut is not None:
                start, h = self.cut
                T = (start[0] + int(DIRS[h, 0]), start[1] + int(DIRS[h, 1]))
                goal = np.zeros_like(own)
                goal[T] = True
                blocked = ~own
                for side in ((h + 1) % 4, (h + 3) % 4):  # step out from the start cell itself, facing across
                    x, y = T[0] + int(DIRS[side, 0]), T[1] + int(DIRS[side, 1])
                    if 0 <= x < game.size and 0 <= y < game.size:
                        blocked[x, y] = True
                blocked[T] = False
                a = bfs_first_action(game, pid, goal, blocked, self._order())
                if a is not None and game.is_safe(pid, a):
                    return a
                self.cut = None
        elif self.cut is not None and not p.trail:
            self.cut = None
        return super().act(game, pid)



class ProwlerBot(ChaserBot):
    """Hunts the way strong players do on a big map: from home it often heads
    for the biggest territory on the map (players know where land is from the
    minimap), waits paused near its edge, and pounces on any trail that shows
    up on its screen (it sees 25 tiles and reacts after a few steps, like the
    human-like bot). Then it goes home and picks again. Travelling and waiting
    leave its own trail out, so it can be killed too."""

    name = "prowler"

    def __init__(self, prowl_prob: float = 0.5, lurk_steps: int = 40, max_trip: int = 90, **kw):
        kw = {"chase_radius": 40, "chase_max_trail": 120, "vision": 25, "reaction": 3, **kw}
        super().__init__(**kw)
        self.prowl_prob = prowl_prob
        self.lurk_steps = lurk_steps
        self.max_trip = max_trip

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        super().reset(game, pid, rng)
        self.target = None  # cells to lurk next to
        self.lurk_left = 0

    def _pick_target(self, game: SplixGame, pid: int):
        sizes = [(int(np.count_nonzero(game.owner == q.pid)), q.pid) for q in game.players
                 if q.alive and q.pid != pid]
        sizes = [x for x in sizes if x[0] > 0]
        if not sizes:
            return None
        _, victim = max(sizes)
        land = game.owner == victim
        # the ring 3 tiles outside its land, nearest part to me
        from scipy.ndimage import binary_dilation

        ring = binary_dilation(land, iterations=3) & ~binary_dilation(land, iterations=2) & (game.owner != pid)
        return ring if ring.any() else None

    def act(self, game: SplixGame, pid: int) -> int:
        p = game.players[pid]
        own = game.owner == pid
        # chasing a visible trail beats everything (ChaserBot's rule, with vision and reaction)
        enemy_trail = (game.trail >= 0) & (game.trail != pid)
        r, c, v = p.pos[0], p.pos[1], self.vision
        sight = np.zeros_like(enemy_trail)
        sight[max(0, r - v) : r + v + 1, max(0, c - v) : c + v + 1] = True
        if (enemy_trail & sight).any() and len(p.trail) < self.chase_max_trail:
            a = super().act(game, pid)
            if self.seen_for > self.reaction:
                return a
        if own[p.pos] and not p.trail:
            if self.target is None and self.rng.random() < self.prowl_prob / 20:
                self.target, self.lurk_left = self._pick_target(game, pid), self.lurk_steps
            if self.target is None:
                return LoopBot.act(self, game, pid)
        if self.target is not None:
            if len(p.trail) > self.max_trip or self.lurk_left <= 0:
                self.target = None
                return self.go_home(game, pid)
            if self.target[p.pos] or int(np.abs(np.argwhere(self.target) - np.array(p.pos)).sum(1).min()) <= 1:
                self.lurk_left -= 1
                return PAUSE if game.is_safe(pid, PAUSE) else STRAIGHT
            a = bfs_first_action(game, pid, self.target, game.trail == pid, self._order())
            if a is not None and game.is_safe(pid, a):
                return a
            self.target = None
        return LoopBot.act(self, game, pid)
