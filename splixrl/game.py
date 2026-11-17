"""Rules engine for a simplified Splix.io-style territory game.

Pure numpy, no Gymnasium. The board is a square grid. Each player owns some
cells (territory). A player outside its own territory leaves a trail. When it
gets back to its own territory, the trail and every cell the trail encloses
become its territory (including cells taken from other players).

Rules, resolved simultaneously every step:

1. Every living player picks a relative action (straight, turn left, turn
   right). Reversing is impossible by construction.
2. Leaving the board kills the player ("wall").
3. Two heads entering the same cell: a player standing on its own territory
   survives, everyone else there dies ("head_on"). Two players swapping cells
   is resolved the same way.
4. Moving onto any trail cell kills the trail's owner: your own trail kills
   you ("own_trail"); someone else's trail kills them ("trail_cut") and you get
   the kill.
5. Dead players lose their territory and trail; their cells become empty.
6. Survivors move. On their own territory with a trail, they capture; off
   their territory, the cell is added to their trail.
7. A living player left with no territory dies ("territory_lost"); the player
   who captured its last cell gets the kill.

Trail collisions are checked against trails as they were before the step, so
a trail includes the cell a player is standing on.

Pausing (action PAUSE, only offered when the env allows it), as on the real
server: the player stops where it is. Its trail stays where it is, so with a
trail out it is a sitting target. Any other action resumes, relative to the
heading it had before pausing, so it can't reverse into its own trail. A
paused player doesn't take part in head-on collisions; anyone moving onto
its trail still kills it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_fill_holes

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
# (row delta, col delta) for each heading
DIRS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)

STRAIGHT, TURN_LEFT, TURN_RIGHT, PAUSE = 0, 1, 2, 3
N_ACTIONS = 3  # without pausing; envs with allow_pause use N_ACTIONS_WITH_PAUSE
N_ACTIONS_WITH_PAUSE = 4
ACTION_NAMES = ("straight", "left", "right", "pause")

EMPTY = -1

DEATH_CAUSES = ("wall", "head_on", "own_trail", "trail_cut", "territory_lost")


def turn(heading: int, action: int) -> int:
    """New absolute heading after a relative action."""
    if action == TURN_LEFT:
        return (heading + 3) % 4
    if action == TURN_RIGHT:
        return (heading + 1) % 4
    return heading


@dataclass
class Player:
    pid: int
    pos: tuple[int, int]
    heading: int
    alive: bool = True
    trail: list[tuple[int, int]] = field(default_factory=list)
    kills: int = 0
    death_cause: str | None = None
    killed_by: int | None = None
    steps_alive: int = 0
    paused: bool = False


@dataclass
class StepEvents:
    """What happened during one step, keyed by player id."""

    captured: dict[int, int] = field(default_factory=dict)  # cells gained
    lost: dict[int, int] = field(default_factory=dict)  # cells taken by others
    deaths: dict[int, str] = field(default_factory=dict)  # cause of death
    kills: dict[int, int] = field(default_factory=dict)  # kills scored


class SplixGame:
    def __init__(
        self,
        size: int = 20,
        n_players: int = 2,
        spawn_radius: int = 2,  # 5x5 start square, like the real game
        min_spawn_gap: int = 3,
        max_steps: int | None = None,
    ):
        if n_players < 1:
            raise ValueError("need at least one player")
        if n_players > 127:
            raise ValueError("at most 127 players (int8 board)")
        self.size = size
        self.n_players = n_players
        self.spawn_radius = spawn_radius
        self.min_spawn_gap = min_spawn_gap
        self.max_steps = max_steps
        self.owner = np.full((size, size), EMPTY, dtype=np.int8)
        self.trail = np.full((size, size), EMPTY, dtype=np.int8)
        self.players: list[Player] = []
        self.t = 0

    # ------------------------------------------------------------------ setup
    def reset(self, rng: np.random.Generator) -> None:
        self.owner.fill(EMPTY)
        self.trail.fill(EMPTY)
        self.t = 0
        self.players = []
        s = self.spawn_radius
        for pid, (r, c) in enumerate(self._spawn_centers(rng)):
            self.owner[r - s : r + s + 1, c - s : c + s + 1] = pid
            self.players.append(Player(pid=pid, pos=(r, c), heading=int(rng.integers(4))))

    def _spawn_centers(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        s = self.spawn_radius
        lo, hi = s + 1, self.size - s - 2  # keep one empty ring between spawn and wall
        if hi < lo:
            raise ValueError(f"board size {self.size} too small for spawn radius {s}")
        need = 2 * s + 1 + self.min_spawn_gap  # Chebyshev distance between centres
        for _ in range(2000):
            centers: list[tuple[int, int]] = []
            for _ in range(self.n_players):
                for _ in range(200):
                    r, c = (int(x) for x in rng.integers(lo, hi + 1, size=2))
                    if all(max(abs(r - r2), abs(c - c2)) >= need for r2, c2 in centers):
                        centers.append((r, c))
                        break
            if len(centers) == self.n_players:
                return centers
        raise ValueError(
            f"could not place {self.n_players} players on a {self.size}x{self.size} board"
        )

    def respawn(self, pid: int, rng: np.random.Generator, tries: int = 300) -> bool:
        """Bring a dead player back with a fresh start square on empty ground,
        away from other heads (like joining the real game). False if no room."""
        if self.players[pid].alive:
            return False
        s = self.spawn_radius
        lo, hi = s + 1, self.size - s - 2
        if hi < lo:
            return False
        need = 2 * s + 1 + self.min_spawn_gap
        free = (self.owner == EMPTY) & (self.trail == EMPTY)
        heads = [p.pos for p in self.players if p.alive]
        for _ in range(tries):
            r, c = (int(x) for x in rng.integers(lo, hi + 1, size=2))
            if not free[r - s : r + s + 1, c - s : c + s + 1].all():
                continue
            if any(max(abs(r - r2), abs(c - c2)) < need for r2, c2 in heads):
                continue
            self.owner[r - s : r + s + 1, c - s : c + s + 1] = pid
            self.players[pid] = Player(pid=pid, pos=(r, c), heading=int(rng.integers(4)))
            return True
        return False

    # ---------------------------------------------------------------- queries
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def next_cell(self, pid: int, action: int) -> tuple[tuple[int, int], int]:
        """Cell a player would move to with this action, and its new heading.
        Pausing stays put."""
        p = self.players[pid]
        if action == PAUSE:
            return p.pos, p.heading
        h = turn(p.heading, action)
        return (p.pos[0] + int(DIRS[h, 0]), p.pos[1] + int(DIRS[h, 1])), h

    def is_safe(self, pid: int, action: int) -> bool:
        """True if the action doesn't kill the player outright (wall / own trail).

        Ignores what opponents might do this step.
        """
        if action == PAUSE:
            return True
        (r, c), _ = self.next_cell(pid, action)
        return self.in_bounds(r, c) and int(self.trail[r, c]) != pid

    def territory(self, pid: int) -> int:
        return int(np.count_nonzero(self.owner == pid))

    def alive_ids(self) -> list[int]:
        return [p.pid for p in self.players if p.alive]

    # ------------------------------------------------------------------- step
    def step(self, actions, rng: np.random.Generator | None = None) -> StepEvents:
        """Advance one step. ``actions[pid]`` is ignored for dead players."""
        ev = StepEvents()
        alive = [p for p in self.players if p.alive]
        new_pos: dict[int, tuple[int, int]] = {}
        new_head: dict[int, int] = {}
        for p in alive:
            a = int(actions[p.pid])
            if a == PAUSE:
                p.paused = True
                continue
            p.paused = False
            new_pos[p.pid], new_head[p.pid] = self.next_cell(p.pid, a)
        movers = [p for p in alive if p.pid in new_pos]

        deaths: dict[int, tuple[str, int | None]] = {}

        def kill(victim: int, cause: str, killer: int | None = None) -> None:
            if victim not in deaths:
                deaths[victim] = (cause, killer)

        # walls
        for pid, (r, c) in new_pos.items():
            if not self.in_bounds(r, c):
                kill(pid, "wall")

        # head-on: several heads entering one cell
        by_cell: dict[tuple[int, int], list[int]] = {}
        for pid, rc in new_pos.items():
            if pid not in deaths:
                by_cell.setdefault(rc, []).append(pid)
        for rc, pids in by_cell.items():
            if len(pids) > 1:
                safe = [q for q in pids if int(self.owner[rc]) == q]
                for q in pids:
                    if q not in safe:
                        kill(q, "head_on", safe[0] if safe else None)

        # head-on: two players swapping cells
        for i, p in enumerate(movers):
            for q in movers[i + 1 :]:
                if new_pos[p.pid] == q.pos and new_pos[q.pid] == p.pos:
                    p_safe = int(self.owner[q.pos]) == p.pid
                    q_safe = int(self.owner[p.pos]) == q.pid
                    if not p_safe:
                        kill(p.pid, "head_on", q.pid if q_safe else None)
                    if not q_safe:
                        kill(q.pid, "head_on", p.pid if p_safe else None)

        # trail hits (trails as they were before this step)
        for pid, (r, c) in new_pos.items():
            if not self.in_bounds(r, c):
                continue
            j = int(self.trail[r, c])
            if j == EMPTY:
                continue
            if j == pid:
                kill(pid, "own_trail")
            else:
                kill(j, "trail_cut", pid)

        for victim, (cause, killer) in deaths.items():
            self._eliminate(victim, cause, killer, ev)

        # survivors move, extend trails, capture
        survivors = [p for p in movers if p.alive]
        for p in alive:
            if p.alive and p.paused:
                p.steps_alive += 1
        if rng is not None and len(survivors) > 1:
            survivors = [survivors[i] for i in rng.permutation(len(survivors))]
        last_taker: dict[int, int] = {}
        for p in survivors:
            p.pos, p.heading = new_pos[p.pid], new_head[p.pid]
            p.steps_alive += 1
            if int(self.owner[p.pos]) == p.pid:
                if p.trail:
                    self._capture(p, ev, last_taker)
            else:
                self.trail[p.pos] = p.pid
                p.trail.append(p.pos)

        # anyone whose whole territory was taken is out
        for p in self.players:
            if p.alive and self.territory(p.pid) == 0:
                self._eliminate(p.pid, "territory_lost", last_taker.get(p.pid), ev)

        self.t += 1
        return ev

    def _capture(self, p: Player, ev: StepEvents, last_taker: dict[int, int]) -> None:
        mask = (self.owner == p.pid) | (self.trail == p.pid)
        gained = binary_fill_holes(mask) & (self.owner != p.pid)
        victims, counts = np.unique(self.owner[gained], return_counts=True)
        for v, n in zip(victims.tolist(), counts.tolist()):
            if v != EMPTY:
                ev.lost[v] = ev.lost.get(v, 0) + n
                last_taker[v] = p.pid
        self.owner[gained] = p.pid
        self.trail[self.trail == p.pid] = EMPTY
        p.trail.clear()
        ev.captured[p.pid] = ev.captured.get(p.pid, 0) + int(np.count_nonzero(gained))

    def _eliminate(self, pid: int, cause: str, killer: int | None, ev: StepEvents) -> None:
        p = self.players[pid]
        if not p.alive:
            return
        p.alive = False
        p.death_cause = cause
        p.killed_by = killer
        self.owner[self.owner == pid] = EMPTY
        self.trail[self.trail == pid] = EMPTY
        p.trail.clear()
        ev.deaths[pid] = cause
        if killer is not None and killer != pid:
            self.players[killer].kills += 1
            ev.kills[killer] = ev.kills.get(killer, 0) + 1

    # ---------------------------------------------------------------- display
    def ansi(self) -> str:
        """Text board: territory a/b/c..., trail A/B/C..., heads 0/1/2..."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        rows = []
        heads = {p.pos: p.pid for p in self.players if p.alive}
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) in heads:
                    row.append(str(heads[(r, c)] % 10))
                elif self.trail[r, c] != EMPTY:
                    row.append(letters[self.trail[r, c] % 26].upper())
                elif self.owner[r, c] != EMPTY:
                    row.append(letters[self.owner[r, c] % 26])
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)
