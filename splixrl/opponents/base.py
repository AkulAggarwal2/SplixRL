from __future__ import annotations

from collections import deque

import numpy as np

from ..game import DIRS, N_ACTIONS, SplixGame, turn


class Opponent:
    """A fixed (non-learning) policy that controls one seat in the game.

    Opponents read the game state directly. One object controls one player
    for one episode; the env builds fresh objects every episode.
    """

    name = "base"

    def reset(self, game: SplixGame, pid: int, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, game: SplixGame, pid: int) -> int:
        raise NotImplementedError


def safe_actions(game: SplixGame, pid: int) -> list[int]:
    return [a for a in range(N_ACTIONS) if game.is_safe(pid, a)]


try:  # compiled BFS: bots path-find every step, and pure Python was ~40% of env time
    from numba import njit
except ImportError:  # pragma: no cover - numba is optional
    njit = None

_DR = np.array([-1, 0, 1, 0], dtype=np.int64)  # same order as game.DIRS (up, right, down, left)
_DC = np.array([0, 1, 0, -1], dtype=np.int64)


def _bfs_core(r0, c0, heading, goal, blocked, order, dr, dc):
    n0, n1 = goal.shape
    visited = np.zeros((n0, n1), dtype=np.bool_)
    visited[r0, c0] = True
    cap = n0 * n1
    qr = np.empty(cap, dtype=np.int64)
    qc = np.empty(cap, dtype=np.int64)
    qa = np.empty(cap, dtype=np.int64)
    head = 0
    tail = 0
    for k in range(order.shape[0]):
        a = order[k]
        h = heading
        if a == 1:
            h = (heading + 3) % 4
        elif a == 2:
            h = (heading + 1) % 4
        r = r0 + dr[h]
        c = c0 + dc[h]
        if 0 <= r < n0 and 0 <= c < n1 and not blocked[r, c] and not visited[r, c]:
            if goal[r, c]:
                return a
            visited[r, c] = True
            qr[tail] = r
            qc[tail] = c
            qa[tail] = a
            tail += 1
    while head < tail:
        r = qr[head]
        c = qc[head]
        a = qa[head]
        head += 1
        for d in range(4):
            nr = r + dr[d]
            nc = c + dc[d]
            if 0 <= nr < n0 and 0 <= nc < n1 and not visited[nr, nc] and not blocked[nr, nc]:
                if goal[nr, nc]:
                    return a
                visited[nr, nc] = True
                qr[tail] = nr
                qc[tail] = nc
                qa[tail] = a
                tail += 1
    return -1


_bfs_fast = njit(cache=True)(_bfs_core) if njit is not None else _bfs_core


def bfs_first_action(
    game: SplixGame,
    pid: int,
    goal: np.ndarray,
    blocked: np.ndarray,
    order: list[int] | None = None,
) -> int | None:
    """First relative action on a shortest path from pid's head to any goal cell.

    ``blocked`` cells are never entered. Returns None if no goal is reachable.
    ``order`` sets the tie-break between equally short first moves.
    """
    p = game.players[pid]
    order_arr = np.asarray(order if order is not None else range(N_ACTIONS), dtype=np.int64)
    a = _bfs_fast(p.pos[0], p.pos[1], p.heading, np.ascontiguousarray(goal, dtype=np.bool_),
                  np.ascontiguousarray(blocked, dtype=np.bool_), order_arr, _DR, _DC)
    return None if a < 0 else int(a)
