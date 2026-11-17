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
    n = game.size
    visited = np.zeros((n, n), dtype=bool)
    visited[p.pos] = True
    queue: deque[tuple[int, int, int]] = deque()
    for a in order or range(N_ACTIONS):
        h = turn(p.heading, a)
        r, c = p.pos[0] + int(DIRS[h, 0]), p.pos[1] + int(DIRS[h, 1])
        if 0 <= r < n and 0 <= c < n and not blocked[r, c] and not visited[r, c]:
            if goal[r, c]:
                return a
            visited[r, c] = True
            queue.append((r, c, a))
    while queue:
        r, c, a = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < n and 0 <= nc < n and not visited[nr, nc] and not blocked[nr, nc]:
                if goal[nr, nc]:
                    return a
                visited[nr, nc] = True
                queue.append((nr, nc, a))
    return None
