"""Draw the board as an RGB array (used by the env, GIF export and manual play)."""

from __future__ import annotations

import numpy as np

from .game import SplixGame

# player 0 (the agent / you) is purple, like the screenshots; then red, yellow, ...
BASE_COLORS = np.array(
    [
        (124, 34, 230),
        (214, 40, 57),
        (230, 190, 40),
        (40, 160, 70),
        (30, 140, 220),
        (240, 120, 30),
        (230, 50, 140),
        (40, 190, 180),
    ],
    dtype=np.float32,
)
BACKGROUND = np.array((62, 56, 52), np.float32)
GRID_LINE = np.array((48, 43, 40), np.float32)
WALL = np.array((20, 18, 17), np.float32)


def player_color(pid: int) -> np.ndarray:
    return BASE_COLORS[pid % len(BASE_COLORS)]


def render_rgb(game: SplixGame, cell: int = 16) -> np.ndarray:
    """(H, W, 3) uint8 picture of the board, ``cell`` pixels per square."""
    n = game.size
    img = np.empty((n, n, 3), np.float32)
    img[:] = BACKGROUND
    for pid in range(game.n_players):
        col = player_color(pid)
        img[game.owner == pid] = col * 0.85
        img[game.trail == pid] = col * 0.55 + 255 * 0.45  # lighter stripe for trails
    big = np.repeat(np.repeat(img, cell, 0), cell, 1)
    # grid lines
    if cell >= 6:
        big[::cell, :] = GRID_LINE
        big[:, ::cell] = GRID_LINE
    # heads: a darker filled square with a light centre
    for p in game.players:
        if not p.alive:
            continue
        r, c = p.pos
        y, x = r * cell, c * cell
        m = max(1, cell // 8)
        big[y + m : y + cell - m, x + m : x + cell - m] = player_color(p.pid) * 0.6
        k = max(1, cell // 3)
        big[y + k : y + cell - k, x + k : x + cell - k] = 255
    # 2-pixel dark frame for the board edge
    out = np.empty((big.shape[0] + 4, big.shape[1] + 4, 3), np.float32)
    out[:] = WALL
    out[2:-2, 2:-2] = big
    return out.clip(0, 255).astype(np.uint8)


def owner_counts(game: SplixGame) -> list[int]:
    return [int(np.count_nonzero(game.owner == pid)) for pid in range(game.n_players)]

