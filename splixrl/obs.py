"""Observations, built from any player's point of view.

Because observations are player-agnostic, a frozen agent can play as an
opponent: the env builds the opponent's observation with the same function.

Two modes:

``local`` (default): an egocentric window of side ``2 * view_radius + 1``
centred on the player's head and rotated so the player always faces up. This
matches the relative action space (straight / left / right) and works on any
board size, so an agent trained on 20x20 can play on 32x32.

``global``: the whole board, not rotated. The heading is given in ``vec``.

Both return ``{"grid": float32 (C, H, W) of 0/1, "vec": float32 (K,)}``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from gymnasium import spaces

from .game import DIRS, EMPTY, SplixGame

LOCAL_CHANNELS = ("own_territory", "own_trail", "opp_territory", "opp_trail", "opp_heads", "wall")
GLOBAL_CHANNELS = ("own_territory", "own_trail", "own_head", "opp_territory", "opp_trail", "opp_heads")
VEC_FEATURES = (
    "in_own_territory",
    "trail_len",
    "territory_frac",
    "home_dist",
    "home_forward",
    "home_right",
    "opp_dist",
    "opp_forward",
    "opp_right",
    "danger_dist",
    "time_frac",
)
GLOBAL_EXTRA_VEC = ("heading_up", "heading_right", "heading_down", "heading_left")

# Distances are divided by a fixed constant (not the board size) so features
# mean the same thing on every board size.
DIST_NORM = 32.0
FAR = 2.0  # value used when there is nothing to measure (e.g. no opponent)


@dataclass(frozen=True)
class ObsConfig:
    mode: str = "local"
    view_radius: int = 7

    def __post_init__(self):
        if self.mode not in ("local", "global"):
            raise ValueError(f"unknown obs mode {self.mode!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObsConfig":
        return cls(**d)

    @property
    def channels(self) -> tuple[str, ...]:
        return LOCAL_CHANNELS if self.mode == "local" else GLOBAL_CHANNELS

    @property
    def vec_features(self) -> tuple[str, ...]:
        return VEC_FEATURES if self.mode == "local" else VEC_FEATURES + GLOBAL_EXTRA_VEC

    def grid_shape(self, board_size: int) -> tuple[int, int, int]:
        if self.mode == "local":
            w = 2 * self.view_radius + 1
            return (len(LOCAL_CHANNELS), w, w)
        return (len(GLOBAL_CHANNELS), board_size, board_size)

    def space(self, board_size: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, self.grid_shape(board_size), np.float32),
                "vec": spaces.Box(-FAR, FAR, (len(self.vec_features),), np.float32),
            }
        )


def egocentric(delta_rc: np.ndarray, heading: int) -> tuple[float, float]:
    """World (row, col) offset -> (forward, right) offset for a heading."""
    fwd = DIRS[heading]
    right = DIRS[(heading + 1) % 4]
    return float(delta_rc @ fwd), float(delta_rc @ right)


def _vec_features(game: SplixGame, pid: int, own: np.ndarray) -> list[float]:
    p = game.players[pid]
    pos = np.array(p.pos)
    size = game.size

    in_own = bool(own[p.pos]) if p.alive else False
    terr = int(np.count_nonzero(own))

    if in_own or terr == 0:
        home = (0.0, 0.0, 0.0) if in_own else (FAR, 0.0, 0.0)
    else:
        cells = np.argwhere(own)
        d = np.abs(cells - pos).sum(1)
        i = int(d.argmin())
        f, r = egocentric(cells[i] - pos, p.heading)
        home = (d[i] / DIST_NORM, f / DIST_NORM, r / DIST_NORM)

    opp_heads = np.array([q.pos for q in game.players if q.alive and q.pid != pid]).reshape(-1, 2)
    if len(opp_heads):
        d = np.abs(opp_heads - pos).sum(1)
        i = int(d.argmin())
        f, r = egocentric(opp_heads[i] - pos, p.heading)
        opp = (d[i] / DIST_NORM, f / DIST_NORM, r / DIST_NORM)
    else:
        opp = (FAR, 0.0, 0.0)

    # how close the nearest opponent head is to any cell of my trail
    if p.trail and len(opp_heads):
        tr = np.array(p.trail)
        danger = np.abs(tr[:, None, :] - opp_heads[None, :, :]).sum(2).min() / DIST_NORM
    else:
        danger = FAR

    time_frac = game.t / game.max_steps if game.max_steps else 0.0
    vals = [
        float(in_own),
        len(p.trail) / DIST_NORM,
        terr / (size * size),
        *home,
        *opp,
        danger,
        time_frac,
    ]
    return [float(np.clip(v, -FAR, FAR)) for v in vals]


def build_obs(game: SplixGame, pid: int, cfg: ObsConfig) -> dict[str, np.ndarray]:
    p = game.players[pid]
    owner, trail = game.owner, game.trail
    own = owner == pid
    vec = _vec_features(game, pid, own)

    if cfg.mode == "local":
        R = cfg.view_radius
        planes = np.zeros((len(LOCAL_CHANNELS), game.size + 2 * R, game.size + 2 * R), np.float32)
        inner = (slice(None), slice(R, R + game.size), slice(R, R + game.size))
        board = planes[inner]
        board[0] = own
        board[1] = trail == pid
        board[2] = (owner != EMPTY) & ~own
        board[3] = (trail != EMPTY) & (trail != pid)
        for q in game.players:
            if q.alive and q.pid != pid:
                board[4][q.pos] = 1.0
        planes[5] = 1.0
        board[5] = 0.0
        r, c = p.pos
        window = planes[:, r : r + 2 * R + 1, c : c + 2 * R + 1]
        # rotate so the heading points up: k quarter turns counter-clockwise
        grid = np.ascontiguousarray(np.rot90(window, k=p.heading, axes=(1, 2)))
        return {"grid": grid, "vec": np.array(vec, np.float32)}

    grid = np.zeros((len(GLOBAL_CHANNELS), game.size, game.size), np.float32)
    grid[0] = own
    grid[1] = trail == pid
    if p.alive:
        grid[2][p.pos] = 1.0
    grid[3] = (owner != EMPTY) & ~own
    grid[4] = (trail != EMPTY) & (trail != pid)
    for q in game.players:
        if q.alive and q.pid != pid:
            grid[5][q.pos] = 1.0
    heading = [0.0] * 4
    heading[p.heading] = 1.0
    return {"grid": grid, "vec": np.array(vec + heading, np.float32)}
