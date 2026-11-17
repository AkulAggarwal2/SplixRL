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

Options for big, crowded maps (local mode; all off by default, so older
checkpoints keep their exact inputs):

* ``coarse_factor = k``: six more channels showing a window k times wider
  (side W*k, max-pooled down to W x W), rotated like the fine view. With W = 15
  and k = 3 that is 45 x 45 tiles, about what a human sees on the real server.
* ``visible_radius = r``: opponents (their land, trails and heads) more than r
  tiles away (Chebyshev) are invisible, as on the real server, which only sends
  what is within 25 tiles. Applies to the grid and to ``vec``.
* ``threat_features``: three more ``vec`` numbers: how close the nearest
  enemy head is to my land, how much enemy trail lies on my land (a threat,
  and a free kill: you can't die on your own land), and the distance from my
  head to the nearest enemy trail.
* ``minimap``: two more channels with the real game's minimap: which parts of
  the whole board are anyone's land (it doesn't say whose), and where I am.
  North-up, not rotated, so the heading is added to ``vec``.
* ``enemy_heading``: one more fine channel (before "wall") marking the tile in
  front of each visible enemy head, so the agent can tell who is coming at it
  or its land, not only where they are.
* ``size_features``: three more ``vec`` numbers on a log scale: my land, the
  nearest visible enemy's visible land, and the difference. Dying costs the
  land you hold, so a small agent can afford more risk against a big one;
  this tells it which case it is in.
* ``reclaim_features``: four more ``vec`` numbers about cheap land: empty
  tiles enclosed by my land, or dents touching it on two or more sides
  (left behind when an enemy that ate into my land dies, since the dead
  player's land is cleared). How many there are, and the distance and
  direction to the nearest one; a tiny loop takes them back.
* ``escape_features``: one more ``vec`` number, the escape margin: (nearest
  visible enemy's distance to my trail or head) - (my distance home). The
  basic safety rule of the game: keep it positive, i.e. only go as far out as
  you can get back before anyone can cut you off. Always the last feature, so
  a checkpoint without it can be extended (new input weights start at 0).
* ``pause_feature``: one more ``vec`` number, 1 while I'm paused (set
  automatically when the env allows pausing). After the escape margin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np
from gymnasium import spaces

from .game import DIRS, EMPTY, SplixGame

LOCAL_CHANNELS = ("own_territory", "own_trail", "opp_territory", "opp_trail", "opp_heads", "wall")
HEADING_LOCAL_CHANNELS = ("own_territory", "own_trail", "opp_territory", "opp_trail", "opp_heads", "opp_next", "wall")
COARSE_CHANNELS = tuple("coarse_" + c for c in LOCAL_CHANNELS)
MINIMAP_CHANNELS = ("map_land", "map_me")
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
THREAT_FEATURES = ("home_threat_dist", "enemy_trail_on_my_land", "enemy_trail_dist")
SIZE_FEATURES = ("my_size", "enemy_size", "size_advantage")
RECLAIM_FEATURES = ("cheap_land", "cheap_dist", "cheap_forward", "cheap_right")
ESCAPE_FEATURES = ("escape_margin",)
SIZE_LOG_NORM = 8.0  # log1p(cells) / 8: ~1 at 3000 cells
HEADING_FEATURES = ("heading_up", "heading_right", "heading_down", "heading_left")
GLOBAL_EXTRA_VEC = HEADING_FEATURES

# Distances are divided by a fixed constant (not the board size) so features
# mean the same thing on every board size.
DIST_NORM = 32.0
FAR = 2.0  # value used when there is nothing to measure (e.g. no opponent)
SERVER_VIEW = 25  # the real server sends players and tiles within this many tiles


@dataclass(frozen=True)
class ObsConfig:
    mode: str = "local"
    view_radius: int = 7
    coarse_factor: int = 0
    visible_radius: int | None = None
    threat_features: bool = False
    minimap: bool = False
    enemy_heading: bool = False
    size_features: bool = False
    reclaim_features: bool = False
    escape_features: bool = False
    pause_feature: bool = False

    def __post_init__(self):
        if self.mode not in ("local", "global"):
            raise ValueError(f"unknown obs mode {self.mode!r}")
        if self.mode == "global" and (self.coarse_factor or self.minimap):
            raise ValueError("coarse_factor and minimap are for local mode")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObsConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def width(self) -> int:
        return 2 * self.view_radius + 1

    @property
    def board_channels(self) -> tuple[str, ...]:
        return HEADING_LOCAL_CHANNELS if self.enemy_heading else LOCAL_CHANNELS

    @property
    def channels(self) -> tuple[str, ...]:
        if self.mode == "global":
            return GLOBAL_CHANNELS
        ch = self.board_channels
        if self.coarse_factor:
            ch += tuple("coarse_" + c for c in self.board_channels)
        if self.minimap:
            ch += MINIMAP_CHANNELS
        return ch

    @property
    def fine_channels(self) -> int:
        """Channels that shift one cell per move (the rest only rotate, or neither)."""
        return len(self.board_channels) if self.mode == "local" else len(GLOBAL_CHANNELS)

    @property
    def rotating_channels(self) -> int:
        """Leading channels that rotate with the heading (fine + coarse)."""
        return self.fine_channels * (2 if self.coarse_factor else 1)

    @property
    def vec_features(self) -> tuple[str, ...]:
        v = VEC_FEATURES
        if self.threat_features:
            v += THREAT_FEATURES
        if self.size_features:
            v += SIZE_FEATURES
        if self.reclaim_features:
            v += RECLAIM_FEATURES
        if self.mode == "global" or self.minimap:
            v += HEADING_FEATURES
        if self.escape_features:
            v += ESCAPE_FEATURES
        if self.pause_feature:
            v += ("paused",)
        return v

    def grid_shape(self, board_size: int) -> tuple[int, int, int]:
        if self.mode == "local":
            return (len(self.channels), self.width, self.width)
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


def _visible_mask(game: SplixGame, pos: tuple[int, int], radius: int | None) -> np.ndarray | None:
    if radius is None:
        return None
    r, c = pos
    m = np.zeros((game.size, game.size), bool)
    m[max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1] = True
    return m


def _min_dist(a: np.ndarray, b: np.ndarray) -> float | None:
    """Smallest Manhattan distance between two sets of (row, col) points."""
    if not len(a) or not len(b):
        return None
    return float(np.abs(a[:, None, :] - b[None, :, :]).sum(2).min())


def _vec_features(game: SplixGame, pid: int, own: np.ndarray, cfg: ObsConfig,
                  visible: np.ndarray | None, enemy_trail: np.ndarray) -> list[float]:
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

    opp_heads = [q.pos for q in game.players if q.alive and q.pid != pid]
    if visible is not None:
        opp_heads = [h for h in opp_heads if visible[h]]
    opp_heads = np.array(opp_heads).reshape(-1, 2)
    if len(opp_heads):
        d = np.abs(opp_heads - pos).sum(1)
        i = int(d.argmin())
        f, r = egocentric(opp_heads[i] - pos, p.heading)
        opp = (d[i] / DIST_NORM, f / DIST_NORM, r / DIST_NORM)
    else:
        opp = (FAR, 0.0, 0.0)

    # how close the nearest opponent head is to any cell of my trail
    danger = _min_dist(np.array(p.trail).reshape(-1, 2), opp_heads)
    danger = FAR if danger is None else danger / DIST_NORM

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
    if cfg.threat_features:
        own_cells = np.argwhere(own)
        # only the edge of my land matters for "how close is an enemy"
        if len(own_cells) > 400:
            edge = own & ~(np.roll(own, 1, 0) & np.roll(own, -1, 0) & np.roll(own, 1, 1) & np.roll(own, -1, 1))
            own_cells = np.argwhere(edge)
        threat = _min_dist(own_cells, opp_heads)
        on_land = int(np.count_nonzero(enemy_trail & own))
        etrail = _min_dist(np.argwhere(enemy_trail), pos[None, :])
        vals += [
            FAR if threat is None else threat / DIST_NORM,
            on_land / DIST_NORM,
            FAR if etrail is None else etrail / DIST_NORM,
        ]
    if cfg.size_features:
        mine = np.log1p(terr) / SIZE_LOG_NORM
        enemy = 0.0
        if len(opp_heads):
            near = opp_heads[int(np.abs(opp_heads - pos).sum(1).argmin())]
            q = next(q for q in game.players if q.alive and q.pid != pid and tuple(q.pos) == tuple(near))
            land = game.owner == q.pid
            if visible is not None:
                land = land & visible
            enemy = np.log1p(int(np.count_nonzero(land))) / SIZE_LOG_NORM
        vals += [mine, enemy, mine - enemy]
    if cfg.reclaim_features:
        cheap = cheap_land(game, own)
        cells = np.argwhere(cheap)
        if len(cells):
            d = np.abs(cells - pos).sum(1)
            i = int(d.argmin())
            f, r = egocentric(cells[i] - pos, p.heading)
            vals += [len(cells) / DIST_NORM, d[i] / DIST_NORM, f / DIST_NORM, r / DIST_NORM]
        else:
            vals += [0.0, FAR, 0.0, 0.0]
    if cfg.mode == "global" or cfg.minimap:
        heading = [0.0] * 4
        heading[p.heading] = 1.0
        vals += heading
    if cfg.escape_features:
        if not len(opp_heads):
            margin = FAR
        else:
            exposed = np.array([*p.trail, p.pos]).reshape(-1, 2) if p.trail else pos[None, :]
            enemy_reach = _min_dist(exposed, opp_heads)
            my_way_home = 0.0 if in_own else (home[0] * DIST_NORM if terr else FAR * DIST_NORM)
            margin = (enemy_reach - my_way_home) / DIST_NORM
        vals.append(margin)
    if cfg.pause_feature:
        vals.append(float(getattr(p, "paused", False)))
    return [float(np.clip(v, -FAR, FAR)) for v in vals]


def cheap_land(game: SplixGame, own: np.ndarray) -> np.ndarray:
    """Empty tiles that are cheap to (re)capture: enclosed by my land, or
    touching it on two or more sides."""
    from scipy.ndimage import binary_fill_holes

    empty = (game.owner == EMPTY) & (game.trail == EMPTY)
    holes = binary_fill_holes(own) & ~own
    p = np.pad(own, 1)
    sides = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return empty & (holes | (sides >= 2))


def _window(planes: np.ndarray, pos: tuple[int, int], radius: int, size: int) -> np.ndarray:
    """(C, 2r+1, 2r+1) crop around pos from board planes; off-board is wall (last channel)."""
    C = planes.shape[0]
    padded = np.zeros((C, size + 2 * radius, size + 2 * radius), np.float32)
    padded[-1] = 1.0
    padded[:, radius : radius + size, radius : radius + size] = planes
    r, c = pos
    return padded[:, r : r + 2 * radius + 1, c : c + 2 * radius + 1]


def minimap_planes(game: SplixGame, pid: int, w: int) -> np.ndarray:
    """(2, w, w): anyone's land on a w x w sampling of the board, and my position.

    Uses ``game.land_hint`` (from the real server's minimap) where present.
    """
    n = game.size
    land = game.owner != EMPTY
    hint = getattr(game, "land_hint", None)
    if hint is not None:
        land = land | hint
    idx = np.minimum(((np.arange(w) + 0.5) * n / w).astype(int), n - 1)
    out = np.zeros((2, w, w), np.float32)
    out[0] = land[np.ix_(idx, idx)]
    p = game.players[pid]
    if p.alive:
        out[1][min(w - 1, p.pos[0] * w // n), min(w - 1, p.pos[1] * w // n)] = 1.0
    return out


def build_obs(game: SplixGame, pid: int, cfg: ObsConfig) -> dict[str, np.ndarray]:
    p = game.players[pid]
    owner, trail = game.owner, game.trail
    own = owner == pid
    visible = _visible_mask(game, p.pos, cfg.visible_radius)
    enemy_trail = (trail != EMPTY) & (trail != pid)
    if visible is not None:
        enemy_trail = enemy_trail & visible
    vec = _vec_features(game, pid, own, cfg, visible, enemy_trail)

    if cfg.mode == "local":
        board = np.zeros((len(cfg.board_channels), game.size, game.size), np.float32)
        board[0] = own
        board[1] = trail == pid
        board[2] = (owner != EMPTY) & ~own
        board[3] = enemy_trail
        for q in game.players:
            if q.alive and q.pid != pid:
                board[4][q.pos] = 1.0
                if cfg.enemy_heading:  # the tile it moves into next if it keeps going
                    nr, nc = q.pos[0] + int(DIRS[q.heading, 0]), q.pos[1] + int(DIRS[q.heading, 1])
                    if 0 <= nr < game.size and 0 <= nc < game.size:
                        board[5][nr, nc] = 1.0
        if visible is not None:
            board[2 : len(cfg.board_channels) - 1] *= visible  # opponents' land, trails, heads (+ heading)
        R = cfg.view_radius
        parts = [_window(board, p.pos, R, game.size)]
        if cfg.coarse_factor:
            k = cfg.coarse_factor
            w = cfg.width
            big = _window(board, p.pos, (w * k) // 2, game.size)[:, : w * k, : w * k]
            if big.shape[1] < w * k:  # odd/even edge case: pad to a multiple of k
                pad = np.zeros((big.shape[0], w * k, w * k), np.float32)
                pad[:, : big.shape[1], : big.shape[2]] = big
                big = pad
            parts.append(big.reshape(big.shape[0], w, k, w, k).max(axis=(2, 4)))
        # rotate so the heading points up: k quarter turns counter-clockwise
        grid = np.concatenate([np.rot90(x, k=p.heading, axes=(1, 2)) for x in parts], 0)
        if cfg.minimap:
            grid = np.concatenate([grid, minimap_planes(game, pid, cfg.width)], 0)
        return {"grid": np.ascontiguousarray(grid), "vec": np.array(vec, np.float32)}

    grid = np.zeros((len(GLOBAL_CHANNELS), game.size, game.size), np.float32)
    grid[0] = own
    grid[1] = trail == pid
    if p.alive:
        grid[2][p.pos] = 1.0
    grid[3] = (owner != EMPTY) & ~own
    grid[4] = enemy_trail
    for q in game.players:
        if q.alive and q.pid != pid and (visible is None or visible[q.pos]):
            grid[5][q.pos] = 1.0
    if visible is not None:
        grid[3] *= visible
    return {"grid": grid, "vec": np.array(vec, np.float32)}
