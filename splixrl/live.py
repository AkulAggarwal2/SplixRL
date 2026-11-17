"""Play on a real Splix game server (run locally, see README "Real game").

    # terminal 1: the official server (downloaded release), 40x40 arena
    .splix-server/server/splixGameServer -s 40

    # terminal 2: a trained agent plus three bots as opponents; watch in the browser
    python -m splixrl.live --actor runs/round0/dyna/seed0/agent.pt --bots loop chaser loop --serve-client

How it works. The server moves every player one tile every 1/0.006 ms
(~6 tiles/s) and a client says "turn to direction D at tile (x, y)"; if that
tile is still ahead, the server applies the turn when the player gets there.
So once per tile, half a tile before reaching the next tile P, we:
  1. rebuild a ``SplixGame`` snapshot from everything the server has told us
     (tiles, trails, player positions dead-reckoned from their last update),
     with our head at P;
  2. ask the actor -- a trained agent or one of our bots -- for an action, through
     exactly the same observation code used in training;
  3. if it turns, send the turn for tile P.
Coordinates: the server's (x, y) is our (col, row); its border tiles are
deadly, so the snapshot covers only the playable area inside them.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .evaluation import make_actor
from .game import DOWN, EMPTY, LEFT, PAUSE, RIGHT, STRAIGHT, UP, Player, SplixGame, turn
from .obs import build_obs

TILE_SEC = 1 / 0.006 / 1000  # seconds per tile (PLAYER_TRAVEL_SPEED in the server config)

# server direction ids: 0 right, 1 down, 2 left, 3 up, 4 paused
SERVER_TO_HEADING = {0: RIGHT, 1: DOWN, 2: LEFT, 3: UP}
HEADING_TO_SERVER = {v: k for k, v in SERVER_TO_HEADING.items()}
SERVER_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}  # (dx, dy)

# client -> server
C_UPDATE_MY_POS, C_SET_USERNAME, C_SKIN, C_READY, C_PING = 1, 2, 3, 4, 7
# server -> client
S_PLAYER_STATE, S_FILL_RECT, S_SET_PLAYER_TRAIL, S_PLAYER_DIE = 2, 3, 4, 5
S_REMOVE_PLAYER, S_MY_SCORE, S_MY_RANK, S_LEADERBOARD = 7, 9, 10, 11
S_MAP_SIZE, S_GAME_OVER, S_PLAYER_SKIN = 12, 13, 15
S_EMPTY_TRAIL, S_READY, S_UNDO_PLAYER_DIE = 16, 17, 22
S_MINIMAP = 14
MINIMAP_SIZE, MINIMAP_PART_W = 80, 20  # 80x80 bitmap of "anyone's land", sent as 4 strips of 20 columns
MINIMAP_SCALE = 3

TILE_WALL, TILE_EMPTY = 0, 1  # other tile values are skin colour + 2
MAX_PAUSE = 20  # same cap as training (SplixEnv.max_pause): after this many paused tiles, go
UNSEEN_OWNER = 120  # snapshot owner id for land whose player isn't in view
DEATH_TYPES = {0: "unknown", 1: "player", 2: "arena-bounds", 3: "self"}


@dataclass
class LivePlayer:
    pid: int
    x: int
    y: int
    dir: int  # server direction id
    t: float  # time (s) at which the player was at (x, y)
    alive: bool = True
    trail: list[tuple[int, int]] = field(default_factory=list)  # trail corner vertices (x, y)
    color: int | None = None

    def pos_at(self, now: float) -> tuple[int, int]:
        if not self.alive or self.dir == 4:
            return self.x, self.y
        k = max(0, int((now - self.t) / TILE_SEC))
        dx, dy = SERVER_DELTA[self.dir]
        return self.x + k * dx, self.y + k * dy


def raster(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cells covered by an axis-aligned polyline."""
    cells: list[tuple[int, int]] = []
    for (x0, y0), (x1, y1) in zip(vertices, vertices[1:]):
        sx = (x1 > x0) - (x1 < x0)
        sy = (y1 > y0) - (y1 < y0)
        x, y = x0, y0
        while (x, y) != (x1, y1):
            cells.append((x, y))
            x += sx
            y += sy
            if sx and sy:  # diagonal: shouldn't happen
                break
    if vertices:
        cells.append(vertices[-1])
    return cells


class LiveState:
    """Everything the server has told one client, plus dead reckoning."""

    def __init__(self):
        self.map_size: int | None = None
        self.tiles: np.ndarray | None = None  # [y, x], tile values as sent
        self.players: dict[int, LivePlayer] = {}
        self.my_color: int | None = None
        self.score_tiles = 0
        self.kills = 0
        self.rank: int | None = None
        self.total_players = 0
        self.leaderboard: list[tuple[str, int]] = []
        self.ready = False
        self.game_over: dict | None = None
        self.my_states: list[tuple[float, int, int, int]] = []  # every state update the server sent about us
        self.minimap = np.zeros((MINIMAP_SIZE, MINIMAP_SIZE), bool)  # [y, x]

    # -- messages ---------------------------------------------------------
    def handle(self, m: bytes, now: float) -> None:
        t = m[0]
        if t == S_MAP_SIZE:
            self.map_size = struct.unpack(">H", m[1:3])[0]
            self._ensure_tiles(self.map_size)
        elif t == S_FILL_RECT:
            x, y, w, h, color = struct.unpack(">HHHHB", m[1:10])
            self._ensure_tiles(max(x + w, y + h, self.map_size or 0))
            self.tiles[y : y + h, x : x + w] = color
        elif t == S_PLAYER_STATE:
            x, y, pid, d = struct.unpack(">HHHB", m[1:8])
            p = self.players.get(pid)
            if p is None:
                p = self.players[pid] = LivePlayer(pid, x, y, d, now)
            # The server only sends a trail's first vertex; every later state
            # update of a player that has a trail is a corner of that trail.
            if p.trail and p.trail[-1] != (x, y):
                p.trail.append((x, y))
            p.x, p.y, p.dir, p.t = x, y, d, now
            if pid == 0:
                self.my_states.append((now, x, y, d))
        elif t == S_PLAYER_SKIN:
            pid, color = struct.unpack(">HB", m[1:4])
            if pid == 0:
                self.my_color = color + 2
            if pid in self.players:
                self.players[pid].color = color + 2
        elif t == S_SET_PLAYER_TRAIL:
            pid = struct.unpack(">H", m[1:3])[0]
            verts = [struct.unpack(">HH", m[3 + 4 * i : 7 + 4 * i]) for i in range((len(m) - 3) // 4)]
            if pid in self.players:
                self.players[pid].trail = verts
        elif t == S_EMPTY_TRAIL:
            pid = struct.unpack(">H", m[1:3])[0]
            if pid in self.players:
                self.players[pid].trail = []
        elif t == S_PLAYER_DIE:
            pid = struct.unpack(">H", m[1:3])[0]
            if pid in self.players:
                self.players[pid].alive = False
        elif t == S_UNDO_PLAYER_DIE:
            pid = struct.unpack(">H", m[1:3])[0]
            if pid in self.players:
                self.players[pid].alive = True
        elif t == S_REMOVE_PLAYER:
            pid = struct.unpack(">H", m[1:3])[0]
            if pid != 0:
                self.players.pop(pid, None)
        elif t == S_MY_SCORE:
            self.score_tiles, self.kills = struct.unpack(">IH", m[1:7])
        elif t == S_MY_RANK:
            self.rank = struct.unpack(">H", m[1:3])[0]
        elif t == S_LEADERBOARD:
            self.total_players = struct.unpack(">H", m[1:3])[0]
            board, i = [], 3
            while i + 5 <= len(m):
                score, nlen = struct.unpack(">IB", m[i : i + 5])
                i += 5
                if i + nlen > len(m):
                    break
                board.append((m[i : i + nlen].decode("utf-8", errors="replace"), score))
                i += nlen
            self.leaderboard = board
        elif t == S_READY:
            self.ready = True
        elif t == S_MINIMAP:
            self.handle_minimap(m[1], m[2:])
        elif t == S_GAME_OVER:
            tiles, kills, rank, alive_s, first_s, death = struct.unpack(">IHHIIB", m[1:18])
            self.game_over = {
                "tiles": tiles, "kills": kills, "best_rank": rank, "seconds_alive": alive_s,
                "death": DEATH_TYPES.get(death, str(death)), "killed_by": m[18:].decode(errors="replace"),
            }

    def handle_minimap(self, part: int, data: bytes) -> None:
        """Bits are column-major within the strip: index i -> (x = i // 80, y = i % 80)."""
        bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")[: MINIMAP_PART_W * MINIMAP_SIZE]
        cols = bits.reshape(MINIMAP_PART_W, MINIMAP_SIZE)  # [x, y]
        x0 = part * MINIMAP_PART_W
        self.minimap[:, x0 : x0 + MINIMAP_PART_W] = cols.T.astype(bool)

    def land_hint(self) -> np.ndarray:
        """The minimap scaled to the playable board ([row, col]), for the agent's minimap channel."""
        W = self.map_size
        n = W - 2
        # board cell (r, c) is arena tile (x = c + 1, y = r + 1); the minimap samples tile floor(i * W / 80)
        idx = np.minimum((np.arange(1, n + 1) * MINIMAP_SIZE) // W, MINIMAP_SIZE - 1)
        return self.minimap[np.ix_(idx, idx)]

    def minimap_px(self, x: int, y: int) -> tuple[int, int]:
        """Server tile (x, y) -> pixel on the 80x80 minimap."""
        w = self.map_size or MINIMAP_SIZE
        return (min(MINIMAP_SIZE - 1, x * MINIMAP_SIZE // w), min(MINIMAP_SIZE - 1, y * MINIMAP_SIZE // w))

    def _ensure_tiles(self, size: int) -> None:
        if self.tiles is None or self.tiles.shape[0] < size:
            new = np.full((size, size), TILE_EMPTY, np.int16)
            if self.tiles is not None:
                n = self.tiles.shape[0]
                new[:n, :n] = self.tiles
            self.tiles = new

    # -- snapshot -----------------------------------------------------------
    def snapshot(self, head: tuple[int, int], heading: int, now: float, steps: int, max_steps: int) -> SplixGame:
        """A SplixGame of the playable area as seen now, with our head at ``head`` (x, y)."""
        W = self.map_size
        n = W - 2
        g = SplixGame(size=n, n_players=1, spawn_radius=0, max_steps=max_steps)
        tiles = self.tiles[1 : W - 1, 1 : W - 1]
        mine = tiles == self.my_color
        g.owner[mine] = 0
        # land of players we can't see yet: owned, but by nobody in the player list
        g.owner[(tiles >= 2) & ~mine] = UNSEEN_OWNER
        g.trail.fill(EMPTY)
        g.t = min(steps, max_steps)
        g.land_hint = self.land_hint()  # far-away land the client only knows from the minimap

        def to_rc(x, y):
            return y - 1, x - 1

        def inside(r, c):
            return 0 <= r < n and 0 <= c < n

        me = self.players[0]
        my_trail_xy = raster(me.trail + [me.pos_at(now)]) if me.trail else []
        for x, y in {me.pos_at(now), head}:  # cells about to join the trail
            r, c = to_rc(x, y)
            if inside(r, c) and g.owner[r, c] != 0:
                my_trail_xy.append((x, y))
        my_trail = []
        for x, y in dict.fromkeys(my_trail_xy):
            r, c = to_rc(x, y)
            if inside(r, c):
                g.trail[r, c] = 0
                my_trail.append((r, c))
        r, c = to_rc(*head)
        g.players = [Player(pid=0, pos=(r, c), heading=heading, trail=my_trail)]

        for p in self.players.values():
            if p.pid == 0 or not p.alive:
                continue
            x, y = p.pos_at(now)
            pr, pc = to_rc(x, y)
            if not inside(pr, pc):
                continue
            q = Player(pid=len(g.players), pos=(pr, pc), heading=SERVER_TO_HEADING.get(p.dir, UP))
            if p.color is not None and p.color != self.my_color:
                g.owner[(tiles == p.color) & (g.owner == UNSEEN_OWNER)] = q.pid  # their land, for size features
            if p.trail:
                for tx, ty in raster(p.trail + [(x, y)]):
                    tr, tc = to_rc(tx, ty)
                    if inside(tr, tc) and g.trail[tr, tc] == EMPTY:
                        g.trail[tr, tc] = q.pid
                        q.trail.append((tr, tc))
            g.players.append(q)
        g.n_players = len(g.players)
        return g


HEADING_NAME = ["up", "right", "down", "left"]
FOLLOW_RADIUS = 22  # tiles around the head when the arena does not fit on screen
PANEL_W = 14 + MINIMAP_SIZE * MINIMAP_SCALE + 14


def minimap_image(
    state: LiveState,
    me: tuple[int, int],
    view: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """80x80 minimap scaled up: land, the camera rectangle, and where ``me`` (server x, y) is."""
    mm = np.zeros((MINIMAP_SIZE, MINIMAP_SIZE, 3), np.uint8)
    mm[:] = (42, 38, 35)
    mm[state.minimap] = (214, 176, 92)
    if view is not None:
        x0, y0, x1, y1 = view
        px0, py0 = state.minimap_px(max(0, x0), max(0, y0))
        px1, py1 = state.minimap_px(x1, y1)
        tone = (110, 168, 214)
        mm[py0 : py1 + 1, px0] = tone
        mm[py0 : py1 + 1, px1] = tone
        mm[py0, px0 : px1 + 1] = tone
        mm[py1, px0 : px1 + 1] = tone
    mx, my = state.minimap_px(*me)
    mm[max(0, my - 1) : my + 2, max(0, mx - 1) : mx + 2] = (168, 86, 255)
    mm[my, mx] = (255, 255, 255)
    return np.repeat(np.repeat(mm, MINIMAP_SCALE, 0), MINIMAP_SCALE, 1)


class LiveWindow:
    """pygame window drawing the live game as one client sees it."""

    def __init__(self, title: str):
        import pygame

        self.pg = pygame
        pygame.init()
        self.title = title
        self.screen = None
        self.font = pygame.font.SysFont("Helvetica", 16)
        self.font_sm = pygame.font.SysFont("menlo,consolas,couriernew,courier", 14)

    def draw(self, game: SplixGame, state: LiveState, name: str, life: int) -> bool:
        from .render import render_rgb

        pg = self.pg
        # A local server is about 40 tiles and fits on screen. The public arenas
        # are 400, and drawing every tile makes a ~2600px window whose frame
        # takes longer than one tile, so the turn arrives after the player has
        # already left it. Follow the head instead.
        me = game.players[0]
        cell = max(6, min(24, 760 // max(game.size, 1)))
        view = None
        if game.size * cell <= 900:
            img = render_rgb(game, cell)
        else:
            cell = 16
            img = render_rgb(game, cell, focus=me.pos, radius=FOLLOW_RADIUS)
            r, c = me.pos
            view = (c + 1 - FOLLOW_RADIUS, r + 1 - FOLLOW_RADIUS, c + 1 + FOLLOW_RADIUS, r + 1 + FOLLOW_RADIUS)
        x, y = me.pos[1] + 1, me.pos[0] + 1
        w = state.map_size or 1
        lines = [
            name,
            f"life {life + 1}",
            "",
            f"you   x {x}   y {y}",
            f"      {100 * x / w:.0f}% across, {100 * y / w:.0f}% down",
            f"facing {HEADING_NAME[me.heading]}",
            "on your land" if game.owner[me.pos] == 0 else f"trailing  {len(me.trail)} tiles",
            "",
            f"tiles {state.score_tiles}    kills {state.kills}",
            f"rank {state.rank if state.rank is not None else '-'} of {state.total_players or '?'}",
            f"in view {len(game.players)}    map {w}",
            "",
            "leaderboard",
        ]
        for i, (who, score) in enumerate(state.leaderboard[:8], start=1):
            mark = ">" if who == name else " "
            lines.append(f"{mark}{i} {who[:14]:<14} {score}")
        lines += ["", "blue box: this view", "white dot: you", "esc: close"]
        mini = minimap_image(state, (x, y), view)
        panel_h = 12 + mini.shape[0] + 10 + len(lines) * 18 + 8
        size = (img.shape[1] + PANEL_W, max(img.shape[0], panel_h, 320))
        if self.screen is None or self.screen.get_size() != size:
            self.screen = pg.display.set_mode(size)
            pg.display.set_caption(self.title)
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                return False
        self.screen.fill((32, 29, 27))
        self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), (0, 0))
        ox = img.shape[1] + 14
        self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(mini.transpose(1, 0, 2))), (ox, 12))
        yy = 12 + mini.shape[0] + 10
        for line in lines:
            font = self.font if line == name else self.font_sm
            self.screen.blit(font.render(line, True, (235, 235, 235)), (ox, yy))
            yy += 18
        pg.display.flip()
        return True


class _FakeEnv:
    """Just enough of SplixEnv for the evaluation Actors."""

    def __init__(self, game: SplixGame, obs_config):
        self.game = game
        self.obs_config = obs_config


class WebLink:
    """The official Splix page, bridged to the real server. The page renders; we steer."""

    def __init__(self, upstream, browser):
        self.upstream = upstream
        self.browser = browser
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._done = False

    async def send(self, data) -> None:
        async with self._send_lock:
            await self.upstream.send(data)

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        while True:
            message = await self.queue.get()
            if message is None:
                break
            yield message

    async def run(self) -> None:
        import websockets

        async def downstream():
            try:
                async for message in self.upstream:
                    if isinstance(message, bytes):
                        self.queue.put_nowait(message)
                    await self.browser.send(message)
            except websockets.ConnectionClosed:
                pass

        async def upstream():
            try:
                async for message in self.browser:
                    # Arrow keys belong to the page. The actor sends the turns.
                    if isinstance(message, bytes) and message[:1] == bytes([C_UPDATE_MY_POS]):
                        continue
                    async with self._send_lock:
                        await self.upstream.send(message)
            except websockets.ConnectionClosed:
                pass

        tasks = [asyncio.create_task(downstream()), asyncio.create_task(upstream())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for sock in (self.upstream, self.browser):
                try:
                    await sock.close()
                except Exception:
                    pass
            if not self._done:
                self._done = True
                self.closed.set()
                self.queue.put_nowait(None)


class WebRelay:
    """Local websocket the splix.io page connects to. Each Play click is one life."""

    def __init__(self, upstream: str):
        self.upstream = upstream
        self._links: asyncio.Queue = asyncio.Queue()
        self._server = None
        self.port: int | None = None

    async def start(self) -> None:
        import websockets

        self._server = await websockets.serve(self._handler, "127.0.0.1", 0, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def next_link(self) -> WebLink:
        return await self._links.get()

    async def _handler(self, browser) -> None:
        import websockets

        async with websockets.connect(
            self.upstream, max_size=None, additional_headers={"Origin": "https://splix.io"}
        ) as upstream:
            link = WebLink(upstream, browser)
            await self._links.put(link)
            await link.run()

    @property
    def page(self) -> str:
        return f"https://splix.io/#ip=ws://localhost:{self.port}"


async def play_life(server: str, actor, name: str, color: int, max_steps: int, verbose: bool, seat: int = 0,
                    window: LiveWindow | None = None, life: int = 0, relay: WebRelay | None = None) -> dict:
    import websockets

    from .obs import ObsConfig

    obs_config = getattr(getattr(actor, "agent", None), "obs_config", None) or ObsConfig()
    state = LiveState()
    # The server rate-limits new connections per IP and takes the IP from
    # X-Real-IP when present. All our clients run on this machine, so each seat
    # gets its own made-up address; otherwise bots respawning would lock out
    # each other (and the browser). Only sent to a server on this machine:
    # telling someone else's server a made-up address would be spoofing.
    from urllib.parse import urlparse

    local = urlparse(server).hostname in ("localhost", "127.0.0.1", "::1")
    headers = {"X-Real-IP": f"10.77.0.{seat + 1}"} if local else {}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def opened():
        if relay is None:
            async with websockets.connect(server, max_size=None, additional_headers=headers) as ws:
                await ws.send(bytes([C_SET_USERNAME]) + name.encode()[:20])
                await ws.send(bytes([C_SKIN, color, 0]))
                await ws.send(bytes([C_READY]))
                yield ws
        else:
            # The website sends the name, the skin and ready. We only steer.
            print("waiting for you to click Play on splix.io...", flush=True)
            yield await relay.next_link()

    async with opened() as ws:
        closed = asyncio.Event()

        async def reader():
            try:
                async for m in ws:
                    if isinstance(m, bytes):
                        state.handle(m, time.monotonic())
            except websockets.ConnectionClosed:
                pass
            finally:
                closed.set()

        reader_task = asyncio.create_task(reader())
        try:
            while not (state.ready and 0 in state.players and state.map_size and state.my_color):
                if closed.is_set():
                    raise ConnectionError("server closed the connection before the game started")
                await asyncio.sleep(0.01)
            decided: set[tuple[int, int, int]] = set()
            steps, last_ping, reset_done = 0, time.monotonic(), False
            # Lag handling. The server drops turn messages while a tick runs slow,
            # and may apply a turn late or rewind us ("lag back"). Every turn we send
            # must come back as a state update at that tile; if it doesn't, resend it
            # (the server then rewinds to that tile), and after a few tries assume it
            # was lost and go back to the straight-line prediction.
            pending: dict | None = None
            last_heading = SERVER_TO_HEADING.get(state.players[0].dir, UP)
            paused_for, last_pause_decision = 0, 0.0
            seen_states = len(state.my_states)
            lag = {"turns": 0, "confirmed": 0, "resent": 0, "lost": 0, "corrections": 0}
            while state.game_over is None and not closed.is_set():
                now = time.monotonic()
                if now - last_ping > 20:
                    await ws.send(bytes([C_PING]))
                    last_ping = now
                for _, sx, sy, sdir in state.my_states[seen_states:]:
                    if pending and (sx, sy, sdir) == (*pending["pos"], pending["dir"]):
                        lag["confirmed"] += 1
                        pending = None
                    elif pending is None or (sx, sy) != pending["pos"]:
                        lag["corrections"] += 1  # the server moved us somewhere we didn't predict
                seen_states = len(state.my_states)
                if pending and now > pending["eta"] + 0.6 * TILE_SEC * pending["tries"]:
                    if pending["tries"] < 3:
                        await ws.send(pending["msg"])
                        pending["tries"] += 1
                        lag["resent"] += 1
                    else:
                        lag["lost"] += 1
                        me = state.players[0]
                        me.x, me.y, me.dir, me.t = pending["before"]  # it never turned: keep going straight
                        if pending["added_corner"] and me.trail and me.trail[-1] == pending["pos"]:
                            me.trail.pop()
                        pending = None
                me = state.players[0]
                if not me.alive:
                    await asyncio.sleep(TILE_SEC / 4)
                    continue

                def choose(game):
                    nonlocal reset_done, steps
                    env = _FakeEnv(game, obs_config)
                    if not reset_done:
                        actor.reset(env, seed=int(time.time()))
                        reset_done = True
                    steps += 1
                    if hasattr(actor, "bot"):
                        return int(actor.bot.act(game, 0))
                    return int(actor.act(env, build_obs(game, 0, obs_config)))

                if me.dir == 4:
                    # Paused: decide once per tile period, from where we stand.
                    if now - last_pause_decision >= TILE_SEC:
                        last_pause_decision = now
                        game = state.snapshot((me.x, me.y), last_heading, now, steps, max_steps)
                        game.players[0].paused = True
                        action = choose(game)
                        paused_for += 1
                        if action != PAUSE or paused_for > MAX_PAUSE:
                            a = STRAIGHT if action == PAUSE else action
                            sd = HEADING_TO_SERVER[turn(last_heading, a)]
                            msg = struct.pack(">BbHH", C_UPDATE_MY_POS, sd, me.x, me.y)
                            await ws.send(msg)
                            lag["turns"] += 1
                            before = (me.x, me.y, me.dir, me.t)
                            me.dir, me.t = sd, now
                            paused_for = 0
                            pending = {"pos": (me.x, me.y), "dir": sd, "eta": now, "tries": 1, "msg": msg,
                                       "before": before, "added_corner": False}
                        if window is not None and not window.draw(game, state, name, life):
                            window = None
                    await asyncio.sleep(TILE_SEC / 10)
                    continue

                k = int((now - me.t) / TILE_SEC)
                phase = (now - me.t) / TILE_SEC - k
                dx, dy = SERVER_DELTA[me.dir]
                nxt = (me.x + (k + 1) * dx, me.y + (k + 1) * dy)
                t_next = me.t + (k + 1) * TILE_SEC
                key = (nxt[0], nxt[1], me.dir)
                if phase >= 0.35 and key not in decided:
                    decided.add(key)
                    if not (1 <= nxt[0] <= state.map_size - 2 and 1 <= nxt[1] <= state.map_size - 2):
                        await asyncio.sleep(TILE_SEC / 10)  # next tile is the border: nothing left to decide
                        continue
                    heading = SERVER_TO_HEADING[me.dir]
                    last_heading = heading
                    game = state.snapshot(nxt, heading, now, steps, max_steps)
                    action = choose(game)
                    if action == PAUSE:  # stop on tile nxt
                        msg = struct.pack(">BbHH", C_UPDATE_MY_POS, 4, nxt[0], nxt[1])
                        await ws.send(msg)
                        lag["turns"] += 1
                        before = (me.x, me.y, me.dir, me.t)
                        me.x, me.y, me.dir, me.t = nxt[0], nxt[1], 4, t_next
                        pending = {"pos": nxt, "dir": 4, "eta": t_next, "tries": 1, "msg": msg,
                                   "before": before, "added_corner": False}
                        paused_for, last_pause_decision = 1, t_next
                    new_heading = heading if action == PAUSE else turn(heading, int(action))
                    if new_heading != heading:
                        sd = HEADING_TO_SERVER[new_heading]
                        msg = struct.pack(">BbHH", C_UPDATE_MY_POS, sd, nxt[0], nxt[1])
                        await ws.send(msg)
                        lag["turns"] += 1
                        before = (me.x, me.y, me.dir, me.t)
                        # predict: from tile nxt onward we move in the new direction,
                        # and if we have a trail, nxt is a new corner of it
                        added = bool(me.trail) and me.trail[-1] != nxt
                        if added:
                            me.trail.append(nxt)
                        me.x, me.y, me.dir, me.t = nxt[0], nxt[1], sd, t_next
                        pending = {"pos": nxt, "dir": sd, "eta": t_next, "tries": 1, "msg": msg,
                                   "before": before, "added_corner": added}
                        decided.add((nxt[0], nxt[1], sd))
                    if window is not None and not window.draw(game, state, name, life):
                        window = None
                await asyncio.sleep(TILE_SEC / 10)
        except websockets.ConnectionClosed:
            pass
        finally:
            reader_task.cancel()
    result = dict(state.game_over or {"tiles": state.score_tiles, "kills": state.kills, "death": "disconnected"})
    result.update({f"lag_{k}": v for k, v in lag.items()} if "lag" in locals() else {})
    result["name"] = name
    if verbose:
        print(f"[{name}] died ({result.get('death')}, killed by {result.get('killed_by') or '-'}) "
              f"after {result.get('seconds_alive')} s: {result.get('tiles')} tiles, {result.get('kills')} kills | "
              f"turns {result.get('lag_turns')}, confirmed {result.get('lag_confirmed')}, resent {result.get('lag_resent')}, "
              f"lost {result.get('lag_lost')}, corrections {result.get('lag_corrections')}", flush=True)
    return result


async def run_forever(server, spec, name, color, max_steps, stop: asyncio.Event, seat: int):
    while not stop.is_set():
        try:
            await play_life(server, make_actor(spec), name, color, max_steps, verbose=False, seat=seat)
        except Exception as e:  # server restarted, connection dropped...
            print(f"[{name}] {type(e).__name__}: {e}; reconnecting", flush=True)
            await asyncio.sleep(1)


def serve_client(port: int) -> None:
    """Serve the downloaded Splix web client so the game can be watched locally."""
    import functools
    import http.server

    root = Path(__file__).resolve().parent.parent / ".splix-server" / "client"
    if not root.exists():
        raise SystemExit(f"no client files at {root}; see README 'Real game'")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


async def amain(args) -> None:
    import webbrowser

    stop = asyncio.Event()
    colors = [5, 1, 9, 3, 7, 11, 2, 4, 6, 8, 10, 12]
    relay = None
    if args.web:
        relay = WebRelay(args.server)
        await relay.start()
        print(f"real client: {relay.page}", flush=True)
        print("The server list says \"From url\". Click Play. That page is the game; the agent steers it.", flush=True)
        webbrowser.open(relay.page)
    bot_tasks = [
        asyncio.create_task(run_forever(args.server, spec, f"{spec}-{i}", colors[(i + 1) % len(colors)], args.max_steps, stop, seat=i + 1))
        for i, spec in enumerate(args.bots)
    ]
    rows = []
    try:
        if args.actor:
            await asyncio.sleep(1.0)  # let the bots spawn first
            window = LiveWindow(f"SplixRL live - {args.name}") if args.show and relay is None else None
            for life in range(args.lives):
                rows.append(await play_life(args.server, make_actor(args.actor), args.name, colors[0], args.max_steps,
                                            verbose=True, window=window, life=life, relay=relay))
        else:
            await asyncio.Event().wait()  # bots only: run until Ctrl-C
    finally:
        stop.set()
        for t in bot_tasks:
            t.cancel()
        if relay is not None:
            await relay.close()
    if rows:
        tiles = [r.get("tiles", 0) for r in rows]
        secs = [r.get("seconds_alive", 0) for r in rows]
        print(f"\n{args.name}: {len(rows)} lives, mean {np.mean(tiles):.1f} tiles, mean {np.mean(secs):.1f} s alive, "
              f"{sum(r.get('kills', 0) for r in rows)} kills")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader()
                w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="ws://127.0.0.1:8080")
    ap.add_argument("--actor", help="checkpoint path or bot name; omit to run only --bots")
    ap.add_argument("--name", default="SplixRL")
    ap.add_argument("--lives", type=int, default=5, help="games the actor plays before stopping")
    ap.add_argument("--bots", nargs="*", default=[], help="extra players (bot names or checkpoints) that respawn forever")
    ap.add_argument("--max-steps", type=int, default=500, help="only scales the agent's time feature")
    ap.add_argument("--out", help="CSV with one row per life")
    ap.add_argument("--serve-client", action="store_true", help="serve the web client at http://127.0.0.1:8000")
    ap.add_argument("--show", action="store_true", help="open a window showing the game as the actor sees it")
    ap.add_argument("--web", action="store_true", help="open splix.io and let it draw the game while the actor steers")
    args = ap.parse_args(argv)
    if args.serve_client:
        serve_client(8000)
        print(f"watch: http://127.0.0.1:8000/#ip={args.server}", flush=True)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
