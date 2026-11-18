import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from splixrl import SplixEnv
from splixrl.agents import load_agent
from splixrl.agents.dyna import DynaAgent, DynaConfig, align
from splixrl.evaluation import AgentActor, BotActor, evaluate, summarize
from splixrl.game import DOWN, EMPTY, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP
from splixrl.obs import LOCAL_CHANNELS, ObsConfig, build_obs
from splixrl.opponents import make_opponent
from splixrl.play import relative_action

WALL = LOCAL_CHANNELS.index("wall")
OWN = LOCAL_CHANNELS.index("own_territory")


def test_gymnasium_api():
    check_env(SplixEnv(size=16), skip_render_check=True)


def test_global_obs_mode():
    env = SplixEnv(size=16, obs_mode="global")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("heading,wall_side", [(UP, "top"), (RIGHT, "top"), (LEFT, "top"), (DOWN, "top")])
def test_local_view_faces_heading(heading, wall_side):
    """Put the player next to the wall it is facing: the wall must appear at the top of the view."""
    env = SplixEnv(size=16, view_radius=3)
    env.reset(seed=0)
    p = env.game.players[0]
    n = env.game.size
    p.pos = {UP: (0, 8), RIGHT: (8, n - 1), DOWN: (n - 1, 8), LEFT: (8, 0)}[heading]
    p.heading = heading
    grid = build_obs(env.game, 0, env.obs_config)["grid"]
    assert grid[WALL, :3, :].all()  # rows above the centre are off the board
    assert not grid[WALL, 3:, :].any()


def test_relative_actions_match_view():
    """Turning right moves the player toward what the view shows on its right."""
    env = SplixEnv(size=16)
    env.reset(seed=0)
    p = env.game.players[0]
    for h in range(4):
        p.heading = h
        (r1, c1), _ = env.game.next_cell(0, TURN_RIGHT)
        dr, dc = r1 - p.pos[0], c1 - p.pos[1]
        from splixrl.obs import egocentric

        fwd, right = egocentric(np.array([dr, dc]), h)
        assert (fwd, right) == (0.0, 1.0)


def test_align_matches_next_view_for_static_cells():
    """align(obs_t, a) == obs_{t+1} on channels that don't change (walls, own land) when nothing is captured."""
    env = SplixEnv(size=16, view_radius=5, opponents=["random_safe"])
    rng = np.random.default_rng(0)
    checked = 0
    for ep in range(20):
        obs, _ = env.reset(seed=ep)
        for _ in range(40):
            a = int(rng.integers(3))
            if not env.game.is_safe(0, a):
                continue
            terr_before = env.game.territory(0)
            nobs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
            if info["captured"] == 0 and env.game.territory(0) == terr_before:
                pred = align(torch.tensor(obs["grid"])[None], torch.tensor([a]), env.obs_config)[0].numpy()
                R = 5
                # the row/column that just entered the view is unknown in pred; compare the rest
                inner = (slice(1, None), slice(None))
                for ch in (WALL, OWN):
                    assert (pred[ch][inner] == nobs["grid"][ch][inner]).all()
                checked += 1
            obs = nobs
    assert checked > 50


def test_relative_action_from_keys():
    assert relative_action(UP, UP) == STRAIGHT
    assert relative_action(UP, RIGHT) == TURN_RIGHT
    assert relative_action(UP, LEFT) == TURN_LEFT
    assert relative_action(UP, DOWN) == STRAIGHT  # reversing is ignored


def test_rewards_and_outcome():
    env = SplixEnv(size=16, opponents=["random"], reward={"death": -5.0})
    total_deaths = 0
    for seed in range(10):
        env.reset(seed=seed)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(STRAIGHT)  # drives into a wall
            done = term or trunc
        assert info["outcome"] in ("win", "loss", "draw")
        if not info["alive"]:
            total_deaths += 1
            assert r <= -5.0 + 3.0 + 1e-6  # death penalty (maybe a kill on the same step)
    assert total_deaths > 0


def test_shaping_is_potential_based():
    """With gamma = 1 the shaping terms telescope: an episode's shaped return equals the unshaped one."""
    base = SplixEnv(size=16, opponents=["loop"], reward={"gamma": 1.0})
    shaped = SplixEnv(size=16, opponents=["loop"], reward={"gamma": 1.0, "shaping": 1.0})
    for seed in range(5):
        rets = []
        for env in (base, shaped):
            env.reset(seed=seed)
            bot = make_opponent("loop")
            bot.reset(env.game, 0, np.random.default_rng(seed))
            ret, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(bot.act(env.game, 0))
                ret += r
                done = term or trunc
            rets.append((ret, term))
        (r0, t0), (r1, t1) = rets
        if t0:  # terminal potential is 0, so the sums match exactly
            assert r0 == pytest.approx(r1, abs=1e-6)


def test_bots_beat_turtling():
    """Regression: an agent that circles inside its start square must not beat the rule-based bots."""
    for opp in ("loop", "chaser"):
        env = SplixEnv(size=20, opponents=[opp])
        wins = 0
        for seed in range(20):
            env.reset(seed=seed)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(TURN_RIGHT)
                done = term or trunc
            wins += info["outcome"] == "win"
        assert wins <= 4, f"turtling beat {opp} {wins}/20 times"


def test_bot_strength_order():
    kw = dict(size=20, max_steps=300)
    res = summarize(
        evaluate(BotActor("loop"), kw, ["random_safe"], 20) + evaluate(BotActor("random_safe"), kw, ["loop"], 20)
    )
    by = {(r["actor"], r["opponent"]): r["win_rate"] for r in res}
    assert by[("loop", "random_safe")] > 0.7
    assert by[("random_safe", "loop")] < 0.3


def test_dyna_save_load_and_freeze(tmp_path):
    env = SplixEnv(size=16)
    obs, _ = env.reset(seed=0)
    agent = DynaAgent(env.obs_config, env.observation_space["grid"].shape, env.observation_space["vec"].shape[0], DynaConfig())
    path = tmp_path / "agent.pt"
    agent.save(str(path))
    loaded = load_agent(str(path))
    assert loaded.act(obs) == agent.act(obs)
    # play as a frozen opponent from seat 1, and through the evaluator
    env2 = SplixEnv(size=16, opponents=[f"frozen:{path}"])
    env2.reset(seed=1)
    for _ in range(20):
        env2.step(0)
    rows = evaluate(AgentActor(loaded), dict(size=16, max_steps=50), ["loop", f"frozen:{path}"], 2)
    assert len(rows) == 4


def test_agent_with_other_view_settings_still_plays():
    env = SplixEnv(size=16, view_radius=4)
    agent = DynaAgent(ObsConfig(view_radius=6), (6, 13, 13), 11)
    rows = evaluate(AgentActor(agent), dict(size=16, max_steps=30, view_radius=4), ["loop"], 1)
    assert rows[0]["length"] > 0


def test_live_trail_corners_and_snapshot():
    """Live client: trail corners come from state updates; snapshot matches the server layout."""
    import struct

    from splixrl.live import LiveState, raster

    assert raster([(2, 2), (2, 5), (4, 5)]) == [(2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (4, 5)]
    st = LiveState()
    st.handle(bytes([12]) + struct.pack(">H", 12), 0.0)  # map size 12 (playable 1..10)
    st.handle(bytes([3]) + struct.pack(">HHHHBBB", 0, 0, 12, 12, 0, 0, 1), 0.0)  # all wall
    st.handle(bytes([3]) + struct.pack(">HHHHBBB", 1, 1, 10, 10, 1, 0, 1), 0.0)  # inner empty
    st.handle(bytes([15]) + struct.pack(">HB", 0, 4), 0.0)  # my skin 4 -> tiles 6
    st.handle(bytes([3]) + struct.pack(">HHHHBBB", 2, 2, 3, 3, 6, 0, 0), 0.0)  # my 3x3 land
    st.handle(bytes([2]) + struct.pack(">HHHBB", 5, 3, 0, 0, 1), 0.0)  # me at (5,3) moving right
    st.handle(bytes([4]) + struct.pack(">HHH", 0, 5, 3), 0.0)  # trail starts at (5,3)
    st.handle(bytes([2]) + struct.pack(">HHHBB", 7, 3, 0, 1, 1), 0.0)  # turned down at (7,3)
    assert st.players[0].trail == [(5, 3), (7, 3)]
    g = st.snapshot((7, 4), 2, 0.0, 10, 500)  # head at (7,4) heading down
    assert g.size == 10
    assert g.owner[1:4, 1:4].tolist() == [[0] * 3] * 3  # server (2..4, 2..4) -> rows/cols 1..3
    for x, y in [(5, 3), (6, 3), (7, 3), (7, 4)]:
        assert g.trail[y - 1, x - 1] == 0
    assert g.players[0].pos == (3, 6)


def test_lag_delays_turns_by_one_tile():
    env = SplixEnv(size=16, opponents=["random_safe"], lag_prob=1.0)
    env.reset(seed=0)
    p = env.game.players[0]
    h0 = p.heading
    _, _, _, _, info = env.step(TURN_LEFT)  # lagged: goes straight
    assert p.heading == h0 and info["lagged"]
    _, _, _, _, info = env.step(STRAIGHT)  # the delayed left turn lands now
    assert p.heading == (h0 + 3) % 4 and info["lagged"]
    no_lag = SplixEnv(size=16, opponents=["random_safe"])
    no_lag.reset(seed=0)
    no_lag.step(TURN_LEFT)
    assert no_lag.game.players[0].heading == (h0 + 3) % 4


def test_death_per_cell_makes_return_track_final_territory():
    """With death_per_cell = capture_per_cell, dying gives back everything captured."""
    env = SplixEnv(size=16, opponents=["random_safe"],
                   reward={"death": 0.0, "kill": 0.0, "death_per_cell": 0.1, "lost_per_cell": 0.1})
    for seed in range(10):
        env.reset(seed=seed)
        bot = make_opponent("loop")
        bot.reset(env.game, 0, np.random.default_rng(seed))
        start, ret, done = env.game.territory(0), 0.0, False
        while not done:
            _, r, term, trunc, info = env.step(bot.act(env.game, 0))
            ret += r
            done = term or trunc
        if not info["alive"]:
            # captures minus stolen minus what was held at death = minus the starting square
            assert ret == pytest.approx(-0.1 * start, abs=1e-6)


def test_territory_rules_respawn_and_target():
    env = SplixEnv(size=20, opponents=["random"], n_opponents=2, respawn=True, respawn_delay=3, max_steps=300)
    env.reset(seed=0)
    respawned = False
    for _ in range(300):
        _, _, term, trunc, info = env.step(TURN_RIGHT)  # circle at home: random bots die and come back
        assert not (term and info["alive"]), "killing opponents must not end the game with respawn on"
        respawned |= info["opp_respawns"] > 0
        if term or trunc:
            break
    assert respawned and info["outcome"] in ("win", "loss", "draw")
    # reaching the target share of the map is a win
    env = SplixEnv(size=16, opponents=["random_safe"], respawn=True, target_frac=0.1)
    env.reset(seed=1)
    bot = make_opponent("loop")
    bot.reset(env.game, 0, np.random.default_rng(1))
    done = False
    while not done:
        _, _, term, trunc, info = env.step(bot.act(env.game, 0))
        done = term or trunc
    assert info["outcome"] in ("win", "loss", "timeout")
    if info["outcome"] == "win":
        assert info["territory_frac"] >= 0.1


def _rich_cfg(**kw):
    return dict(coarse_factor=3, visible_radius=25, threat_features=True, minimap=True, **kw)


def test_rich_observation_shapes_and_contents():
    env = SplixEnv(size=40, opponents=["loop"], n_opponents=3, **_rich_cfg())
    obs, _ = env.reset(seed=3)
    cfg = env.obs_config
    assert obs["grid"].shape == (6 + 6 + 2, 15, 15) and env.observation_space.contains(obs)
    assert len(obs["vec"]) == 11 + 3 + 4
    # minimap: my position is marked once, and my start square shows up as land
    assert obs["grid"][13].sum() == 1
    assert obs["grid"][12].sum() >= 1
    # coarse wall channel sees the board edge from far away (45x45 window on a 40x40 board)
    assert obs["grid"][6 + 5].sum() > 0
    check_env(env, skip_render_check=True)
    assert cfg.rotating_channels == 12 and cfg.fine_channels == 6


def test_visibility_hides_far_opponents():
    env = SplixEnv(size=60, opponents=["random_safe"], visible_radius=5)
    env.reset(seed=0)
    me, opp = env.game.players
    me.pos, opp.pos = (30, 30), (30, 50)  # 20 tiles away
    obs = build_obs(env.game, 0, env.obs_config)
    assert obs["vec"][6] == 2.0  # opp_dist = FAR: not visible
    opp.pos = (30, 33)
    obs = build_obs(env.game, 0, env.obs_config)
    assert obs["vec"][6] < 1.0


def test_threat_features_see_enemy_trail_on_my_land():
    env = SplixEnv(size=20, opponents=["random_safe"], threat_features=True)
    env.reset(seed=0)
    me = env.game.players[0]
    r, c = me.pos
    base = build_obs(env.game, 0, env.obs_config)["vec"]
    env.game.trail[r - 1, c - 1] = 1  # an enemy trail cell inside my start square
    obs = build_obs(env.game, 0, env.obs_config)["vec"]
    i = 11 + 1  # enemy_trail_on_my_land
    assert obs[i] > base[i]
    assert obs[i + 1] < 0.2  # enemy trail is 2 tiles from my head


def test_dyna_trains_with_rich_observations(tmp_path):
    env = SplixEnv(size=24, opponents=["loop"], **_rich_cfg())
    obs, _ = env.reset(seed=0)
    agent = DynaAgent(env.obs_config, env.observation_space["grid"].shape, env.observation_space["vec"].shape[0])
    from splixrl.agents.dyna import ReplayBuffer

    buf = ReplayBuffer(256, env.observation_space["grid"].shape, env.observation_space["vec"].shape[0])
    for _ in range(80):
        a = agent.act(obs, deterministic=False, epsilon=0.5)
        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs["grid"], obs["vec"], a, r, nobs["grid"], nobs["vec"], float(term))
        obs = nobs if not (term or trunc) else env.reset()[0]
    agent.q_update(buf.sample(16, "cpu"))
    agent.model_update(buf.sample(16, "cpu"), with_metrics=True)
    path = tmp_path / "a.pt"
    agent.save(str(path))
    assert load_agent(str(path)).obs_config == env.obs_config


def test_live_minimap_marks_you_and_reads_the_leaderboard():
    import struct

    from splixrl.live import LiveState, minimap_image

    st = LiveState()
    st.handle(bytes([12]) + struct.pack(">H", 400), 0.0)
    st.handle(bytes([10]) + struct.pack(">H", 4), 0.0)
    msg = bytearray([11]) + struct.pack(">H", 29)
    msg += struct.pack(">IB", 36022, 6) + b"Turkey"
    msg += struct.pack(">IB", 25, 7) + b"SplixRL"
    st.handle(bytes(msg), 0.0)
    assert st.rank == 4 and st.total_players == 29
    assert st.leaderboard == [("Turkey", 36022), ("SplixRL", 25)]
    img = minimap_image(st, (100, 200), (78, 178, 122, 222))
    assert img.shape == (80 * 3, 80 * 3, 3)
    mx, my = st.minimap_px(100, 200)  # (20, 40) on a 400 map
    assert (mx, my) == (20, 40)
    assert tuple(img[my * 3, mx * 3]) == (255, 255, 255)  # you
    assert tuple(img[my * 3, mx * 3 - 3]) == (168, 86, 255)
    # the camera rectangle is the blue box
    vx, vy = st.minimap_px(78, 178)
    assert tuple(img[vy * 3, vx * 3]) == (110, 168, 214)


def test_web_relay_forwards_the_server_and_drops_keyboard_turns():
    """The splix.io page must see server bytes, and its arrow keys must not override the actor."""
    import asyncio

    import websockets

    from splixrl.live import WebRelay

    got = []

    async def upstream(ws):
        async for message in ws:
            got.append(message)
            if message[:1] == bytes([2]):
                await ws.send(bytes([17]))

    async def run():
        async with websockets.serve(upstream, "127.0.0.1", 0, max_size=None) as server:
            port = server.sockets[0].getsockname()[1]
            relay = WebRelay(f"ws://127.0.0.1:{port}")
            await relay.start()
            try:
                async with websockets.connect(f"ws://127.0.0.1:{relay.port}", max_size=None) as browser:
                    waiting = asyncio.create_task(relay.next_link())
                    await browser.send(bytes([2]) + b"me")
                    await browser.send(bytes([1, 0, 0, 0, 0, 0]))
                    assert await asyncio.wait_for(browser.recv(), 2) == bytes([17])
                    link = await waiting
                    await link.send(bytes([1, 3, 0, 5, 0, 6]))
                    await asyncio.sleep(0.05)
            finally:
                await relay.close()

    asyncio.run(run())
    assert bytes([2]) + b"me" in got
    assert bytes([1, 0, 0, 0, 0, 0]) not in got
    assert bytes([1, 3, 0, 5, 0, 6]) in got


def test_web_turn_hook_reaches_the_page_and_not_the_server():
    """The page is asked to turn; that request is not a second turn to the server."""
    import asyncio
    import time

    import websockets

    from splixrl.live import SPLIX_TURN_HOOK, WebRelay

    got = []

    async def upstream(ws):
        async for message in ws:
            got.append(message)
            if message[:1] == bytes([7]):
                await ws.send(bytes([21]))

    async def run():
        async with websockets.serve(upstream, "127.0.0.1", 0, max_size=None) as server:
            port = server.sockets[0].getsockname()[1]
            relay = WebRelay(f"ws://127.0.0.1:{port}")
            await relay.start()
            try:
                async with websockets.connect(f"ws://127.0.0.1:{relay.port}", max_size=None) as browser:
                    waiting = asyncio.create_task(relay.next_link())
                    await browser.send(bytes([7]))
                    assert await asyncio.wait_for(browser.recv(), 2) == bytes([21])
                    link = await waiting
                    await asyncio.sleep(0.02)
                    assert link._ping_at is None
                    assert 0 < link.rtt < 0.5
                    link.request_turn(time.monotonic() + 0.05, 1)
                    shown = await asyncio.wait_for(browser.recv(), 1)
                    assert shown == bytes([SPLIX_TURN_HOOK, 1])
            finally:
                await relay.close()

    asyncio.run(run())
    assert bytes([SPLIX_TURN_HOOK, 1]) not in got


def test_splix_client_hook_calls_the_page_turn_routine():
    from splixrl.live import patch_splix_client

    src = b"prefix f=new Uint8Array(e.data);f[0]==ki suffix"
    out = patch_splix_client(src)
    assert b"if(f[0]==255)return void Bn(f[1]);" in out


def test_live_minimap_decode():
    import struct

    from splixrl.live import LiveState

    st = LiveState()
    st.handle(bytes([12]) + struct.pack(">H", 100), 0.0)
    bits = np.zeros(20 * 80, np.uint8)
    bits[5 * 80 + 10] = 1  # strip 1, column 5 -> minimap x = 25, y = 10
    st.handle(bytes([14, 1]) + np.packbits(bits, bitorder="little").tobytes(), 0.0)
    assert st.minimap[10, 25] and st.minimap.sum() == 1
    hint = st.land_hint()  # board 98x98; minimap tile (25,10) covers arena x 31..32, y 12..13
    assert hint.shape == (98, 98) and hint.any()
    rows, cols = np.nonzero(hint)
    assert set(rows + 1) <= {12, 13, 14} and set(cols + 1) <= {31, 32, 33}


def test_nstep_returns_stop_at_episode_end():
    from splixrl.agents.dyna import ReplayBuffer

    buf = ReplayBuffer(16, (1, 1, 1), 1, seed=0)
    g = np.zeros((1, 1, 1))
    # episode A: rewards 1, 2, 3 (ends, terminal); episode B: 10, 20 ...
    for r, done, end in [(1, 0, 0), (2, 0, 0), (3, 1, 1), (10, 0, 0), (20, 0, 0), (30, 0, 0)]:
        buf.add(g, [r], 0, r, g, [r], done, end=end)
    gamma = 0.5
    grid, vec, a, ret, ng, nvec, done, disc = buf.sample_nstep(200, 3, gamma, "cpu")
    for v, R, d, dc, nv in zip(vec[:, 0], ret, done, disc, nvec[:, 0]):
        v = float(v)
        if v == 1:  # 1 + .5*2 + .25*3 = 2.75, then the episode ended (terminal)
            assert R == pytest.approx(2.75) and d == 1 and dc == pytest.approx(0.125)
        if v == 2:  # 2 + .5*3, stops at the end of episode A
            assert R == pytest.approx(3.5) and nv == 3 and dc == pytest.approx(0.25)
        if v == 10:  # 10 + .5*20 + .25*30 = 27.5, bootstraps from the 3rd step's next state
            assert R == pytest.approx(27.5) and d == 0 and nv == 30
    assert 30 not in vec[:, 0].tolist()  # the last steps can't start a full 3-step window


def test_stability_options_train_and_reload(tmp_path):
    from splixrl.agents.dyna import ReplayBuffer

    env = SplixEnv(size=16, opponents=["loop"])
    obs, _ = env.reset(seed=0)
    cfg = DynaConfig(n_step=3, dueling=True, tau=0.01, model_gate=0.5)
    shp, nv = env.observation_space["grid"].shape, env.observation_space["vec"].shape[0]
    agent = DynaAgent(env.obs_config, shp, nv, cfg)
    buf = ReplayBuffer(256, shp, nv)
    for _ in range(100):
        a = agent.act(obs, deterministic=False, epsilon=0.5)
        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs["grid"], obs["vec"], a, r, nobs["grid"], nobs["vec"], float(term), end=term or trunc)
        obs = nobs if not (term or trunc) else env.reset()[0]
    before = [p.clone() for p in agent.q_target.parameters()]
    agent.q_update(buf.sample_nstep(16, 3, cfg.gamma, "cpu"))
    after = list(agent.q_target.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))  # soft update moved the target
    path = tmp_path / "d.pt"
    agent.save(str(path))
    loaded = load_agent(str(path))
    assert loaded.cfg.dueling and loaded.act(obs) == agent.act(obs)


def test_enemy_heading_and_size_features():
    env = SplixEnv(size=30, opponents=["random_safe"], enemy_heading=True, size_features=True,
                   coarse_factor=3, minimap=True, visible_radius=25)
    obs, _ = env.reset(seed=2)
    cfg = env.obs_config
    assert cfg.board_channels[5] == "opp_next" and cfg.fine_channels == 7 and cfg.rotating_channels == 14
    assert obs["grid"].shape == (7 + 7 + 2, 15, 15)
    me, opp = env.game.players
    me.pos, me.heading = (15, 15), 0
    opp.pos, opp.heading = (15, 19), 3  # 4 tiles to my right, facing left (towards me)
    g = build_obs(env.game, 0, cfg)["grid"]
    assert g[4, 7, 11] == 1 and g[5, 7, 10] == 1  # head, and the tile it moves into next
    v = dict(zip(cfg.vec_features, build_obs(env.game, 0, cfg)["vec"]))
    assert v["my_size"] == pytest.approx(np.log1p(25) / 8) and v["enemy_size"] == pytest.approx(np.log1p(25) / 8)
    env.game.owner[0:5, 0:20] = 1  # the enemy grows
    v = dict(zip(cfg.vec_features, build_obs(env.game, 0, cfg)["vec"]))
    assert v["size_advantage"] < 0


def test_reclaim_features_find_holes_left_by_a_dead_enemy():
    env = SplixEnv(size=30, opponents=["random_safe"], reclaim_features=True)
    env.reset(seed=4)
    g = env.game
    me = g.players[0]
    me.pos, me.heading = (15, 15), 0
    g.owner[g.owner == 0] = EMPTY
    g.owner[10:21, 10:21] = 0  # an 11x11 block of my land
    v = dict(zip(env.obs_config.vec_features, build_obs(g, 0, env.obs_config)["vec"]))
    assert v["cheap_land"] == 0
    g.owner[12:15, 17:20] = EMPTY  # an enemy that had eaten a 3x3 hole died: the hole is empty now
    v = dict(zip(env.obs_config.vec_features, build_obs(g, 0, env.obs_config)["vec"]))
    assert v["cheap_land"] == pytest.approx(9 / 32)
    assert v["cheap_dist"] == pytest.approx(3 / 32)  # nearest hole cell (14,17) from (15,15)
    assert v["cheap_forward"] > 0 and v["cheap_right"] > 0  # ahead and to the right


def test_episode_randomization_and_bot_jitter():
    env = SplixEnv(size=30, opponents=["loop?jitter=true"], n_opponents=2, size_range=(24, 48),
                   n_opponents_range=(1, 5), lag_range=(0.0, 0.1), respawn=True, coarse_factor=3, minimap=True)
    seen = set()
    shape = None
    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        seen.add((env.game.size, len(env.game.players) - 1))
        assert 24 <= env.game.size <= 48 and 1 <= len(env.game.players) - 1 <= 5 and 0 <= env.lag_prob <= 0.1
        assert shape is None or obs["grid"].shape == shape  # input shape never changes
        shape = obs["grid"].shape
        env.step(0)
    assert len(seen) > 4
    sizes = {(b.max_out, b.danger_dist) for b in env.opponents}
    bots = [make_opponent("loop?jitter=true") for _ in range(10)]
    for i, b in enumerate(bots):
        b.reset(env.game, 1, np.random.default_rng(i))
    assert len({(b.max_out, b.danger_dist, b.keep_away) for b in bots}) > 3


def test_escape_margin_sign():
    env = SplixEnv(size=40, opponents=["random_safe"], escape_features=True)
    env.reset(seed=5)
    g = env.game
    me, opp = g.players
    g.owner.fill(EMPTY)
    g.owner[18:23, 5:10] = 0  # my land, cols 5-9
    me.pos, me.heading = (20, 15), 1  # 6 tiles out to the right of my land
    me.trail = [(20, c) for c in range(10, 16)]
    for cell in me.trail:
        g.trail[cell] = 0
    g.owner[0:3, 30:35] = 1
    opp.pos = (2, 32)  # far away: safe
    m = build_obs(g, 0, env.obs_config)["vec"][-1]
    assert m > 0
    opp.pos = (22, 14)  # 2 tiles from my trail, I'm 6 from home: cut off
    m = build_obs(g, 0, env.obs_config)["vec"][-1]
    assert m < 0


def test_big_crowd_on_small_board_shrinks_instead_of_crashing():
    env = SplixEnv(size=20, opponents=["loop"], n_opponents=4, size_range=(20, 22), n_opponents_range=(10, 12))
    env.reset(seed=0)
    assert 1 <= env.n_opponents < 10


def test_pause_rules():
    from splixrl.game import PAUSE

    env = SplixEnv(size=20, opponents=["random_safe"], allow_pause=True, max_pause=3)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 4 and env.obs_config.vec_features[-1] == "paused"
    me = env.game.players[0]
    pos, h = me.pos, me.heading
    obs, *_ = env.step(PAUSE)
    assert me.pos == pos and me.paused and obs["vec"][-1] == 1.0
    env.step(PAUSE)
    env.step(PAUSE)
    env.step(PAUSE)  # 4th pause in a row: over max_pause, goes straight instead
    assert me.pos != pos and not me.paused and me.heading == h
    # a paused player with a trail out can be cut
    g = env.game
    me.trail = [me.pos]
    g.trail[me.pos] = 0
    me.paused = True
    opp = g.players[1]
    r, c = me.pos
    opp.pos, opp.heading = (r, c - 1) if c > 0 else (r, c + 1), 1 if c > 0 else 3
    g.owner[opp.pos] = 1
    g.step([PAUSE, 0])
    assert not me.alive and me.death_cause == "trail_cut"
    # without allow_pause the action is refused
    plain = SplixEnv(size=16)
    plain.reset(seed=0)
    with pytest.raises(ValueError):
        plain.step(PAUSE)


def test_dyna_with_pause_and_warm_start_from_three_actions(tmp_path):
    from splixrl.agents.dyna import ReplayBuffer, align
    from splixrl.game import PAUSE
    from splixrl.train_dyna import load_extended

    old_env = SplixEnv(size=16, opponents=["loop"], escape_features=True)
    new_env = SplixEnv(size=16, opponents=["loop"], escape_features=True, allow_pause=True)
    old = DynaAgent(old_env.obs_config, old_env.observation_space["grid"].shape, old_env.observation_space["vec"].shape[0])
    cfg = DynaConfig(n_actions=4)
    new = DynaAgent(new_env.obs_config, new_env.observation_space["grid"].shape, new_env.observation_space["vec"].shape[0], cfg)
    load_extended(new.q, old.q.state_dict())
    obs_new, _ = new_env.reset(seed=1)
    obs_old = {"grid": obs_new["grid"], "vec": obs_new["vec"][:-1]}
    g_new, v_new = torch.tensor(obs_new["grid"])[None], torch.tensor(obs_new["vec"])[None]
    with torch.no_grad():
        q_old = old.q(g_new, torch.tensor(obs_old["vec"])[None])
        q_new = new.q(g_new, v_new)
    assert torch.allclose(q_old, q_new[:, :3], atol=1e-5)  # same values for the old actions
    # paused rows of align keep the view
    grid = torch.rand(2, *new_env.observation_space["grid"].shape).round()
    out = align(grid, torch.tensor([PAUSE, 0]), new_env.obs_config)
    assert torch.equal(out[0], grid[0])
    # a few updates with 4 actions work
    buf = ReplayBuffer(128, new_env.observation_space["grid"].shape, new_env.observation_space["vec"].shape[0])
    obs = obs_new
    for _ in range(60):
        a = new.act(obs, deterministic=False, epsilon=0.7)
        nobs, r, term, trunc, _ = new_env.step(a)
        buf.add(obs["grid"], obs["vec"], a, r, nobs["grid"], nobs["vec"], float(term), end=term or trunc)
        obs = nobs if not (term or trunc) else new_env.reset()[0]
    new.q_update(buf.sample(16, "cpu"))
    new.model_update(buf.sample(16, "cpu"), with_metrics=True)
    new.save(str(tmp_path / "p.pt"))
    assert load_agent(str(tmp_path / "p.pt")).cfg.n_actions == 4


def test_rank_bonus_rewards_finishing_ahead():
    from splixrl.rewards import RewardConfig

    rc = RewardConfig.preset("score_rank")
    env = SplixEnv(size=20, opponents=["random_safe"], n_opponents=2, max_steps=3, reward=rc)
    env.reset(seed=0)
    g = env.game
    g.owner[g.owner == 1] = EMPTY  # opponent 1 has almost nothing...
    g.owner[0, 0] = 1
    done, ret = False, 0.0
    while not done:
        _, r, term, trunc, info = env.step(TURN_RIGHT)  # circle at home
        ret += r
        done = term or trunc
    assert info["alive"] and info["rank_share"] >= 0.5
    assert ret >= rc.rank_bonus * 0.5 - 1e-6


def test_memory_channels_remember_enemies_and_see_growth():
    env = SplixEnv(size=60, opponents=["random_safe"], minimap=True, visible_radius=10, memory=50)
    env.reset(seed=1)
    g = env.game
    cfg = env.obs_config
    ch = cfg.channels
    i_mem, i_grow = ch.index("map_enemy_memory"), ch.index("map_growth")
    me, opp = g.players
    me.pos, opp.pos = (30, 30), (30, 36)  # visible
    build_obs(g, 0, cfg)
    opp.pos = (30, 55)  # walks out of view
    for _ in range(5):
        g.t += 1
        grid = build_obs(g, 0, cfg)["grid"]
    assert grid[i_mem].max() > 0.8  # still remembered where it was last seen
    g.t += 60
    grid = build_obs(g, 0, cfg)["grid"]
    assert grid[i_mem].max() == 0  # forgotten after the horizon
    g.owner[0:20, 0:20] = 1  # someone grows a lot far away
    g.t += 20
    grid = build_obs(g, 0, cfg)["grid"]
    assert grid[i_grow].sum() > 0 and grid[i_grow][0:4, 0:4].sum() > 0


def test_stalker_hunts_from_far_away():
    env = SplixEnv(size=60, opponents=["stalker"], n_opponents=1)
    env.reset(seed=2)
    g = env.game
    me, st = g.players
    me.trail = [(10, c) for c in range(10, 20)]
    for cell in me.trail:
        g.trail[cell] = 0
    st.pos = (50, 50)
    d0 = abs(st.pos[0] - 10) + abs(st.pos[1] - 15)
    bot = env.opponents[0]
    for _ in range(8):
        g.step([0 if g.is_safe(0, 0) else 1, bot.act(g, 1)])
        if not me.alive:
            break
    d1 = min(abs(st.pos[0] - r) + abs(st.pos[1] - c) for r, c in me.trail) if me.trail else 0
    assert d1 < d0 - 5  # it headed for my trail from ~75 tiles away


def test_memory_notices_land_being_stolen():
    env = SplixEnv(size=40, opponents=["random_safe"], minimap=True, memory=40)
    env.reset(seed=3)
    g, cfg = env.game, env.obs_config
    me = g.players[0]
    g.owner[g.owner == 0] = EMPTY
    g.owner[10:20, 10:20] = 0  # my land
    me.pos, me.heading = (15, 15), 0
    g.obs_memory = {}  # forget the snapshot taken at reset, before this layout existed
    build_obs(g, 0, cfg)
    g.owner[10:13, 17:20] = 1  # an enemy takes a 3x3 corner
    g.t += 15
    o = build_obs(g, 0, cfg)
    v = dict(zip(cfg.vec_features, o["vec"]))
    assert v["land_lost"] == pytest.approx(9 / 32)
    assert v["lost_forward"] > 0 and v["lost_right"] > 0  # ahead and to the right of me
    assert o["grid"][cfg.channels.index("map_my_land_lost")].sum() >= 1


def test_curriculum_warm_start_matches_inputs_by_name():
    """A basic 20x20 agent grown to the full observation set computes the same
    Q-values for its old actions (new inputs start at zero weight)."""
    from splixrl.train_dyna import load_by_name

    basic = SplixEnv(size=20, opponents=["loop"])
    rich = SplixEnv(size=20, opponents=["loop"], coarse_factor=3, visible_radius=25, threat_features=True, minimap=True,
                    memory=100, enemy_heading=True, size_features=True, reclaim_features=True, escape_features=True,
                    allow_pause=True)
    old = DynaAgent(basic.obs_config, basic.observation_space["grid"].shape, basic.observation_space["vec"].shape[0])
    new = DynaAgent(rich.obs_config, rich.observation_space["grid"].shape, rich.observation_space["vec"].shape[0],
                    DynaConfig(n_actions=4))
    notes = load_by_name(new.q, old.q.state_dict(), basic.obs_config, rich.obs_config)
    assert any("6/" in n for n in notes) and any("11/" in n for n in notes)
    basic.reset(seed=3)
    rich.game = basic.game  # same board for both views
    for _ in range(5):
        o_old = build_obs(basic.game, 0, basic.obs_config)
        o_new = build_obs(basic.game, 0, rich.obs_config)
        with torch.no_grad():
            q_old = old.q(*[torch.tensor(o_old[k])[None] for k in ("grid", "vec")])
            q_new = new.q(*[torch.tensor(o_new[k])[None] for k in ("grid", "vec")])
        assert torch.allclose(q_old, q_new[:, :3], atol=1e-5)
        basic.step(0 if basic.game.is_safe(0, 0) else 1)


def test_turn_and_pause_costs():
    env = SplixEnv(size=20, opponents=["random_safe"], allow_pause=True,
                   reward={"turn_cost": 0.5, "pause_cost": 0.25, "capture_per_cell": 0.0, "death": 0.0, "kill": 0.0})
    env.reset(seed=0)
    from splixrl.game import PAUSE
    _, r_turn, *_ = env.step(TURN_LEFT)
    _, r_pause, *_ = env.step(PAUSE)
    _, r_straight, *_ = env.step(STRAIGHT)
    assert r_turn == pytest.approx(-0.5) and r_pause == pytest.approx(-0.25) and r_straight == pytest.approx(0.0)


def test_pause_is_free_only_near_enemies():
    from splixrl.game import PAUSE

    env = SplixEnv(size=40, opponents=["random_safe"], allow_pause=True,
                   reward={"pause_cost": 0.5, "pause_free_radius": 6, "capture_per_cell": 0.0, "death": 0.0, "kill": 0.0})
    env.reset(seed=0)
    me, opp = env.game.players
    opp.pos = (me.pos[0] + 20 if me.pos[0] < 20 else me.pos[0] - 20, me.pos[1])
    _, r_far, *_ = env.step(PAUSE)
    opp.pos = (me.pos[0], me.pos[1] + 3 if me.pos[1] < 36 else me.pos[1] - 3)
    _, r_near, *_ = env.step(PAUSE)
    assert r_far == pytest.approx(-0.5) and r_near == pytest.approx(0.0)


def _options_agent(env_kw):
    from splixrl.options import N_OPTIONS

    env = SplixEnv(**env_kw)
    agent = DynaAgent(env.obs_config, env.observation_space["grid"].shape, env.observation_space["vec"].shape[0],
                      DynaConfig(n_actions=N_OPTIONS))
    agent.options = {"safety": 2.0, "lag_slack": 30.0}
    return env, agent


def test_options_env_runs_whole_plans_and_discounts():
    from splixrl.options import N_OPTIONS, OptionsEnv

    env = OptionsEnv(SplixEnv(size=30, opponents=["loop"], n_opponents=2, respawn=True, allow_pause=True,
                              max_steps=800, lag_feature=True, lag_prob=0.05), gamma=0.9)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == N_OPTIONS and obs["vec"][-1] == pytest.approx(0.5)  # lag 0.05, x10
    for opt in range(N_OPTIONS):
        obs, r, term, trunc, info = env.step(opt)
        k = info["option_steps"]
        assert 1 <= k <= 200 and info["option_disc"] == pytest.approx(0.9 ** k)
        if term or trunc:
            env.reset(seed=opt)


def test_options_agent_save_load_evaluate_and_freeze(tmp_path):
    kw = dict(size=24, opponents=["loop"], allow_pause=True, lag_feature=True)
    env, agent = _options_agent(kw)
    path = tmp_path / "opt.pt"
    agent.save(str(path))
    from splixrl.evaluation import make_actor
    from splixrl.options import OptionsActor

    actor = make_actor(str(path))
    assert isinstance(actor, OptionsActor)
    rows = evaluate(actor, dict(size=24, max_steps=200, lag_feature=True), ["loop", f"frozen:{path}"], 2)
    assert len(rows) == 4 and all(r["length"] > 0 for r in rows)
    assert len(actor.chosen) > 2  # it picked several options per game, not one per tile
