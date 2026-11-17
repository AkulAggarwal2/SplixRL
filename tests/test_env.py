import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from splixrl import SplixEnv
from splixrl.agents import load_agent
from splixrl.agents.dyna import DynaAgent, DynaConfig, align
from splixrl.evaluation import AgentActor, BotActor, evaluate, summarize
from splixrl.game import DOWN, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP
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
                pred = align(torch.tensor(obs["grid"])[None], torch.tensor([a]), "local")[0].numpy()
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
    env = SplixEnv(size=16, opponents=["random_safe"], reward={"death": 0.0, "kill": 0.0, "death_per_cell": 0.1})
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
            assert ret == pytest.approx(-0.1 * start, abs=1e-6)  # only the starting square is "lost"


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
