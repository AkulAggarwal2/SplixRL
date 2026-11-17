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
