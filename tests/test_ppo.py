import torch

from splixrl import SplixEnv
from splixrl.agents import load_agent
from splixrl.agents.ppo import PPOAgent
from splixrl.train_ppo import build_parser, train


def make_agent():
    env = SplixEnv(size=16, opponents=["random_safe"])
    obs, _ = env.reset(seed=0)
    agent = PPOAgent(env.obs_config, obs["grid"].shape, obs["vec"].shape[0])
    return env, agent


def test_save_load_roundtrip(tmp_path):
    "Save -> load_agent gives back a PPOAgent with identical weights and actions"
    env, agent = make_agent()
    obs, _ = env.reset(seed=0)
    path = tmp_path / "ppo.pt"
    agent.save(str(path))
    loaded = load_agent(str(path))
    assert isinstance(loaded, PPOAgent)
    for k, v in agent.net.state_dict().items():
        assert torch.equal(v, loaded.net.state_dict()[k])
    assert loaded.act(obs) == agent.act(obs)


def test_ppo_as_frozen_opponent(tmp_path):
    "A saved PPO agent can play from an opponent seat (\"frozen:<path>\")"
    _, agent = make_agent()
    path = tmp_path / "ppo.pt"
    agent.save(str(path))
    env = SplixEnv(size=16, opponents=[f"frozen:{path}"])
    env.reset(seed=0)
    for _ in range(5):
        env.step(0)


def test_train_smoke(tmp_path):
    "A tiny training run finishes and writes every output file"
    run = tmp_path / "run"
    train(build_parser().parse_args([
        "--run-dir", str(run), "--steps", "512", "--num-envs", "2", "--rollout-steps", "64",
        "--minibatch-size", "64", "--eval-every", "256", "--eval-episodes", "1",
    ]))
    for f in ("agent.pt", "best.pt", "eval.csv", "metrics.csv", "train.csv"):
        assert (run / f).exists()
