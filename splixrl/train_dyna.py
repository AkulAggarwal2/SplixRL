"""Train the Dyna agent.

    python -m splixrl.train_dyna --run-dir runs/round0/dyna/seed0 --seed 0

Writes to the run directory:
    config.json   everything needed to reproduce the run
    train.csv     one row per training episode
    metrics.csv   losses, world-model accuracy, speed (every --log-every steps)
    eval.csv      greedy evaluation vs the fixed opponents (every --eval-every steps)
    agent.pt      final agent (load_agent / "frozen:<path>" / watch / evaluate)
    checkpoints/  agent at every evaluation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agents.dyna import DynaAgent, DynaConfig, ReplayBuffer, pick_device
from .env import SplixEnv
from .evaluation import CsvLogger, TrainingEvaluator
from .opponents import DEFAULT_POOL
from .rewards import RewardConfig

TRAIN_FIELDS = ("step", "episode", "return", "length", "territory_frac", "peak_territory", "total_captured", "kills", "outcome", "death_cause", "opponent", "epsilon")
METRIC_FIELDS = ("step", "q_loss", "model_loss", "model_grid_err", "model_changed_err", "model_reward_mse", "model_done_acc", "imagined", "q_updates", "steps_per_sec")


def add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    for f in dataclasses.fields(cls):
        flag = "--" + prefix + f.name.replace("_", "-")
        typ = type(f.default)
        parser.add_argument(flag, type=typ, default=None, help=f"default {f.default}")


def dataclass_from_args(cls, args, prefix: str = ""):
    kw = {}
    for f in dataclasses.fields(cls):
        v = getattr(args, (prefix + f.name).replace("-", "_"))
        if v is not None:
            kw[f.name] = v
    return cls(**kw)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000, help="real environment steps")
    p.add_argument("--device", default="auto", help="auto = cuda, else mps (Apple GPU), else cpu")
    # env
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--spawn-radius", type=int, default=2)
    p.add_argument("--n-opponents", type=int, default=1)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_POOL), help="training pool (bots or frozen:<path>)")
    p.add_argument("--obs-mode", default="local", choices=["local", "global"])
    p.add_argument("--view-radius", type=int, default=7)
    p.add_argument("--lag-prob", type=float, default=0.0, help="chance a turn lands one tile late (live-server lag)")
    # eval
    p.add_argument("--eval-opponents", nargs="+", default=list(DEFAULT_POOL))
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--log-every", type=int, default=2_000)
    add_dataclass_args(p, DynaConfig)
    add_dataclass_args(p, RewardConfig, prefix="reward-")
    return p


def linear_eps(step: int, total: int, cfg: DynaConfig) -> float:
    frac = min(1.0, step / max(1, cfg.eps_fraction * total))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(args: argparse.Namespace) -> Path:
    run = Path(args.run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = dataclass_from_args(DynaConfig, args)
    reward = dataclass_from_args(RewardConfig, args, prefix="reward_")
    reward.gamma = cfg.gamma  # shaping is only policy-invariant with the learner's discount

    env_kwargs = dict(
        size=args.size,
        opponents=args.opponents,
        n_opponents=args.n_opponents,
        max_steps=args.max_steps,
        spawn_radius=args.spawn_radius,
        obs_mode=args.obs_mode,
        view_radius=args.view_radius,
        reward=reward.to_dict(),
        lag_prob=args.lag_prob,
    )
    (run / "config.json").write_text(
        json.dumps(
            {"algo": "dyna", "seed": args.seed, "steps": args.steps, "env": env_kwargs, "dyna": cfg.to_dict(),
             "eval": {"opponents": args.eval_opponents, "episodes": args.eval_episodes, "every": args.eval_every}},
            indent=2,
        )
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    env = SplixEnv(**env_kwargs)
    grid_shape = env.observation_space["grid"].shape
    n_vec = env.observation_space["vec"].shape[0]
    agent = DynaAgent(env.obs_config, grid_shape, n_vec, cfg, device=pick_device(args.device), seed=args.seed)
    real = ReplayBuffer(cfg.buffer_size, grid_shape, n_vec, seed=args.seed)
    imagined = ReplayBuffer(cfg.model_buffer_size, grid_shape, n_vec, seed=args.seed + 1)

    train_log = CsvLogger(run / "train.csv", TRAIN_FIELDS)
    metric_log = CsvLogger(run / "metrics.csv", METRIC_FIELDS)
    evaluator = TrainingEvaluator(run, env_kwargs, args.eval_opponents, args.eval_episodes)

    best = {"score": -1.0}

    def run_eval(step: int) -> None:
        agent.sync_actor()
        summary = evaluator(agent, step)
        score = float(np.mean([r["win_rate"] for r in summary]))
        if score > best["score"]:  # keep the checkpoint with the best mean win rate
            best["score"] = score
            agent.save(str(run / "best.pt"), extra={"env": env_kwargs, "step": step, "mean_win_rate": score})
        agent.save(str(run / "checkpoints" / f"agent_{step}.pt"), extra={"env": env_kwargs, "step": step})

    obs, info = env.reset(seed=args.seed)
    ep_ret, episode = 0.0, 0
    q_loss = float("nan")
    model_stats: dict = {}
    t0, last_t, last_step = time.time(), time.time(), 0
    run_eval(0)
    for step in range(1, args.steps + 1):
        eps = linear_eps(step, args.steps, cfg)
        action = agent.act(obs, deterministic=False, epsilon=eps)
        nobs, r, term, trunc, info = env.step(action)
        real.add(obs["grid"], obs["vec"], action, r, nobs["grid"], nobs["vec"], float(term))
        ep_ret += r
        if term or trunc:
            train_log.log({
                "step": step, "episode": episode, "return": ep_ret, "length": info["steps"],
                "territory_frac": info["territory_frac"], "peak_territory": info["peak_territory"],
                "total_captured": info["total_captured"], "kills": info["kills"], "outcome": info["outcome"],
                "death_cause": info["death_cause"] or "", "opponent": "+".join(info["opponents"]), "epsilon": eps,
            })
            episode += 1
            ep_ret = 0.0
            obs, info = env.reset()
        else:
            obs = nobs

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            loss = agent.q_update(real.sample(cfg.batch_size, agent.device))
            want_metrics = step % args.log_every < cfg.train_freq
            if want_metrics:
                q_loss = float(loss)
            if agent.planning:
                stats = agent.model_update(real.sample(cfg.model_batch_size, agent.device), with_metrics=want_metrics)
                model_stats = stats or model_stats
                if step >= cfg.model_warmup and imagined.size >= cfg.batch_size:
                    for _ in range(cfg.planning_updates):
                        agent.q_update(imagined.sample(cfg.batch_size, agent.device))
        if agent.planning and step >= cfg.model_warmup and step % cfg.rollout_freq == 0:
            agent.imagine(real, imagined, eps)

        if step % args.log_every == 0:
            now = time.time()
            metric_log.log({"step": step, "q_loss": q_loss, **model_stats, "imagined": imagined.size,
                            "q_updates": agent.q_updates, "steps_per_sec": (step - last_step) / (now - last_t)})
            last_t, last_step = now, step
        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step)

    agent.save(str(run / "agent.pt"), extra={"env": env_kwargs, "step": args.steps})
    for log in (train_log, metric_log, evaluator):
        log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {run / 'agent.pt'}")
    return run / "agent.pt"


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
