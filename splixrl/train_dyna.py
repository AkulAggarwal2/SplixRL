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
import functools
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agents.dyna import DynaAgent, DynaConfig, MultiBuffer, ReplayBuffer, pick_device
from .env import SplixEnv
from .evaluation import CsvLogger, TrainingEvaluator
from .obs import ObsConfig
from .opponents import DEFAULT_POOL, opponent_label
from .rewards import RewardConfig

TRAIN_FIELDS = ("step", "episode", "return", "length", "territory_frac", "peak_territory", "total_captured", "kills", "outcome", "death_cause", "opponent", "epsilon")
METRIC_FIELDS = ("step", "q_loss", "skipped_updates", "model_loss", "model_grid_err", "model_changed_err", "model_reward_mse", "model_done_acc", "imagined", "q_updates", "steps_per_sec")


def add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    for f in dataclasses.fields(cls):
        flag = "--" + prefix + f.name.replace("_", "-")
        typ = type(f.default)
        if typ is bool:  # argparse's bool("False") is True
            typ = lambda v: str(v).lower() in ("1", "true", "yes")  # noqa: E731
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
    p.add_argument("--num-envs", type=int, default=1, help="games played at once in worker processes")
    # env
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--spawn-radius", type=int, default=2)
    p.add_argument("--n-opponents", type=int, default=1)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_POOL), help="training pool (bots or frozen:<path>)")
    p.add_argument("--obs-mode", default="local", choices=["local", "global"])
    p.add_argument("--view-radius", type=int, default=7)
    p.add_argument("--coarse-factor", type=int, default=0, help="add a k-times-wider pooled view (3 = 45x45 tiles)")
    p.add_argument("--visible-radius", type=int, default=None, help="hide opponents beyond this many tiles (server: 25)")
    p.add_argument("--threat-features", action="store_true", help="add enemy-near-my-land features")
    p.add_argument("--minimap", action="store_true", help="add the real game's minimap channels")
    p.add_argument("--enemy-heading", action="store_true", help="mark the tile in front of each enemy head")
    p.add_argument("--size-features", action="store_true", help="add my-size vs nearest-enemy-size features")
    p.add_argument("--reclaim-features", action="store_true", help="add features for cheap land to take back")
    p.add_argument("--escape-features", action="store_true", help="add the escape margin (enemy reach - my way home)")
    p.add_argument("--allow-pause", action="store_true", help="a 4th action: stop moving, as on the real server")
    p.add_argument("--memory", type=int, default=0, help="steps of enemy-memory and minimap-growth channels (needs --minimap)")
    p.add_argument("--max-pause", type=int, default=20, help="most steps in a row the agent may stay paused")
    p.add_argument("--respawn", action="store_true", help="territory rules: dead opponents come back")
    p.add_argument("--respawn-delay", type=int, default=10)
    p.add_argument("--target-frac", type=float, default=None, help="territory rules: win at this share of the map")
    p.add_argument("--lag-prob", type=float, default=0.0, help="chance a turn lands one tile late (live-server lag)")
    p.add_argument("--size-range", type=int, nargs=2, metavar=("LO", "HI"), help="new board size every episode")
    p.add_argument("--n-opponents-range", type=int, nargs=2, metavar=("LO", "HI"), help="new opponent count every episode")
    p.add_argument("--lag-range", type=float, nargs=2, metavar=("LO", "HI"), help="new lag probability every episode")
    # league / curriculum
    p.add_argument("--league-every", type=int, default=0, help="every N steps a frozen copy of the agent joins the opponent pool")
    p.add_argument("--league-size", type=int, default=4, help="keep at most this many frozen copies in the league")
    p.add_argument("--league-prob", type=float, default=0.15, help="chance each seat is a league copy instead of a bot")
    p.add_argument("--init-from", help="start from this checkpoint's weights (same observation settings)")
    # eval
    p.add_argument("--eval-size", type=int, help="fixed board size for evaluation (default: training settings)")
    p.add_argument("--eval-n-opponents", type=int, help="fixed opponent count for evaluation")
    p.add_argument("--eval-lag", type=float, help="fixed lag for evaluation")
    p.add_argument("--best-metric", default="win_rate", choices=["win_rate", "return", "peak_territory", "territory_frac"],
                   help="evaluation number (averaged over opponents) that picks best.pt")
    p.add_argument("--eval-opponents", nargs="+", default=list(DEFAULT_POOL))
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--log-every", type=int, default=2_000)
    add_dataclass_args(p, DynaConfig)
    p.add_argument("--reward-preset", choices=["v1", "v2", "score", "score_rank"], default="v1",
                   help="starting reward settings; --reward-* flags override single values")
    add_dataclass_args(p, RewardConfig, prefix="reward-")
    return p


def load_extended(module: torch.nn.Module, state: dict, exact_only: bool = False) -> list[str]:
    """Load a checkpoint into a network whose inputs or outputs grew.

    Same-shape tensors are copied. A weight with more input columns (new
    ``vec`` features, which are always appended last) gets the old columns and
    zeros for the new ones, so it computes the same function as before. A
    layer with more outputs (a new action) keeps fresh weights for the new
    rows. With ``exact_only`` mismatched tensors keep their fresh values.
    """
    notes = []
    own = module.state_dict()
    for name, new in own.items():
        old = state.get(name)
        if old is None:
            notes.append(f"{name}: new")
            continue
        if old.shape == new.shape:
            own[name] = old.clone()
            continue
        if exact_only or old.dim() != new.dim() or any(o > n for o, n in zip(old.shape, new.shape)):
            notes.append(f"{name}: fresh {tuple(new.shape)}")
            continue
        grown = new.clone()
        if old.dim() == 2 and old.shape[1] < new.shape[1]:
            grown[:, old.shape[1]:] = 0.0  # new inputs start with no influence
        grown[tuple(slice(0, d) for d in old.shape)] = old
        own[name] = grown
        notes.append(f"{name}: {tuple(old.shape)} -> {tuple(new.shape)}")
    module.load_state_dict(own)
    return notes


def load_by_name(q: torch.nn.Module, state: dict, old_obs, new_obs) -> list[str]:
    """Warm-start a Q-network whose inputs changed (curriculum stages).

    Input channels of the first conv layer and ``vec`` columns of the first
    linear layer are matched by name; inputs the old network didn't have start
    at zero weight, so the new network computes what the old one did on the
    inputs they share. Other same-shape tensors are copied; a larger output
    layer (e.g. a new pause action) keeps fresh rows for the new actions.
    """
    own = q.state_dict()
    notes = []
    old_ch, new_ch = list(old_obs.channels), list(new_obs.channels)
    old_vec, new_vec = list(old_obs.vec_features), list(new_obs.vec_features)
    conv0 = "conv.0.weight"
    first_lin = "body.0.weight" if "body.0.weight" in own else "head.0.weight"
    for name, new in own.items():
        old = state.get(name)
        if old is None:
            notes.append(f"{name}: new")
            continue
        if name == conv0:
            w = torch.zeros_like(new)
            hits = 0
            for j, ch in enumerate(new_ch):
                if ch in old_ch:
                    w[:, j] = old[:, old_ch.index(ch)]
                    hits += 1
            own[name] = w
            notes.append(f"conv input: {hits}/{len(new_ch)} channels carried over")
            continue
        if name == first_lin and old.shape != new.shape:
            n_conv = new.shape[1] - len(new_vec)
            if old.shape[1] - len(old_vec) != n_conv:
                notes.append(f"{name}: conv size changed, fresh")
                continue
            w = torch.zeros_like(new) if old.shape[0] == new.shape[0] else new.clone()
            w[: old.shape[0], :n_conv] = old[:, :n_conv]
            hits = 0
            for j, f in enumerate(new_vec):
                if f in old_vec:
                    w[: old.shape[0], n_conv + j] = old[:, n_conv + old_vec.index(f)]
                    hits += 1
            own[name] = w
            notes.append(f"vec input: {hits}/{len(new_vec)} features carried over")
            continue
        if old.shape == new.shape:
            own[name] = old.clone()
        elif old.dim() == new.dim() and all(o <= n for o, n in zip(old.shape, new.shape)):
            grown = new.clone()
            grown[tuple(slice(0, d) for d in old.shape)] = old
            own[name] = grown
            notes.append(f"{name}: {tuple(old.shape)} -> {tuple(new.shape)}")
        else:
            notes.append(f"{name}: fresh")
    q.load_state_dict(own)
    return notes


def linear_eps(step: int, total: int, cfg: DynaConfig) -> float:
    frac = min(1.0, step / max(1, cfg.eps_fraction * total))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(args: argparse.Namespace) -> Path:
    run = Path(args.run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = dataclass_from_args(DynaConfig, args)
    cfg.n_actions = 4 if args.allow_pause else 3
    reward = RewardConfig.preset(args.reward_preset)
    for k, v in vars(dataclass_from_args(RewardConfig, args, prefix="reward_")).items():
        if getattr(args, "reward_" + k) is not None:  # explicit flags win over the preset
            setattr(reward, k, v)
    reward.gamma = cfg.gamma  # shaping is only policy-invariant with the learner's discount

    env_kwargs = dict(
        size=args.size,
        opponents=args.opponents,
        n_opponents=args.n_opponents,
        max_steps=args.max_steps,
        spawn_radius=args.spawn_radius,
        obs_mode=args.obs_mode,
        view_radius=args.view_radius,
        coarse_factor=args.coarse_factor,
        visible_radius=args.visible_radius,
        threat_features=args.threat_features,
        minimap=args.minimap,
        enemy_heading=args.enemy_heading,
        size_features=args.size_features,
        reclaim_features=args.reclaim_features,
        escape_features=args.escape_features,
        allow_pause=args.allow_pause,
        memory=args.memory,
        league_prob=args.league_prob,
        max_pause=args.max_pause,
        reward=reward.to_dict(),
        lag_prob=args.lag_prob,
        respawn=args.respawn,
        respawn_delay=args.respawn_delay,
        target_frac=args.target_frac,
        size_range=args.size_range,
        n_opponents_range=args.n_opponents_range,
        lag_range=args.lag_range,
    )
    eval_kwargs = dict(env_kwargs)
    if args.eval_size or args.eval_n_opponents or args.eval_lag is not None:
        eval_kwargs.update(size_range=None, n_opponents_range=None, lag_range=None)
        eval_kwargs["size"] = args.eval_size or args.size
        eval_kwargs["n_opponents"] = args.eval_n_opponents or args.n_opponents
        if args.eval_lag is not None:
            eval_kwargs["lag_prob"] = args.eval_lag
    (run / "config.json").write_text(
        json.dumps(
            {"algo": "dyna", "seed": args.seed, "steps": args.steps, "env": env_kwargs, "dyna": cfg.to_dict(),
             "eval": {"opponents": args.eval_opponents, "episodes": args.eval_episodes, "every": args.eval_every,
                      "env": eval_kwargs, "best_metric": args.best_metric},
             "league": {"every": args.league_every, "size": args.league_size}, "init_from": args.init_from},
            indent=2,
        )
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    env = SplixEnv(**env_kwargs)
    cfg.n_actions = int(env.action_space.n)
    grid_shape = env.observation_space["grid"].shape
    n_vec = env.observation_space["vec"].shape[0]
    agent = DynaAgent(env.obs_config, grid_shape, n_vec, cfg, device=pick_device(args.device), seed=args.seed)
    if args.init_from:  # curriculum: continue from an earlier stage's weights
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        old_obs = ObsConfig.from_dict(ck["obs_config"])
        if old_obs != env.obs_config:  # different inputs: line them up by name
            notes = load_by_name(agent.q, ck["q"], old_obs, env.obs_config)
        else:
            notes = load_extended(agent.q, ck["q"])
        agent.q_target.load_state_dict(agent.q.state_dict())
        if "model" in ck:
            notes += load_extended(agent.model, ck["model"], exact_only=True)
        agent.sync_actor()
        print(f"initialised from {args.init_from}" + (f" ({'; '.join(notes)})" if notes else ""), flush=True)
    real = MultiBuffer(max(1, args.num_envs), cfg.buffer_size, grid_shape, n_vec, seed=args.seed)
    imagined = ReplayBuffer(cfg.model_buffer_size, grid_shape, n_vec, seed=args.seed + 1)

    train_log = CsvLogger(run / "train.csv", TRAIN_FIELDS)
    metric_log = CsvLogger(run / "metrics.csv", METRIC_FIELDS)
    evaluator = TrainingEvaluator(run, eval_kwargs, args.eval_opponents, args.eval_episodes)
    league: list[str] = []

    best = {"score": -float("inf")}

    def run_eval(step: int) -> None:
        agent.sync_actor()
        summary = evaluator(agent, step)
        score = float(np.mean([r[args.best_metric] for r in summary]))
        if score > best["score"]:  # keep the checkpoint with the best mean win rate
            best["score"] = score
            agent.save(str(run / "best.pt"), extra={"env": env_kwargs, "step": step, "mean_win_rate": score,
                                                    "best_metric": args.best_metric})
        agent.save(str(run / "checkpoints" / f"agent_{step}.pt"), extra={"env": env_kwargs, "step": step})

    # Several games at once in worker processes (--num-envs). Each game keeps its
    # own replay buffer, so n-step returns never mix games.
    N = max(1, args.num_envs)
    make = functools.partial(SplixEnv, **env_kwargs)
    same_step = gym.vector.AutoresetMode.SAME_STEP
    if N > 1:
        envs = gym.vector.AsyncVectorEnv([make] * N, autoreset_mode=same_step)
    else:
        envs = gym.vector.SyncVectorEnv([make], autoreset_mode=same_step)
    obs, _ = envs.reset(seed=[args.seed + i for i in range(N)])
    ep_ret = np.zeros(N)
    episode = 0
    q_loss = float("nan")
    model_stats: dict = {}
    t0, last_t, last_step = time.time(), time.time(), 0
    run_eval(0)
    step, owed = 0, 0.0

    def crossed(prev: int, now: int, every: int) -> bool:
        return every > 0 and prev // every != now // every

    while step < args.steps:
        eps = linear_eps(step, args.steps, cfg)
        actions = agent.act_batch(obs, epsilon=eps)
        nobs, r, term, trunc, info = envs.step(actions)
        done = term | trunc
        final_obs = info.get("final_obs")
        final_info = info.get("final_info", {})
        for i in range(N):
            nxt = final_obs[i] if done[i] else {"grid": nobs["grid"][i], "vec": nobs["vec"][i]}
            real.bufs[i].add(obs["grid"][i], obs["vec"][i], int(actions[i]), float(r[i]),
                             nxt["grid"], nxt["vec"], float(term[i]), end=bool(done[i]))
            ep_ret[i] += r[i]
            if done[i]:
                fi = {k: v[i] for k, v in final_info.items() if not k.startswith("_")}
                train_log.log({
                    "step": step + i + 1, "episode": episode, "return": ep_ret[i], "length": fi["steps"],
                    "territory_frac": fi["territory_frac"], "peak_territory": fi["peak_territory"],
                    "total_captured": fi["total_captured"], "kills": fi["kills"], "outcome": fi["outcome"],
                    "death_cause": fi["death_cause"] or "", "opponent": "+".join(fi["opponents"]), "epsilon": eps,
                })
                episode += 1
                ep_ret[i] = 0.0
        obs = nobs
        prev, step = step, step + N

        owed += N / cfg.train_freq  # keep the update-to-data ratio whatever N is
        while owed >= 1.0 and step >= cfg.learning_starts:
            owed -= 1.0
            if cfg.n_step > 1:
                batch = real.sample_nstep(cfg.batch_size, cfg.n_step, cfg.gamma, agent.device)
            else:
                batch = real.sample(cfg.batch_size, agent.device)
            loss = agent.q_update(batch)
            want_metrics = crossed(prev, step, args.log_every) or (cfg.model_gate < 1.0 and crossed(prev, step, 500))
            if want_metrics:
                q_loss = float(loss)
            if agent.planning:
                stats = agent.model_update(real.sample(cfg.model_batch_size, agent.device), with_metrics=want_metrics)
                model_stats = stats or model_stats
                model_ok = model_stats.get("model_changed_err", 1.0) < cfg.model_gate or cfg.model_gate >= 1.0
                if step >= cfg.model_warmup and imagined.size >= cfg.batch_size and model_ok:
                    for _ in range(cfg.planning_updates):
                        agent.q_update(imagined.sample(cfg.batch_size, agent.device))
        gate_open = cfg.model_gate >= 1.0 or model_stats.get("model_changed_err", 1.0) < cfg.model_gate
        if agent.planning and step >= cfg.model_warmup and crossed(prev, step, cfg.rollout_freq) and gate_open:
            agent.imagine(real, imagined, eps)

        if crossed(prev, step, args.log_every):
            now = time.time()
            metric_log.log({"step": step, "q_loss": q_loss, **model_stats, "imagined": imagined.size,
                            "q_updates": agent.q_updates, "skipped_updates": agent.skipped_updates, "steps_per_sec": (step - last_step) / (now - last_t)})
            last_t, last_step = now, step
        if crossed(prev, step, args.eval_every) or step >= args.steps:
            run_eval(step)
        if crossed(prev, step, args.league_every):  # self-play league
            (run / "league").mkdir(exist_ok=True)
            path = run / "league" / f"agent_{step}.pt"
            agent.sync_actor()
            agent.save(str(path), extra={"env": env_kwargs, "step": step})
            league.append(f"frozen:{path}")
            if len(league) > args.league_size:
                league.pop(0)
            envs.set_attr("league", [list(league) for _ in range(N)])
            print(f"[league {step:>8}] league: {[opponent_label(x) for x in league]}", flush=True)
    envs.close()

    agent.save(str(run / "agent.pt"), extra={"env": env_kwargs, "step": args.steps})
    for log in (train_log, metric_log, evaluator):
        log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {run / 'agent.pt'}")
    return run / "agent.pt"


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
