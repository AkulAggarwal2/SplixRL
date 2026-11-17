"""Evaluation shared by every algorithm.

Same opponents, same seeds (episode i always uses seed ``seed + i``), greedy
policies. Anything that can act is an "actor": a saved agent of any
algorithm, or one of the bots playing in the agent's seat (that gives the
random and rule-based baselines in the same table).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .env import SplixEnv
from .obs import build_obs
from .opponents import BOTS, make_opponent, opponent_label

EPISODE_FIELDS = (
    "actor",
    "opponent",
    "seed",
    "return",
    "length",
    "territory",
    "territory_frac",
    "peak_territory",
    "total_captured",
    "kills",
    "alive",
    "death_cause",
    "outcome",
    "won",
)


class Actor:
    name = "actor"

    def reset(self, env: SplixEnv, seed: int) -> None:
        pass

    def act(self, env: SplixEnv, obs) -> int:
        raise NotImplementedError


class AgentActor(Actor):
    def __init__(self, agent, name: str = "agent", deterministic: bool = True):
        self.agent = agent
        self.name = name
        self.deterministic = deterministic

    def act(self, env, obs) -> int:
        if self.agent.obs_config != env.obs_config:  # agent trained with other view settings
            obs = build_obs(env.game, 0, self.agent.obs_config)
        return int(self.agent.act(obs, deterministic=self.deterministic))


class BotActor(Actor):
    """A bot sitting in the agent's seat (player 0)."""

    def __init__(self, spec: str):
        self.spec = spec
        self.name = spec

    def reset(self, env, seed):
        self.bot = make_opponent(self.spec)
        self.bot.reset(env.game, 0, np.random.default_rng(seed + 7_777))

    def act(self, env, obs) -> int:
        return int(self.bot.act(env.game, 0))


def make_actor(spec: str, device: str = "cpu") -> Actor:
    """Bot name (``loop``, ``random`` ...) or path to a checkpoint (``frozen:`` optional)."""
    if spec.startswith("frozen:"):
        spec = spec[len("frozen:") :]
    if spec.partition("?")[0] in BOTS:
        return BotActor(spec)
    from .agents import load_agent

    return AgentActor(load_agent(spec, device=device), name=opponent_label("frozen:" + spec)[len("frozen:") :])


def run_episode(actor: Actor, env: SplixEnv, seed: int, opponents: Sequence | None = None) -> dict:
    obs, info = env.reset(seed=seed, options={"opponents": list(opponents)} if opponents else None)
    actor.reset(env, seed)
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(actor.act(env, obs))
        ret += r
        done = term or trunc
    return {
        "actor": actor.name,
        "opponent": "+".join(info["opponents"]),
        "seed": seed,
        "return": ret,
        "length": info["steps"],
        "territory": info["territory"],
        "territory_frac": info["territory_frac"],
        "peak_territory": info["peak_territory"],
        "total_captured": info["total_captured"],
        "kills": info["kills"],
        "alive": int(info["alive"]),
        "death_cause": info["death_cause"] or "",
        "outcome": info["outcome"],
        "won": int(info["won"]),
    }


def evaluate(
    actor: Actor,
    env_kwargs: dict,
    opponents: Iterable[str],
    episodes: int,
    seed: int = 10_000,
) -> list[dict]:
    """``episodes`` games against each opponent spec (all seats get that spec)."""
    kwargs = dict(env_kwargs)
    n_opp = kwargs.get("n_opponents", 1)
    rows = []
    for spec in opponents:
        kwargs["opponents"] = [spec]
        env = SplixEnv(**kwargs)
        for i in range(episodes):
            rows.append(run_episode(actor, env, seed + i, [spec] * n_opp))
    return rows


def _mean_ci(xs: Sequence[float]) -> tuple[float, float]:
    """Mean and 95% normal-approximation half-width."""
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return math.nan, math.nan
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs)))


SUMMARY_METRICS = ("won", "return", "territory_frac", "peak_territory", "length", "alive", "kills")


def summarize(rows: Sequence[dict], keys: Sequence[str] = ("actor", "opponent")) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(keys, key))
        row["episodes"] = len(rs)
        for m in SUMMARY_METRICS:
            mean, ci = _mean_ci([float(r[m]) for r in rs])
            name = "win_rate" if m == "won" else "survival_rate" if m == "alive" else m
            row[name] = mean
            row[name + "_ci"] = ci
        out.append(row)
    return out


def format_table(summary: Sequence[dict], keys: Sequence[str] = ("actor", "opponent")) -> str:
    cols = [*keys, "episodes", "win_rate", "return", "territory_frac", "peak_territory", "length", "survival_rate"]
    def cell(r, c):
        v = r[c]
        if isinstance(v, float):
            ci = r.get(c + "_ci")
            return f"{v:.3f}±{ci:.3f}" if ci is not None else f"{v:.3f}"
        return str(v)
    table = [cols] + [[cell(r, c) for c in cols] for r in summary]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    return "\n".join("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in table)


def write_csv(path: str | Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or (rows[0].keys() if rows else EPISODE_FIELDS))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class CsvLogger:
    """Appends rows to a CSV, writing the header on first use."""

    def __init__(self, path: str | Path, fields: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.f = self.path.open("a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def log(self, row: dict) -> None:
        self.w.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()


EVAL_FIELDS = (
    "step", "opponent", "episodes",
    "win_rate", "win_rate_ci", "return", "return_ci", "territory_frac", "territory_frac_ci",
    "peak_territory", "peak_territory_ci", "length", "length_ci",
    "survival_rate", "survival_rate_ci", "kills", "kills_ci",
)


class TrainingEvaluator:
    """Periodic evaluation for any algorithm's training loop.

    Writes ``<run_dir>/eval.csv`` in the format ``scripts/plot.py`` reads, so
    every algorithm's learning curves can go on the same plot::

        evaluator = TrainingEvaluator(run_dir, env_kwargs, ["random_safe", "loop", "chaser"], 20)
        evaluator(agent, step)   # every N env steps
    """

    def __init__(self, run_dir, env_kwargs: dict, opponents: Sequence[str], episodes: int, seed: int = 10_000):
        self.env_kwargs = {k: v for k, v in env_kwargs.items() if k != "opponents"}
        self.opponents = list(opponents)
        self.episodes = episodes
        self.seed = seed
        self.log = CsvLogger(Path(run_dir) / "eval.csv", EVAL_FIELDS)

    def __call__(self, agent, step: int, verbose: bool = True) -> list[dict]:
        rows = evaluate(AgentActor(agent, getattr(agent, "algo", "agent")), self.env_kwargs, self.opponents, self.episodes, self.seed)
        summary = summarize(rows, keys=("opponent",))
        for s in summary:
            self.log.log({"step": step, **s})
        if verbose:
            parts = ", ".join(f"{s['opponent']} {s['win_rate']:.2f}" for s in summary)
            print(f"[eval {step:>8}] win rate: {parts}", flush=True)
        return summary

    def close(self) -> None:
        self.log.close()
