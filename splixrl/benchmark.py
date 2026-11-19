"""A test closer to the real game than the small fixed boards.

The real arena is about 600 x 600 with 20-27 players (about 14,000 tiles each).
600 x 600 runs at only a few steps a second in this simulator, so the default
here is 300 x 300 with 12 opponents (about 7,000 tiles each), 3000-step games,
and a mix of opponents that includes players who take land from others: deep
raiders, line cutters, big loopers, human-like hunters and ordinary loopers.
Every opponent seat is drawn from the mix, with the same seeds for every actor.

    python -m splixrl.benchmark --actor runs/options/BEST_OPTIONS_AGENT.pt --games 20 --workers 6
    python -m splixrl.benchmark --actor ... --set STRIPS=0   # an options rule switched off

``--set NAME=value`` changes a setting in ``splixrl.options`` (options agents only).
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp

import numpy as np

MIX = (
    "loop?jitter=true",
    "greedy?jitter=true",
    "invader?reach=40&depth=10",  # raids 10 tiles deep into others' land, from 40 away
    "cutter",  # cuts lines through land enclosed by its own
    "human?jitter=true",
    "hunter?jitter=true",
)
# tougher: prowlers that go and wait near the biggest territory (usually the agent's)
MIX_HARD = MIX + ("prowler", "prowler", "prowler")


def _apply(sets):
    import splixrl.options as O

    for kv in sets:
        k, v = kv.split("=", 1)
        val = eval(v, {}, {})  # numbers, True/False, None
        cur = getattr(O, k)
        setattr(O, k, val if (val is None or cur is None) else type(cur)(val))


def _games(job):
    actor_path, seeds, size, n_opp, steps, lag, sets, mix = job
    _apply(sets)
    from splixrl import SplixEnv
    from splixrl.evaluation import make_actor

    actor = make_actor(actor_path)
    agent = getattr(actor, "agent", None)
    cfg = {} if agent is None else {k: v for k, v in agent.obs_config.to_dict().items() if k not in ("mode", "pause_feature")}
    env = SplixEnv(size=size, n_opponents=n_opp, opponents=list(mix), max_steps=steps, respawn=True,
                   allow_pause=True, lag_prob=lag, **cfg)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        actor.reset(env, seed)
        done, ret, caps, lost_max, peak_land = False, 0.0, [], 0, 0
        while not done:
            obs, r, te, tr, info = env.step(actor.act(env, obs))
            done = te or tr
            ret += r
            if info["captured"]:
                caps.append(info["captured"])
        mine = np.argwhere(env.game.owner == 0)
        shape = len(mine) / float(np.prod(mine.max(0) - mine.min(0) + 1)) if len(mine) else 0.0
        rows.append({
            "shape": shape,
            "return": ret, "survived": not te, "death": info.get("death_cause") if te else None,
            "kills": info["kills"], "peak": info["peak_territory"], "land": info["territory"],
            "captures": caps,
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--opponents", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lag", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=50_000, help="game i uses seed + i, for every actor")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="NAME=value settings in splixrl.options")
    ap.add_argument("--mix", choices=["normal", "hard"], default="normal", help="hard adds prowlers")
    args = ap.parse_args(argv)

    seeds = [args.seed + i for i in range(args.games)]
    chunks = [seeds[i :: args.workers] for i in range(args.workers) if seeds[i :: args.workers]]
    mix = MIX_HARD if args.mix == "hard" else MIX
    jobs = [(args.actor, c, args.size, args.opponents, args.steps, args.lag, args.set, mix) for c in chunks]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rows = [r for part in pool.map(_games, jobs) for r in part]

    n = len(rows)
    caps = np.array([c for r in rows for c in r["captures"]]) if any(r["captures"] for r in rows) else np.zeros(1)
    deaths = collections.Counter(r["death"] for r in rows if r["death"])
    label = " ".join(args.set) or "defaults"
    label += ", hard mix" if args.mix == "hard" else ""
    print(f"{args.size}x{args.size}, {args.opponents} opponents, {n} games of {args.steps} steps ({label}):")
    print(f"  return {np.mean([r['return'] for r in rows]):.1f} | survived {np.mean([r['survived'] for r in rows]):.0%} "
          f"| kills {np.mean([r['kills'] for r in rows]):.1f} | peak land {np.mean([r['peak'] for r in rows]):.0f} "
          f"| land at the end {np.mean([r['land'] for r in rows]):.0f} | median capture {np.median(caps):.0f} "
          f"| shape (land / bounding box) {np.mean([r['shape'] for r in rows if r['land']] or [0]):.2f}")
    print(f"  deaths: {dict(deaths) or 'none'}")


if __name__ == "__main__":
    main()
