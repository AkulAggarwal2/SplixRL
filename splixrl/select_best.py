"""Pick the best checkpoint of a run with a larger, fixed evaluation.

    python -m splixrl.select_best --checkpoints "runs/curriculum/B/checkpoints/*.pt" runs/curriculum/B/agent.pt \
        --opponents loop chaser greedy invader --episodes 30 --size 32 --n-opponents 3 --lag 0.03 \
        --respawn --target-frac 0.5 --max-steps 1500 --out runs/curriculum/B/selected.txt

The periodic evaluation during training uses few games (noisy), so best.pt can
be a lucky early checkpoint. This re-evaluates candidates on the same seeds
with more games, prints a table, and writes the winner's path to --out.
Score = mean evaluation return over the opponents (``--metric`` to change).
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

from .evaluation import evaluate, make_actor, summarize


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="paths or globs")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--respawn", action="store_true")
    ap.add_argument("--target-frac", type=float, default=None)
    ap.add_argument("--metric", default="return")
    ap.add_argument("--every", type=int, default=1, help="only every k-th checkpoint (by step), plus the last")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    paths = []
    for pat in args.checkpoints:
        paths += sorted(glob.glob(pat)) or [pat]
    def step_of(p):
        digits = "".join(ch for ch in p.rsplit("/", 1)[-1] if ch.isdigit())
        return int(digits) if digits else 10**12
    paths = sorted(dict.fromkeys(paths), key=step_of)
    if args.every > 1:
        paths = paths[:-1][:: args.every] + paths[-1:]

    env_kwargs = dict(size=args.size, n_opponents=args.n_opponents, max_steps=args.max_steps, lag_prob=args.lag,
                      respawn=args.respawn, target_frac=args.target_frac)
    scores = []
    for p in paths:
        actor = make_actor(p)
        actor.name = p
        summary = summarize(evaluate(actor, env_kwargs, args.opponents, args.episodes))
        score = float(np.mean([s[args.metric] for s in summary]))
        scores.append((score, p))
        per = "  ".join(f"{s['opponent'].split('+')[0]}: {s[args.metric]:.1f}" for s in summary)
        print(f"{score:8.2f}  {p}   ({per})", flush=True)
    best = max(scores)
    print(f"\nbest: {best[1]} ({args.metric} {best[0]:.2f})")
    with open(args.out, "w") as f:
        f.write(best[1] + "\n")


if __name__ == "__main__":
    main()
