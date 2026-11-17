"""Evaluate actors against opponents: the results table and the cross-play matrix.

    # baselines + trained agents vs the fixed bots, 100 games each
    python -m splixrl.evaluate --actors random random_safe loop \\
        runs/round0/dyna/seed*/agent.pt --opponents random_safe loop chaser --episodes 100

    # cross-play between frozen agents (win rate of row vs column)
    python -m splixrl.evaluate --actors A.pt B.pt C.pt --opponents frozen:A.pt frozen:B.pt frozen:C.pt --pivot win_rate

Every actor meets every opponent on the same seeds. Actors are bot names or
checkpoint paths; opponents are bot names or frozen:<checkpoint>.
"""

from __future__ import annotations

import argparse

from .evaluation import evaluate, format_table, make_actor, summarize, write_csv
from .opponents import DEFAULT_POOL, opponent_label


def pivot(summary: list[dict], metric: str) -> str:
    actors = list(dict.fromkeys(r["actor"] for r in summary))
    opps = list(dict.fromkeys(r["opponent"] for r in summary))
    val = {(r["actor"], r["opponent"]): r[metric] for r in summary}
    w = max(len(a) for a in actors) + 2
    cw = max(10, *(len(o) + 2 for o in opps))
    lines = [f"{metric} (row actor vs column opponent)", " " * w + "".join(o.ljust(cw) for o in opps)]
    for a in actors:
        lines.append(a.ljust(w) + "".join(f"{val.get((a, o), float('nan')):.3f}".ljust(cw) for o in opps))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actors", nargs="+", required=True)
    ap.add_argument("--opponents", nargs="+", default=list(DEFAULT_POOL))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=10_000, help="episode i uses seed + i")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--n-opponents", type=int, default=1)
    ap.add_argument("--respawn", action="store_true")
    ap.add_argument("--target-frac", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="CSV of every episode")
    ap.add_argument("--summary-out", help="CSV of the summary table")
    ap.add_argument("--pivot", help="also print a matrix of this metric, e.g. win_rate")
    args = ap.parse_args(argv)

    env_kwargs = dict(size=args.size, max_steps=args.max_steps, n_opponents=args.n_opponents,
                      respawn=args.respawn, target_frac=args.target_frac)
    rows = []
    for spec in args.actors:
        actor = make_actor(spec, device=args.device)
        rows += evaluate(actor, env_kwargs, args.opponents, args.episodes, args.seed)
        print(f"done: {actor.name}", flush=True)
    for r in rows:  # shorter opponent names in tables
        r["opponent"] = "+".join(opponent_label(o) if o.startswith("frozen:") else o for o in r["opponent"].split("+"))
    summary = summarize(rows)
    print()
    print(format_table(summary))
    if args.pivot:
        print()
        print(pivot(summary, args.pivot))
    if args.out:
        write_csv(args.out, rows)
    if args.summary_out:
        write_csv(args.summary_out, summary)


if __name__ == "__main__":
    main()
