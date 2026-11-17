"""Learning curves from eval.csv files: mean over seeds with a 95% band.

    python -m splixrl.plot --run "Dyna=runs/round0/dyna/seed*" \\
        --run "Double DQN=runs/round0/ddqn/seed*" --run "PPO=runs/round0/ppo/seed*" \\
        --metric win_rate --out plots/round0_win_rate.png

One panel per opponent. Any algorithm whose training loop writes eval.csv
with ``TrainingEvaluator`` can go on the same plot.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def load_runs(pattern: str) -> list[dict[str, dict[int, float]]]:
    """For each run dir matching the glob: {opponent: {step: value}} per metric row."""
    runs = []
    for d in sorted(glob.glob(pattern)):
        f = Path(d) / "eval.csv"
        if f.exists():
            with f.open() as fh:
                runs.append(list(csv.DictReader(fh)))
    return runs


def curves(runs, metric: str, opponent: str):
    per_seed = []
    for rows in runs:
        pts = {int(r["step"]): float(r[metric]) for r in rows if r["opponent"] == opponent}
        if pts:
            per_seed.append(pts)
    if not per_seed:
        return None
    steps = sorted(set.intersection(*(set(p) for p in per_seed)))
    vals = np.array([[p[s] for s in steps] for p in per_seed])
    mean = vals.mean(0)
    ci = 1.96 * vals.std(0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros_like(mean)
    return np.array(steps), mean, ci, len(vals)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='LABEL=GLOB of run directories')
    ap.add_argument("--metric", default="win_rate", help="win_rate, return, territory_frac, peak_territory, length, survival_rate, kills")
    ap.add_argument("--opponents", nargs="*", help="default: every opponent in the files")
    ap.add_argument("--out", default="plots/curves.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for spec in args.run:
        label, _, pattern = spec.partition("=")
        series.append((label, load_runs(pattern)))
    opps = args.opponents or list(dict.fromkeys(r["opponent"] for _, runs in series for rows in runs for r in rows))
    fig, axes = plt.subplots(1, len(opps), figsize=(4.2 * len(opps), 3.4), squeeze=False, sharey=True)
    for ax, opp in zip(axes[0], opps):
        for label, runs in series:
            c = curves(runs, args.metric, opp)
            if c is None:
                continue
            steps, mean, ci, n = c
            line, = ax.plot(steps, mean, label=f"{label} (n={n})")
            ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=line.get_color())
        ax.set_title(f"vs {opp}")
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(args.metric.replace("_", " "))
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
