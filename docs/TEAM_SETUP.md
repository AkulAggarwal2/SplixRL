# Adding Double DQN and PPO (for Akul and Sai)

Everything except Double DQN (Akul) and PPO (Sai) is done: the game, the
Gymnasium env, the fixed bots, evaluation, plotting, manual play and Dyna
(Ahaan). This page is what you need to make your results directly comparable
with Dyna's.

## The one rule (Misha's feedback)

Results can only be compared between agents that **trained and were evaluated
against exactly the same opponents, with the same settings**. So all three
algorithms use the same environment, reward, step budget, seeds and
evaluation. Training against frozen copies of each other finds best
responses ("X exploits frozen Y"), not a ranking.

## The settings: don't copy them, import them

`splixrl/experiment.py` has the Round 0 settings that the reported Dyna runs
used (`runs/round0_reward2`, checked to match):

| | |
|---|---|
| board | 20×20, 1 opponent drawn per episode from `random_safe`, `loop`, `chaser` |
| episode | 500 steps, no lag, local view (radius 7) |
| reward | **v2**: +0.1 per tile captured, +3 per kill, −3 on death, −0.1 per tile lost or held at death |
| discount | γ = 0.99 |
| budget | 200,000 env steps per run, seeds 0, 1, 2 |
| training eval | every 10,000 steps, 20 games per opponent |
| final eval | 100 games per opponent |

```python
from splixrl import SplixEnv
from splixrl.evaluation import TrainingEvaluator
from splixrl.experiment import ROUND0, round0_env_kwargs

env_kwargs = round0_env_kwargs()
env = SplixEnv(**env_kwargs)
evaluator = TrainingEvaluator(run_dir, env_kwargs, ROUND0["eval_opponents"], ROUND0["eval_episodes"])
# in your training loop:
if step % ROUND0["eval_every"] == 0:
    evaluator(agent, step)  # writes run_dir/eval.csv
```

Watch out: `python -m splixrl.train_dyna` defaults to reward **v1**. The
Dyna results use `--reward-preset v2`.

## What to implement

`splixrl/agents/ddqn.py` and `splixrl/agents/ppo.py` have the interface in
their docstrings. In short:

1. A class subclassing `Agent` with `act(obs, deterministic=True) -> int`,
   `save(path)` and `load(path)`.
2. `save` writes a dict with `"algo": "ddqn"` (or `"ppo"`) and
   `"obs_config": self.obs_config.to_dict()`. Then `load_agent(path)`,
   `frozen:<path>` opponents, `splixrl.evaluate` and `splixrl.watch` all work
   with your checkpoints with no other changes.
3. A training script (e.g. `splixrl/train_ddqn.py`, `splixrl/train_ppo.py`)
   that saves `agent.pt` (final) and `best.pt` (best training eval) in
   `runs/round0/<algo>/seed<k>/`.

Inputs: `obs["grid"]` float32 (6, 15, 15) and `obs["vec"]` float32 (11,);
`splixrl.agents.base.obs_to_torch` batches them. Actions: 0 straight,
1 turn left, 2 turn right.

## Getting Dyna's runs to compare against

`runs/` isn't in git (130 MB). Either ask Ahaan for `runs/round0_reward2/`,
or rerun Dyna (about 80 minutes per seed):

```bash
python -m splixrl.train_dyna --run-dir runs/round0/dyna/seed0 --seed 0 --steps 200000 --reward-preset v2
```

## Comparing

```bash
python -m splixrl.evaluate --actors random_safe loop runs/round0/*/seed*/best.pt \
    --opponents random_safe loop chaser --episodes 100 --summary-out results/round0.csv
python -m splixrl.plot --run "Dyna=runs/round0/dyna/seed*" --run "Double DQN=runs/round0/ddqn/seed*" \
    --run "PPO=runs/round0/ppo/seed*" --metric win_rate --out plots/round0_win_rate.png
```

`evaluate` uses the same games for every actor (episode i uses seed 10000 + i)
and scores the return column with reward v2 by default. Report win rate and
territory, with mean ± std over the 3 seeds.

What Dyna found, for the report (`results/SUMMARY.md`, sections 1–2): reward
v1 collapses late (fixed by v2), and planning makes the late collapse worse.
The same learner without planning (`--planning-updates 0`) is more consistent
across seeds. That learner is Double DQN plus n-step returns, so it's also a
useful sanity check for Akul's version.

## Experiment 2 (optional): the same algorithms over plans

After the per-tile version works. Each algorithm picks one of 17 plans
(`splixrl.options.N_OPTIONS`), carried out by the same scripted
controllers. Details are in the docstrings and in the README, "Experiment 2".
The key points:

* Env: `splixrl.options.make_options_env(gamma=..., **env_kwargs)`. One `step`
  runs a whole plan and returns its discounted reward.
* Bootstrap with `info["option_disc"]` (γ^k for a k-tile plan) instead of γ.
* Save `"options": {"safety": 6.0, "lag_slack": 30.0}` in the checkpoint.
* Report the x-axis in game steps, not decisions (one decision is about 10 tiles).
* The plan controllers do much of the work (an untrained options agent already
  beats the bots on 20×20), so use the harder setting in the README and
  compare return and peak territory, not win rate.
