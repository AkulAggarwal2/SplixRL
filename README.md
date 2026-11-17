# SplixRL

Comparing RL algorithms on a Splix.io-style territory game (CS 4180/5180 final project:
Akul Aggarwal, Sai Dhanasiri, Ahaan Kallat).

* **Game + Gymnasium env** (`splixrl/game.py`, `splixrl/env.py`): fast Python rules engine (~10k steps/s).
* **Fixed opponents** (`splixrl/opponents/`): random, random-safe, LoopBot, ChaserBot, and any saved agent frozen as an opponent.
* **Shared evaluation** (`splixrl/evaluation.py`): same opponents, same seeds, same metrics for every algorithm.
* **Dyna** (Ahaan, `splixrl/agents/dyna.py`): Double-DQN learner plus a learned world model that generates imagined transitions.
* **Double DQN** (Akul, `splixrl/agents/ddqn.py`) and **PPO** (Sai, `splixrl/agents/ppo.py`): interface stubs to fill in.
* Manual play, live/GIF viewer, learning-curve plots.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # rules, env, bots, save/load
```

## Try it

```bash
python -m splixrl.play                                   # you vs LoopBot (arrows/WASD)
python -m splixrl.play --opponents chaser --n-opponents 2
python -m splixrl.watch --actor loop --opponents chaser  # watch two bots
python -m splixrl.watch --actor runs/round0/dyna/seed0/agent.pt --opponents chaser --gif dyna.gif
```

## Rules

Our rules follow the open-source Splix game server
([jespertheend/splix](https://github.com/jespertheend/splix), `gameServer/src/gameplay/Player.js`).
Everything happens on a square grid, and all players move at once each step.

* Each player starts with a 5×5 square. Actions are relative: **straight, left, right**. Reversing is impossible.
* Off your own land you leave a **trail**. Returning to your land captures the trail plus everything it encloses, including other players' land.
* **Leaving the board** kills you.
* **Running into any trail kills that trail's owner.** That's you if it's your own trail; if it's someone else's, they die and you get the kill.
* **Head-on collision** (two heads in the same cell, or swapping cells): a player standing on its own land survives, everyone else there dies.
* **Elimination** wipes your land and trail. Losing all your land also eliminates you.
* **The episode ends** when the agent dies, when every opponent is dead, or at `max_steps` (default 500).
* **Win:** all opponents dead, or more territory than every living opponent at the step limit.

**Observation.** A dict `{"grid", "vec"}`.
* `grid`: a 15×15 window centred on the player's head and rotated so the player faces up. Channels are own land, own trail, enemy land, enemy trail, enemy heads, wall.
* `vec`: 11 numbers: in-own-land, trail length, territory share, distance and direction to home and to the nearest enemy, enemy-to-trail distance, time.

`obs_mode="global"` gives the whole board instead.

**Reward** (`RewardConfig`, the same for every algorithm):
* +0.1 per cell captured
* +3 per kill
* −3 on death

Optional extras: potential-based shaping on trail length (Ng et al. 1999), a per-step cost, a win bonus, and a penalty for cells lost.

## Opponents

| spec | behaviour |
|---|---|
| `random` | uniform random moves (dies fast) |
| `random_safe` | random, but never drives into a wall or its own trail |
| `loop` | small rectangular loops away from enemies, heads home when an enemy is within 4 cells |
| `chaser` | hunts enemy trails within 10 cells, otherwise loops |
| `frozen:<path>` | any saved agent, weights frozen, playing from its own seat |
| `loop?max_out=8&danger_dist=2` | bots take keyword arguments |

Win rates between bots over 40 games each on 20×20 (row vs column):

| | random_safe | loop | chaser |
|---|---|---|---|
| **loop** | 0.98 | 0.40 | 0.13 |
| **chaser** | 0.90 | 0.90 | 0.28 |

## Experiment protocol (following Misha's feedback)

Comparisons are only valid between agents that **trained and were evaluated against exactly the same opponents**.
Iterated "train against the frozen previous agent" learns best responses: it shows what exploits what, not which algorithm is better.

**Round 0: the main comparison.**
* All three algorithms train against the bot pool `random_safe loop chaser` with the same env settings, reward, step budget and seeds.
* Evaluate with the same opponents and seeds (100 games per opponent).
* Only these results support statements like "Dyna gets more territory than PPO".

```bash
python -m splixrl.train_dyna --run-dir runs/round0/dyna/seed0 --seed 0 --steps 200000
python -m splixrl.evaluate --actors random random_safe loop runs/round0/*/seed*/agent.pt \
    --opponents random_safe loop chaser --episodes 100 --summary-out results/round0.csv
python -m splixrl.plot --run "Dyna=runs/round0/dyna/seed*" --run "DDQN=runs/round0/ddqn/seed*" \
    --run "PPO=runs/round0/ppo/seed*" --metric win_rate --out plots/round0_win_rate.png
```

**Round 1+: find and fix problems.**
* Freeze the Round 0 agents.
* All three algorithms train against the *same* pool: the bots plus every frozen Round 0 agent.
* Look for what breaks (exploits, overfitting to one opponent, degenerate strategies), fix the environment or reward, re-run, and report before/after.
* Keep evaluating against the bots every round: they are a yardstick that never changes. Report head-to-head results as "X exploits frozen Y" and call the procedure iterated best response. With few rounds, don't claim a Nash equilibrium.

```bash
POOL="random_safe loop chaser frozen:runs/round0/dyna/seed0/agent.pt frozen:runs/round0/ddqn/seed0/agent.pt frozen:runs/round0/ppo/seed0/agent.pt"
python -m splixrl.train_dyna --run-dir runs/round1/dyna/seed0 --opponents $POOL
python -m splixrl.evaluate --actors runs/round1/*/seed0/agent.pt --opponents $POOL --pivot win_rate
```

### Problems found so far

1. **Turtling beat the bots.** An agent that just circles inside its start square beat LoopBot 48/50 and ChaserBot 44/50: the bots drew loops right past it and got cut. Fix: bots leave home on the side away from enemy heads and keep 2 cells from them. After the fix the turtler loses 48/50 and 47/50. `test_bots_beat_turtling` guards this.

## Adding Double DQN / PPO

See the docstrings in `splixrl/agents/ddqn.py` and `ppo.py`. In short:

* Subclass `Agent` with `act(obs, deterministic)`, `save(path)` and `load(path)`. Save a dict that includes `"algo"` and `"obs_config"`; then `load_agent`, `frozen:<path>`, evaluate and watch all work.
* Train on `SplixEnv(**env_kwargs)` with the same settings as Dyna (see `runs/.../config.json`).
* Call `TrainingEvaluator(run_dir, env_kwargs, DEFAULT_POOL, episodes)(agent, step)` periodically, so `eval.csv` plots next to the others.

## Dyna details

`python -m splixrl.train_dyna --help` lists every setting. Defaults:
* 200k real steps, Adam 2.5e-4, γ 0.99, batch 64, replay 50k.
* Double DQN target with a hard target sync every 1000 updates; ε 1→0.05 over 30% of training.
* One real Q-update and one world-model update every 4 env steps, plus **3 imagined Q-updates** per real one.
* Imagined data: every 250 steps the model rolls 512 real start states forward `--rollout-horizon` steps (default 1).

The world model is a dilated CNN that predicts the next view, the change in `vec`, the reward and termination. Before predicting, the current view is shifted and rotated by the action. That part is pure geometry of the egocentric view, so the model only learns what actually changes. `metrics.csv` logs model error, both overall and on changed cells. `--planning-updates 0` gives the no-planning ablation.

Learning runs on CUDA or the Apple GPU when available (`--device auto`); acting uses a CPU copy of the network.

## Layout

```
splixrl/game.py          rules engine
splixrl/env.py           Gymnasium env (SplixRL-v0)
splixrl/obs.py           observations from any player's seat
splixrl/rewards.py       RewardConfig
splixrl/opponents/       bots, frozen agents, spec parsing
splixrl/agents/          Agent interface, Dyna, DDQN/PPO stubs, load_agent
splixrl/evaluation.py    evaluate, summarize, TrainingEvaluator
splixrl/train_dyna.py    Dyna training CLI
splixrl/evaluate.py      results table / cross-play CLI
splixrl/plot.py          learning curves
splixrl/watch.py, play.py, viewer.py, render.py
tests/
```
