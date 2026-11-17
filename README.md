# SplixRL

Comparing RL algorithms on a Splix.io-style territory game (CS 4180/5180 final project:
Akul Aggarwal, Sai Dhanasiri, Ahaan Kallat).

* **Game + Gymnasium env** (`splixrl/game.py`, `splixrl/env.py`): fast Python rules engine (~10k steps/s).
* **Fixed opponents** (`splixrl/opponents/`): random, random-safe, LoopBot, ChaserBot, and any saved agent frozen as an opponent.
* **Shared evaluation** (`splixrl/evaluation.py`): same opponents, same seeds, same metrics for every algorithm.
* **Dyna** (Ahaan, `splixrl/agents/dyna.py`): Double-DQN learner plus a learned world model that generates imagined transitions.
* **Double DQN** (Akul, `splixrl/agents/ddqn.py`) and **PPO** (Sai, `splixrl/agents/ppo.py`): interface stubs to fill in.
* Manual play, live/GIF viewer, learning-curve plots.
* **Real game** (`splixrl/live.py`): agents and bots play on the official Splix server, run locally.

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

**Pausing** (`--allow-pause`, also `python -m splixrl.play --allow-pause` with P): a 4th action that stops the player, as the real server allows. Trails stay where they are, so a paused player with a trail out can still be cut. Any other action resumes relative to the old heading. `--max-pause` caps how long it may stand still. Warm starts (`--init-from`) can grow a 3-action checkpoint into a 4-action one: the old outputs are kept, and new inputs start at zero weight.

**Options for big, crowded maps** (local mode, all off by default):

| option | adds | why |
|---|---|---|
| `--coarse-factor 3` | 6 channels: a 45×45-tile window max-pooled to 15×15 | a human sees about 50×50 tiles on the real server; the fine view is only 15×15 |
| `--visible-radius 25` | hides opponents more than 25 tiles away, in the grid and in `vec` | the real server only sends what's within 25 tiles, so without this the agent trains with information it won't have |
| `--threat-features` | 3 `vec` numbers: nearest enemy head to my land, enemy trail cells on my land, distance to the nearest enemy trail | defending land (stolen cells cost reward), and an enemy trail on your land is a free kill |
| `--minimap` | 2 channels: anyone's land on the whole board, and my position, north-up; heading added to `vec` | the real minimap: an 80×80 bitmap of occupied land, decoded by `live.py` |

| `--enemy-heading` | 1 channel: the tile in front of each visible enemy head | tells who is coming at you or your land, not just where they are |
| `--size-features` | 3 `vec` numbers: my land, nearest enemy's visible land, the difference (log scale) | dying costs your land, so a small agent can risk more against a big one |
| `--reclaim-features` | 4 `vec` numbers: cheap empty land enclosed by or dented into mine, and the distance and direction to the nearest | when an enemy that ate into your land dies, its land is cleared and a tiny loop takes it back |

| `--memory 100` (needs `--minimap`) | 3 minimap-frame channels: enemy heads seen in the last 100 steps (fading), land that appeared on the minimap, my land now owned by someone else; plus 4 `vec` numbers on stolen land (amount, distance, direction) | humans hunt from the minimap and eat into your land from outside your view; this gives the agent memory of who was around, who is expanding toward it, and what it just lost |

The world model shifts and rotates the fine channels, only rotates the coarse ones, and leaves the minimap alone.

**Learner options** (Dyna; defaults are the original learner): `--n-step 3` (multi-step returns), `--dueling true` (separate state value and action advantage), `--tau 0.005` (soft target updates), `--model-gate 0.35` (plan only while the world model's error on changed cells is below 0.35).

**Reward** (`RewardConfig`, the same for every algorithm):
* +0.1 per cell captured
* +3 per kill
* −3 on death

Optional extras:
* `death_per_cell`: a death penalty per cell owned, i.e. dying costs everything you hold (see problem 2 below);
* potential-based shaping on trail length (Ng et al. 1999);
* a per-step cost, a win bonus, and a penalty for cells lost.

**Territory rules** (`--respawn --target-frac 0.5`, with a longer `--max-steps`) are closer to the real game. Dead opponents come back after 10 steps, so killing them never ends the game, and you win by owning that share of the map. Dying is a loss, and running out of steps without reaching the target is a timeout. How often the bots reach 50% of the map:

| setting | LoopBot | ChaserBot |
|---|---|---|
| 20×20, 1 LoopBot opponent, 1500 steps | 80% of games (peak 195 cells) | 70% |
| 40×40, 4 LoopBot opponents, 3000 steps | 0% (peak 363 cells ≈ 23%) | 5% |

**Reward presets** (`--reward-preset`): `v1` as above; `v2` adds 0.1 per cell lost on death or stolen; `score_rank` is `score` plus an end-of-game bonus (up to 3) for holding more land than the opponents. `score` matches the real game's score. The in-game panel ("Kills: 1, Blocks: 3748, Score: 4248") shows a kill is worth 500 tiles, so `score` pays 0.01 per tile and 5 per kill and keeps the loss of land on death. Under v1/v2 a kill was worth only 30 tiles, which is one reason agents rarely went for kills.

`lag_prob` makes a requested turn land one tile late with that probability, imitating the real server's lag (see "Real game").

## Opponents

| spec | behaviour |
|---|---|
| `random` | uniform random moves (dies fast) |
| `random_safe` | random, but never drives into a wall or its own trail |
| `loop` | small rectangular loops away from enemies, heads home when an enemy is within 4 cells |
| `chaser` | hunts enemy trails within 10 cells, otherwise loops |
| `invader` | raids the nearest enemy land (drives 4 tiles in, then home), so agents practise noticing and punishing land theft |
| `hunter` | a chaser that goes after exposed trails up to 20 tiles away |
| `stalker` | a chaser that sees exposed trails anywhere on the map and travels to hunt them: a worst-case minimap hunter |
| `greedy` | big loops, barely checks for danger: lots of land and long trails to cut. A ChaserBot gets 18.7 kills per game against four of them on 40×40 |
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

### Round 0 results so far (Dyna, 20×20, 1 opponent, 50 games per opponent, best checkpoint by evaluation)

| actor | vs random_safe | vs loop | vs chaser |
|---|---|---|---|
| LoopBot (baseline) | 0.90 | 0.48 | 0.10 |
| Dyna, reward v1 (seed 1 at 50k, its peak) | 0.96 | 0.98 | 0.56 |
| Dyna, reward v2, seed 0 | 0.96 | 0.78 | 0.68 |
| Dyna, reward v2, seed 1 | 0.98 | 0.94 | 0.76 |
| Dyna, reward v2, seed 2 | 0.96 | 0.46 | 0.16 |

Under v1 every seed later collapsed (0–0.05 vs chaser after 70k). v2 held up much longer, but still drifted late (0–0.1 vs chaser at 190k), and seed 2 never got good. `best.pt` keeps the peak. **Ablation (seed 0): planning causes the late collapse.** Win rate vs loop / chaser at 40k, 120k and 200k steps:

| | 40k | 120k | 200k |
|---|---|---|---|
| Dyna, 3 imagined updates per real one | 0.85 / 0.70 | 0.40 / 0.00 | 0.35 / 0.00 |
| same learner, planning off | 0.90 / 0.50 | 0.75 / 0.65 | 0.70 / 0.35 |

Imagined transitions from an imperfect world model slowly pull the Q-values off course; without them the learner holds up. This is one seed so far, and more are needed. `runs/ablation/dyna_stable` (n-step, dueling, soft target, learning rate 1e-4, gated planning) learns much more slowly, most likely because of the learning rate.

### Runs toward the real game

`runs/territory40/dyna/seed0` uses:
* 40×40 board, 4 opponents from the pool `random_safe loop chaser greedy invader`;
* `--reward-preset score --gamma 0.995` (big loops pay out late; 0.99 made the agent too cautious);
* territory rules: respawn, win at 50% of the map, 1500 steps;
* all the observation options above, and the learner options;
* `--lag-prob 0.05`;
* 400k steps.

This is the first agent trained with roughly what a player has on the real server.

### Problems found so far

1. **Turtling beat the bots.** An agent that just circles inside its start square beat LoopBot 48/50 and ChaserBot 44/50: the bots drew loops right past it and got cut. Fix: bots leave home on the side away from enemy heads and keep 2 cells from them. After the fix the turtler loses 48/50 and 47/50. `test_bots_beat_turtling` guards this.
2. **The reward paid for risk the game punishes.** With reward v1 (+0.1 per cell, flat −3 for dying), Dyna's training return kept rising after 50k steps (peak territory 52 → 103 cells), while its win rate against LoopBot fell from 0.85–1.0 to 0.15–0.2 and against ChaserBot from 0.35–0.65 to 0–0.05, in all 3 seeds. Big loops pay +6 and up, so getting cut sometimes is worth it under the reward, but losing everything on death loses the game. Fix (reward v2): `--reward-death-per-cell 0.1 --reward-lost-per-cell 0.1`. The return then adds up to final territory plus kills, like the real score. Runs: `runs/round0` (v1) vs `runs/round0_reward2` (v2). Training also keeps `best.pt`, the checkpoint with the best mean evaluation win rate.

3. **Kill farming.** On 40×40 with respawning opponents, the agent learned to sit at home and cut InvaderBots as they raided: 20–25 kills a game, while its land stayed at 30–50 cells. The invaders walked into its land predictably. Fix: invaders abort a raid when an enemy head gets close and don't start one next to an enemy. An agent circling at home now gets 0 kills against them.
4. **Wall hugging under lag.** An agent trained without lag turns on the last tile before walls: with 5% of turns landing a tile late, it dies to walls in 21 of 30 simulator games (0 of 30 without lag), and in 61 of 100 lives on the real server. Fix: train with `--lag-prob` / `--lag-range`.

### Training for the real game

`runs/realgame/dyna/seed0`, 600k steps (this replaced an earlier stage-2 run that was missing pausing, the rank bonus and the hunter bot):
* **Every episode is different:** board 24–72 tiles, 2–10 opponents, 0–10% lag, respawning opponents spawned anywhere. Bots draw new loop sizes and caution each life (`?jitter=true`).
* **Self-play league:** every 25k steps a frozen copy of the agent joins the opponent pool (newest 4 kept).
* **All observation options,** plus `--escape-features`: the escape margin, (nearest enemy's distance to my trail) − (my distance home). That's the game's basic safety rule.
* **Pausing** (`--allow-pause`): a 4th action, stop moving, at most 20 steps in a row.
* **Reward `score_rank`:** the real score (1 per tile, 500 per kill, land lost on death), plus up to 3 at the end for holding more land than the opponents. Creeping along with 1–2-tile loops ends up behind, so it no longer pays.
* **Memory** (`--memory 100`), and **stalker bots** that hunt exposed trails from anywhere on the map.
* **Hunter bots,** which chase exposed trails from 20 tiles away. Loops too big for how close enemies are get punished, so the escape-margin rule is learned from experience.
* **Learner:** γ 0.995, 3-step returns, dueling network, soft target, 1 gated planning update.
* **Evaluation** is fixed at 48×48, 6 opponents, 5% lag, and `best.pt` is chosen by evaluation return (real-score units).
* `--init-from` can warm-start a stage from an earlier checkpoint with the same inputs.

## Real game

Trained agents (and our bots) can play on the official, open-source Splix server
([jespertheend/splix](https://github.com/jespertheend/splix), MIT), running on your own machine, so no public servers or real players are involved.

```bash
# once: download the macOS server and web client into .splix-server/ (gitignored)
mkdir -p .splix-server && cd .splix-server
gh release download v0.19.0 -R jespertheend/splix -p splixGameServer_macos.tar -p client.tar
mkdir -p server client && tar -xf splixGameServer_macos.tar -C server && tar -xf client.tar -C client && cd ..

# terminal 1: the server (40x40 arena; default is 100x100)
.splix-server/server/splixGameServer -s 40

# terminal 2: an agent plays 10 lives against 4 bots, with a window showing what it sees
python -m splixrl.live --actor runs/round0_reward2/dyna/seed0/best.pt --lives 10 \
    --bots loop chaser loop random_safe --show --out runs/live/dyna.csv
```

`--bots` alone (no `--actor`) keeps bots playing until Ctrl-C. Then join yourself from a browser at
`https://splix.io/#ip=ws://localhost:8080` (the server's own instructions) and play against them.

How it works. Once per tile, `live.py` rebuilds a `SplixGame` snapshot from what the server sent and asks the actor for a move. The snapshot holds tiles, trails, and player positions dead-reckoned at 6 tiles/s. The actor sees it through the same observation code as in training.

**Lag.** The server drops turn messages while a tick runs slow, and it can apply a turn late or rewind a player. The client therefore checks that every turn comes back from the server, re-sends turns that don't (the server then rewinds to that tile), and logs turns, confirmations, re-sends and lost turns per life.

That cut Dyna's wall deaths on the server from 4/10 lives to 1/10: mean tiles went from 83 to 101 and mean time alive from 30 s to 36.5 s, at 50–60k training steps against 4 bots. To make agents robust to the rest, train with `--lag-prob 0.05`, where turns sometimes land a tile late. The world model then has to learn that moves are uncertain too.

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
splixrl/live.py          client for the real Splix server
splixrl/watch.py, play.py, viewer.py, render.py
tests/
```
