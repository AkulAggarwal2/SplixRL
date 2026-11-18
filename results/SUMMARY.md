# SplixRL: Dyna results so far (for the report)

All numbers are from this repo's runs (`runs/`), and the figures are in `plots/`.

## 1. Round 0: Dyna on 20×20 vs one fixed bot

Setting: 20×20 board, 1 opponent drawn from {random_safe, loop, chaser}, 500 steps, Dyna (Double-DQN learner + world model, 3 imagined updates per real update), 200k steps, 3 seeds.

**Reward v1** (+0.1/cell, +3/kill, −3 death): all three seeds peaked around 50k steps (win rate vs loop 0.85–1.0, vs chaser 0.35–0.65), then **collapsed** (vs chaser 0–0.05 from 70k on).

**Diagnosis:** training return kept rising (peak territory 52 → 103 cells) while win rate fell. Big risky loops paid under the reward, even though dying loses everything in the game.

**Reward v2** (adds −0.1 per cell held at death or stolen, so the return tracks final territory): held up longer, but still drifted late. Figure: `plots/round0_reward_v1_vs_v2_win_rate.png`.

Best checkpoint per seed, 50 games per opponent:

| actor | vs random_safe | vs loop | vs chaser |
|---|---|---|---|
| LoopBot (baseline) | 0.90 | 0.48 | 0.10 |
| Dyna, v1 (seed 1 at its 50k peak) | 0.96 | 0.98 | 0.56 |
| Dyna, v2, seed 0 | 0.96 | 0.78 | 0.68 |
| Dyna, v2, seed 1 | 0.98 | 0.94 | 0.76 |
| Dyna, v2, seed 2 | 0.96 | 0.46 | 0.16 |

## 2. Ablation: planning makes Dyna's late collapse much worse (3 seeds each, reward v2)

Figure: `plots/ablation_planning_win_rate.png`. Win rate, mean ± std over 3 seeds:

| | 40k | 120k | 200k |
|---|---|---|---|
| Dyna (3 imagined updates per real one), vs loop | 0.65 ± 0.28 | 0.45 ± 0.39 | 0.48 ± 0.22 |
| same learner, planning off, vs loop | 0.83 ± 0.05 | 0.87 ± 0.08 | 0.68 ± 0.10 |
| Dyna, vs chaser | 0.58 ± 0.24 | 0.22 ± 0.31 | 0.05 ± 0.07 |
| planning off, vs chaser | 0.45 ± 0.04 | 0.57 ± 0.19 | 0.22 ± 0.09 |

Without planning the learner is better and much more consistent across seeds; its spread is a third to a fifth of Dyna's. Planning-off also slips against the chaser late (0.57 → 0.22), so planning amplifies an instability that is already there rather than causing it alone. The likely mechanism is imagined transitions from an imperfect world model pulling the Q-values off course. In the large-map setting the model's error on changed cells stayed around 0.85, so gated planning never switched on.

## 3. Problems found and fixed (Misha's "find problems and fix them")

1. **Turtling beat the bots.** Circling inside the start square beat LoopBot 48/50 and ChaserBot 44/50. Fixed by making bots leave home away from enemies and keep their distance. After the fix the turtler loses 48/50 and 47/50.
2. **Reward/objective mismatch** (section 1). Fixed with reward v2.
3. **Kill farming.** On 40×40 with respawning invaders, the agent sat at home cutting raiders: 20–25 kills a game, land 30–50 cells. Fixed by making invaders abort raids near enemies. A home-circler now gets 0 kills.
4. **Sim-to-real gap from lag.** On the real (locally hosted, open-source) Splix server, the 20×20 agent died to walls in 61 of 100 lives. In simulation: 0 of 30 games without lag, 21 of 30 with 5% of turns delayed one tile. The agent had learned to turn on the last tile. Fixed by training with simulated lag (`--lag-prob`/`--lag-range`), plus turn confirmation and resending in the client.
5. **Training too hard from scratch.** In the full setting (24–72 boards, up to 10 opponents including hunters/stalkers) the agent was still dying within 50–220 steps at 100k steps. Fixed with a curriculum (section 4).

## 4. Curriculum toward the real game

* **A:** the 20×20 v2 seed-1 agent.
* **B:** 24–40 boards, 1–4 opponents, minimap/memory/threat/escape inputs, pausing, lag up to 5%. 400k steps.
* **C:** full setting (running).

Inputs carry over by name: new inputs start at zero weight, so the transferred network computes the old agent's values on the inputs they share.

**Stage B checkpoints**, re-evaluated with 20 games per opponent (figure `plots/curriculum_B_selection.png`). Mean evaluation return went from 18.3 at the start of B to 25.8 at 350k (+41%):

| | loop | greedy | invader | chaser |
|---|---|---|---|---|
| step 0 | 10.9 | 32.3 | 23.0 | 7.0 |
| 350k | 26.3 | 51.3 | 23.3 | 2.4 |

It improved a lot against loop and greedy, but got **worse against the trail-hunting chaser**. That's the main open problem.

## 5. Engineering notes worth a sentence in the report

* The environment rules were checked against the open-source Splix server code (`gameServer/src/gameplay/Player.js`).
* Speed went from 27 to 300+ steps/s on the full setting, via numba pathfinding, capped spawns, per-seat league sampling and parallel games.
* A NaN bug from asynchronous GPU copies was caught by noticing identical evaluations, and is now guarded (`skipped_updates`).

## 6. Fine-tuning erodes the agent (seen three times)

Continuing training past an early peak consistently made agents worse:
* the 20×20 runs after ~50k steps;
* curriculum stage C: peak land vs greedy 251 → 72 cells over 350k steps;
* fine-tune B2 (turn/pause costs, learning rate 2.5e-4). It did fix the behavior: idle pausing 20% → 1–8% of steps, zigzag turns 43% → 20–33%. But mean return fell 25.8 → 9.6 and peak land 110 → 71 (80 games per checkpoint).

Fine-tune B3 repeats B2 with learning rate 5e-5 and a 2.5× slower target network, to test whether a gentler learner adds behavior without erasing what it knows.
