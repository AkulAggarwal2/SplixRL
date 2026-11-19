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

**B3 result: a gentler learner fixes the erosion.** Same setup and costs as B2, but learning rate 5e-5 (was 2.5e-4), soft target τ 0.002 (was 0.005) and ε ≤ 0.08. 80 games per checkpoint, same seeds:

| | return | peak land | game length | kills | idle pausing | zigzag turns |
|---|---|---|---|---|---|---|
| previous best (B at 350k) | 25.8 | 110 | 324 | 5.3 | 20% | 43% |
| B2 at 250k (lr 2.5e-4) | 9.6 | 71 | 228 | 2.1 | 1% | 29% |
| **B3 at 100k (lr 5e-5)** | **45.0** | **147** | **467** | **9.0** | **2%** | 38% |

Every B3 checkpoint from 25k on beat the previous best. B3 at 100k is now `runs/curriculum/BEST_REALGAME_AGENT.pt` (the old one is `..._v1.pt`). Stage C is being redone with the same gentle settings (`runs/curriculum/C2`).

## 7. Options agent: plans instead of tiles

Per-tile agents made wiggly, staircase and runaway trails, because a clean loop needs 20–40 per-tile decisions that all agree. The options agent (`splixrl/options.py`) picks among 14 plans: 10 rectangular loops, home, reclaim, attack and ambush. Scripted controllers carry the plans out. One shared, lag-aware safety rule aborts a plan and goes home when an enemy could cut the trail first. Learning is over the semi-MDP (Sutton, Precup & Singh 1999), with the bootstrap discount γ^k per option. The agent warm-starts from the per-tile agent's learned vision.

Same 60 games (48×48, 6 opponents including hunters and stalkers, 5% lag):

| | return | peak land | game length | survived | kills |
|---|---|---|---|---|---|
| per-tile best | 25.0 | 147 | 291 | 0% | 3.9 |
| **options agent (15k decisions)** | **68.6** | **239** | **594** | **12%** | 2.6 |

Even random option choices reached a mean peak of 317 cells on 40×40 with 3 bots. Most of the gain comes from the plans, and learning chooses between them. Training-time evaluations peaked at 15k decisions and then drifted, as the per-tile runs did.

**Defend reflex.** Every option now cuts an enemy trail lying on the agent's own land within 15 tiles, moving only over its own land. Same 36 games each, 40×40, 4 opponents:

| vs | without | with |
|---|---|---|
| invader | return 63.6, 2.8 kills, peak 225 | return 103.3, 9.3 kills, peak 308 |
| greedy | return 76.9, 5.9 kills, peak 325 | return 111.9, 16.0 kills, peak 403 |
| loop | return 85.4, 2.0 kills, peak 384 | return 103.1, 7.3 kills, peak 449 |

With the defend reflex, the same 60 hard games (48×48, 6 opponents incl. hunters and stalkers, 5% lag): the options agent at 15k decisions scores return 83.9, peak 306, 637 steps, 8.1 kills, 13% survival. That's 3.4× the per-tile agent's 25.0. Later checkpoints of the same run (135k, 150k) were within noise of it (76–79), with longer games and more kills.

**Options run B** (defend reflex built in during training, learning rate 5e-5) beat the earlier best on the same 60 hard games: return 89.1 vs 74.4, game length 733 vs 575, kills 11.5 vs 7.4, survival 23% vs 15%, peak land 262 vs 286. Note the earlier best scored 83.9 on its first run of these 60 games, which shows how noisy a 60-game evaluation still is. B is now `runs/options/BEST_OPTIONS_AGENT.pt`, and run C adds the 2 adaptive options on top of it.

**Safety cushion.** Against humans the agent was caught on long trips through enemy territory: the abort rule (escape margin ≤ 2 tiles) assumes the enemy only reacts once it's close. Sweep on the same 60 hard games (best options agent):

| cushion | return | peak land | game length | survived | trail-cut deaths | vs hunter | vs stalker |
|---|---|---|---|---|---|---|---|
| 2 | 89.1 | 262 | 733 | 23% | 42/60 | 4.0 | 18.7 |
| 4 | 100.9 | 358 | 900 | 18% | 33/60 | 32.6 | 84.1 |
| **6** | **118.4** | **394** | **1119** | **33%** | **21/60** | **40.0** | **91.9** |
| 9 | 101.9 | 384 | 1104 | 32% | 20/60 | 39.8 | 83.4 |

6 tiles is now the default.

**Options run C** (adds open_loop and safe_loop) vs run B, both with the 6-tile cushion and remembered enemies, same 60 hard games: return 119.1 vs 110.6, survival 48% vs 40%, kills 21.0 vs 17.9, peak land 359 vs 385. C is now `BEST_OPTIONS_AGENT.pt` (B kept as `_B.pt`). Most decisions are short ones (home 36%, ambush 34%), and the adaptive safe_loop is used 3% of the time. Run D adds steal_edge (17 options).

**Race-based defend reflex.** The reflex now also fires while the agent is out on a plan: it goes home first, then cuts. It goes for any enemy trail on its land that it can reach before the owner gets back to their own land, instead of only within 15 tiles. On the 60 hard games it's neutral (return 114.9 vs 119.1, survival 47% vs 48%, within noise): the bots rarely run long trails through the agent's land, so the simulator barely exercises the case it is meant for (long encirclements by humans).

**Loop sizing, memory and idling** (same 60 hard games):
* In crowded games 74–86% of loops were aborted partway, capturing about 4–8 tiles each, almost always with an enemy in view. Loops are now **sized to the safety budget before they start** (out + along ≤ 0.8 × (nearest enemy reach − cushion)). That gave survival 57% vs 45%, trail-cut deaths 13 vs 15, kills 22.5 vs 19.9, at the same return (124).
* The remembered-enemy guess was too pessimistic at full speed. At half speed (an unseen enemy covers half the distance it could), return was 117.6 vs 93.5 at full speed.
* An **idle guard** (expand instead of waiting when no enemy is near) made it *worse*: survival 43–45% vs 57%. The `stalker` bot sees exposed trails anywhere on the map, so the simulator rewards waiting even when nobody is in view. Real players can't do that, which is a sim-to-real mismatch, and the likely reason the agent idles against humans. The fix belongs in training: a stalker limited to human-like vision and reaction time.

**Thin strips** (against people the agent went a few tiles out, turned back beside its own trail and captured 2-wide strips with nobody in view). Two causes: remembered enemies were treated as possibly right next to the trail even when they would have had to be inside the agent's view to be there, and loops shrunk to fit the budget could become 1-wide. Fixes: an enemy not in view now is assumed to be at least 5 tiles inside the edge of the view (`UNSEEN_MARGIN`), and budget-sized loops are kept close to square (each side ≥ 2). Same run C agent, crowded = the 60 hard games (48×48, 6 opponents), sparse = 30 games on 96×96 with 4 opponents:

| | median capture (sparse) | loops cut short (sparse) | survival (90 games) | trail-cut deaths | peak land (crowded) | return crowded / sparse |
|---|---|---|---|---|---|---|
| old rules | 4 tiles | 31% | 38/90 | 22 | 367 | 103.9 / 161.8 |
| view bound at the edge, square loops | 9 | 21% | 33/90 | 28 | 399 | 105.2 / 169.7 |
| **view bound 5 tiles in, square loops** | **10** | **18%** | **40/90** | **21** | **447** | **119.7 / 158.4** |

**Run D** (adds steal_edge, 17 options) vs run C, current rules, same 60 hard games: D's best checkpoint scores return 121.5 vs 110.9 at the same survival (33%) but more trail-cut deaths (24 vs 18). It isn't clearly better, so C stays `BEST_OPTIONS_AGENT.pt`.

**Human-like hunter and the idle guard again.** New bot `human`: a ChaserBot that only sees trails on its own screen (25 tiles), starts chasing after a trail has been in sight for 4 steps, and chases up to 40 tiles away. With `human` in place of `stalker`, the idle guard (expand instead of waiting when nobody is in view) now helps, which supports the sim-to-real explanation above. Run C agent, current rules:

| | return | peak land | survival | trail-cut deaths |
|---|---|---|---|---|
| sparse (96×96, 4 opponents, 30 games), guard off | 168.7 | 1281 | 40% | 5 |
| sparse, guard on | **262.3** | **1853** | **53%** | 7 |
| crowded (48×48, 6 opponents, 60 games), guard off | 118.9 | 454 | 42% | 20 |
| crowded, guard on | 125.8 | 479 | 42% | 17 |

The guard is now on by default.

**Race features** (`--race-features`): six more inputs giving the enemies' side of the race. For the visible enemy whose trail the agent could cut with the most time to spare: (its distance back to its land − my distance to its trail), its distance home, its trail length, my distance to its trail, and the direction to it. The agent already had its own side (trail length, distance home, escape margin).

**Run E**: from run C with the current rules (idle guard on), race features, `human` instead of `stalker`, and bigger maps (32–96). Also fixed: a warm start whose inputs changed used to reset an options agent's learned option values; it now keeps them and only adds rows for new options.

**Odd territory shapes.** Each loop on its own is nearly a rectangle (median capture fills ~0.9 of its bounding box). The territory as a whole looked strange for two reasons. First, many 1–2 tile captures: 47% of captures on the crowded map, mostly loops turned back by the safety rule a step after leaving, which left notches along the edge. Second, loops jutted out from wherever the agent left its land. Fixes: don't start a loop the safety budget can't fit (`NO_DOOMED_LOOPS`, needs room for ≥ 4 tiles), and turn the side leg toward the agent's own land so the loop lies along the edge (`HUG_EDGE`). Run C agent, crowded = 48 games on 48×48 with 6 opponents, sparse = 24 games on 96×96 with 4 opponents (bots incl. `human`):

| | return crowded / sparse | survival crowded / sparse | trail-cut deaths | peak land sparse | 1–2 tile captures sparse | territory fill of bounding box sparse |
|---|---|---|---|---|---|---|
| neither | 129.7 / 280.0 | 44% / 71% | 18 | 1897 | 27% | 0.43 |
| no doomed loops only | 108.9 / 283.6 | 38% / 50% | 18 | 2045 | 15% | 0.44 |
| **both** | **133.1 / 305.7** | **56% / 67%** | **7** | **2252** | **15%** | **0.50** |

The guard alone hurt: without hugging, the loops it does start jut out, so they take longer and get caught. Both are on. The territory still fills only half its bounding box; a plan that lays long strips along a whole side would be the next step, and needs training.

**Run E vs run C** (current rules, bots incl. `human`; crowded = 60 games on 48×48 with 6 opponents, sparse = 30 games on 96×96 with 4 opponents). E's `best.pt` is its 60k checkpoint.

| | return crowded / sparse | peak land crowded / sparse | survival crowded / sparse | trail-cut deaths |
|---|---|---|---|---|
| **run C** | 102.8 / **318.6** | 438 / **2337** | 45% / **87%** | 14 |
| E at 60k | 111.9 / 288.8 | 481 / 2047 | 42% / 70% | 13 |
| E at 100k | **124.5** / 298.1 | **530** / 2226 | 43% / 73% | 15 |

E is better in crowded games and worse on the sparse map, and survives less overall (48 vs 53 of 90). That isn't clearly better, so C stays `BEST_OPTIONS_AGENT.pt`. With the current rules, C on the sparse map is now at 87% survival and 2337 peak land, vs 40% and 1281 before the idle-guard and shape fixes, so the rule changes did most of the work. The race features start at zero weight, and 100k decisions of gentle fine-tuning was apparently not enough for the agent to make use of them.

**Surviving, killing safely, strips (Nov 19).** Measured first: most of the agent's deaths were not enemies but walls. 5% of turns land a tile late, and the agent turned on the last tile before a wall. Kills already came mostly from cutting trails on its own land. Changes, each tested on fixed games (crowded = 48 games on 48×48 with 6 opponents, sparse = 24 games on 96×96 with 4, bots incl. `human`), run C agent:
* `LAG_GUARD`: never end a move facing a wall or my own trail from right next to it (turn one tile early). Crowded deaths 21 → 10 of 48 and wall deaths to 0, survival 56% → 79%, return 128 → 145, kills 17.2 → 18.8. Sparse deaths 9 → 4 of 24, return 282 → 320.
* `SAFE_ATTACK`: attack only races it wins (reaches the trail ≥ 2 steps before the owner gets home or could reach its trail); during such a race the target doesn't count in the safety rule, everyone else does. Deaths 10 → 8 (crowded), 4 → 2 (sparse), kills about the same. `KILL_REFLEX` (take any such race within 12 tiles from home) added a kill per game but almost doubled deaths: off.
* `STRIPS`: when a loop is due, lay a d × L strip flush along a straight stretch of my edge, scored by new land per step × (area / bounding box)². Crowded return 141 → 160, deaths 9 → 7, peak 589 → 689, fill 0.58 → 0.64. Sparse return 262 → 352, peak 2097 → 2645, median capture 12 → 24 tiles, fill 0.52 → 0.67. (A first version that extended the outermost edge stretched peninsulas and was worse.)
* `VIEW_DEFEND`: the defend reflex sees only enemy trails on screen, as the real client does (no measurable change in these games).
* `BREAK_OUT`: if all my land is a hole in one enemy's land, their next loop anywhere takes all of it and kills me. A break-out reflex cuts a strip to the open. It only helps a little (swallowed 32 → 30 of 39 with a 4-tile ring): bots close a loop every few steps.
* `LAND_ALARM` (tile count falls with nobody in view → walk toward my land out of view) cost land in testing (sparse return 352 → 294): off.
* `LAND_ALARM`, second version (guided by what the minimap shows): when the tile count falls with nothing in view, walk over my land to border that other land has grown toward in the last 30 steps, else border other land touches, else my land out of view. At most one patrol per 60 steps (the first version started 135 patrols in 20 theft games, this one 13). Crowded: return 159.6 → 160.5, deaths 8 → 6 of 48. Sparse: 351.6 → 336.4, deaths 2 → 3 of 24. Both within noise. With strips on, invaders in the theft scenario no longer get any land (0 lost with or without it), so the simulator barely produces the out-of-view theft seen against people. It's on, because it's aimed at that case and costs nothing measurable.
