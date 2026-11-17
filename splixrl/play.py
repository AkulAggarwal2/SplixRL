"""Play the game yourself (you are purple, player 0).

    python -m splixrl.play                         # vs the loop bot
    python -m splixrl.play --opponents chaser --n-opponents 2
    python -m splixrl.play --opponents frozen:runs/round0/dyna/seed0/agent.pt
    python -m splixrl.play --turn-based            # the game waits for your key

Arrow keys or WASD steer (reversing is ignored, like the real game).
Space pauses, R restarts, Esc quits. By default the game moves 6 cells a
second, the speed of the real game.
"""

from __future__ import annotations

import argparse

from .env import SplixEnv
from .game import STRAIGHT, TURN_LEFT, TURN_RIGHT
from .opponents import DEFAULT_POOL
from .viewer import Viewer


def relative_action(heading: int, desired: int | None) -> int:
    """Absolute direction (0 up, 1 right, 2 down, 3 left) -> relative action."""
    if desired is None:
        return STRAIGHT
    diff = (desired - heading) % 4
    return {0: STRAIGHT, 1: TURN_RIGHT, 3: TURN_LEFT}.get(diff, STRAIGHT)  # 2 = reverse: ignored


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponents", nargs="+", default=["loop"], help=f"bots ({', '.join(DEFAULT_POOL)}, random) or frozen:<path>")
    ap.add_argument("--n-opponents", type=int, default=1)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fps", type=float, default=6.0, help="moves per second")
    ap.add_argument("--turn-based", action="store_true", help="advance one step per key press")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    env = SplixEnv(size=args.size, opponents=args.opponents, n_opponents=args.n_opponents, max_steps=args.max_steps)
    viewer = Viewer(env, "SplixRL - you are purple")
    pg = viewer.pg
    keymap = {pg.K_UP: 0, pg.K_w: 0, pg.K_RIGHT: 1, pg.K_d: 1, pg.K_DOWN: 2, pg.K_s: 2, pg.K_LEFT: 3, pg.K_a: 3}

    seed = args.seed
    obs, info = env.reset(seed=seed)
    queue: list[int] = []  # buffered direction presses, so quick double turns work
    paused, over, total = False, False, 0.0
    help_lines = ["arrows / WASD: steer", "space: pause   R: restart", "esc: quit"]
    running = True
    while running:
        step_now = False
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                running = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_r:
                    seed = None if seed is None else seed + 1
                    obs, info = env.reset(seed=seed)
                    queue, over, paused, total = [], False, False, 0.0
                elif e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key in keymap and not over:
                    if len(queue) < 2:
                        queue.append(keymap[e.key])
                    step_now = args.turn_based
        if not running:
            break

        if not over and not paused and (step_now or not args.turn_based):
            heading = env.game.players[0].heading
            desired = None
            while queue:  # skip presses that wouldn't change anything
                d = queue.pop(0)
                if relative_action(heading, d) != STRAIGHT:
                    desired = d
                    break
            obs, r, term, trunc, info = env.step(relative_action(heading, desired))
            total += r
            over = term or trunc

        names = ["you"] + env.opponent_names
        banner = None
        if over:
            banner = {"win": "You win!", "loss": "You lose", "draw": "Draw"}[info["outcome"]]
        elif paused:
            banner = "paused"
        footer = [f"reward so far {total:.1f}", ""] + help_lines
        viewer.draw(names, banner, footer)
        viewer.clock.tick(60 if args.turn_based else args.fps)
    viewer.close()


if __name__ == "__main__":
    main()
