"""Watch an agent or bot play, live or as a GIF.

    python -m splixrl.watch --actor runs/round0/dyna/seed0/agent.pt --opponents chaser
    python -m splixrl.watch --actor loop --opponents chaser --gif loop_vs_chaser.gif
"""

from __future__ import annotations

import argparse

from .env import SplixEnv
from .evaluation import make_actor
from .render import render_rgb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True, help="checkpoint path or bot name, plays as purple")
    ap.add_argument("--opponents", nargs="+", default=["loop"])
    ap.add_argument("--n-opponents", type=int, default=1)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--gif", help="write a GIF instead of opening a window")
    ap.add_argument("--cell", type=int, default=14, help="GIF pixels per cell")
    args = ap.parse_args(argv)

    env = SplixEnv(size=args.size, opponents=args.opponents, n_opponents=args.n_opponents, max_steps=args.max_steps)
    actor = make_actor(args.actor)

    if args.gif:
        from PIL import Image

        frames = []
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            actor.reset(env, args.seed + ep)
            frames.append(render_rgb(env.game, args.cell))
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(actor.act(env, obs))
                done = term or trunc
                frames.append(render_rgb(env.game, args.cell))
            frames.extend([frames[-1]] * int(args.fps))  # hold the last frame for a second
            print(f"episode {ep}: {info['outcome']}, territory {info['territory']}, {info['steps']} steps")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
        print(f"wrote {args.gif} ({len(frames)} frames)")
        return

    from .viewer import Viewer

    viewer = Viewer(env, f"SplixRL - {actor.name}")
    pg = viewer.pg
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        actor.reset(env, args.seed + ep)
        done = False
        while not done:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    viewer.close()
                    return
            obs, r, term, trunc, info = env.step(actor.act(env, obs))
            done = term or trunc
            viewer.draw([actor.name] + env.opponent_names)
            viewer.clock.tick(args.fps)
        viewer.draw([actor.name] + env.opponent_names, banner=info["outcome"])
        pg.time.wait(1500)
    viewer.close()


if __name__ == "__main__":
    main()
