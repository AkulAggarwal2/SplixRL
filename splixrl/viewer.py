"""pygame window shared by manual play and watching agents."""

from __future__ import annotations

import numpy as np

from .env import SplixEnv
from .render import player_color, render_rgb

PANEL_W = 260


class Viewer:
    def __init__(self, env: SplixEnv, title: str = "SplixRL", max_board_px: int = 760):
        import pygame

        self.pg = pygame
        pygame.init()
        self.env = env
        self.cell = max(6, min(32, max_board_px // env.size))
        board_px = env.size * self.cell + 4
        self.screen = pygame.display.set_mode((board_px + PANEL_W, max(board_px, 420)))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Helvetica", 18)
        self.big = pygame.font.SysFont("Helvetica", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.board_px = board_px

    def draw(self, names: list[str], banner: str | None = None, footer: list[str] | None = None) -> None:
        pg, env = self.pg, self.env
        g = env.game
        self.screen.fill((32, 29, 27))
        img = render_rgb(g, self.cell)
        self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), (0, 0))

        x, y = self.board_px + 16, 14
        def text(s, color=(235, 235, 235), font=None):
            nonlocal y
            surf = (font or self.font).render(s, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 6

        text(f"step {g.t} / {env.max_steps}")
        y += 6
        total = g.size * g.size
        for p in g.players:
            col = tuple(int(v) for v in player_color(p.pid))
            label = names[p.pid] if p.pid < len(names) else f"player {p.pid}"
            status = "" if p.alive else f"  (dead: {p.death_cause})"
            text(f"{label}{status}", col)
            text(f"   {g.territory(p.pid)} cells  {100 * g.territory(p.pid) / total:.1f}%   kills {p.kills}")
        if banner:
            y += 10
            text(banner, (255, 255, 255), self.big)
        for line in footer or []:
            text(line, (170, 170, 170))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()
