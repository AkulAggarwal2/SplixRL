"""Options (temporally extended actions): an agent that picks plans, not tiles.

A per-tile agent has to make 20-40 decisions that all agree to draw one clean
loop; with limited training some of them don't, which shows up as wiggly,
staircase or runaway trails. Here the agent picks an option ("draw a 7x5 loop
turning left", "go home", "reclaim", "attack", "ambush") and a scripted
controller carries it out tile by tile. The learner decides what matters:
when to expand, how far, which way, when to attack, defend or go home.

This is the options framework (Sutton, Precup & Singh 1999). Learning over
options is a semi-MDP: each decision's reward is the discounted sum over the
option's steps, and the bootstrap discount is gamma ** (option length).

Every option shares one safety rule: while a trail is out, if the escape
margin (nearest visible enemy's distance to my trail minus my distance home)
falls to ``safety + lag_slack * lag``, abort and take the shortest safe way
home. The default cushion of 6 tiles was picked on 60 hard test games
(48x48, 6 opponents incl. hunters/stalkers): return 89 at 2 tiles, 101 at 4,
118 at 6, 102 at 9 -- humans and hunters head for your trail as soon as you
leave, so the race has to be called early. More lag means acting earlier, because a turn can land a tile late.
No option runs longer than MAX_OPTION_STEPS, and a loop always ends at home.

Every option also shares a defend reflex: when an enemy trail lies on my own
land within DEFEND_RADIUS, go cut it, moving only over my own land (where I
can't be killed). An enemy eating into my territory is the easiest kill in
the game, so it is not left to the learner to notice.
"""

from __future__ import annotations

import numpy as np

from .game import DIRS, EMPTY, PAUSE, STRAIGHT, TURN_LEFT, TURN_RIGHT, SplixGame
from .opponents.base import bfs_first_action, safe_actions

LOOP_SIZES = ((3, 3), (5, 4), (7, 5), (10, 6), (14, 8))  # (tiles out, tiles along)
OPTIONS: tuple[tuple, ...] = tuple(
    ("loop", out, side, turn) for out, side in LOOP_SIZES for turn in (TURN_LEFT, TURN_RIGHT)
) + (("home",), ("reclaim",), ("attack",), ("ambush",)) + (
    # adaptive plans (appended last, so older options agents can be grown into this set):
    ("open_loop",),  # head for the most open direction, loop sized to the free space there
    ("safe_loop",),  # same direction choice, loop sized to stay ahead of the nearest enemy
    ("steal_edge",),  # bite a strip off the neighbouring enemy land (land just taken from me first)
)
N_OPTIONS = len(OPTIONS)
OPTION_NAMES = tuple(
    f"loop {o[1]}x{o[2]} {'L' if o[3] == TURN_LEFT else 'R'}" if o[0] == "loop" else o[0] for o in OPTIONS
)


def _open_space(game: SplixGame, pid: int, h: int, reach: int = 18) -> float:
    """How attractive direction h is: free tiles in a window ahead, minus enemies there."""
    p = game.players[pid]
    n = game.size
    dr, dc = int(DIRS[h, 0]), int(DIRS[h, 1])
    half = reach // 2
    cr, cc = p.pos[0] + dr * (half + 2), p.pos[1] + dc * (half + 2)
    r0, r1, c0, c1 = max(0, cr - half), min(n, cr + half + 1), max(0, cc - half), min(n, cc + half + 1)
    if r0 >= r1 or c0 >= c1:
        return -1e9
    free = (game.owner[r0:r1, c0:c1] == EMPTY) & (game.trail[r0:r1, c0:c1] == EMPTY)
    score = float(free.sum())
    for r, c in _enemy_heads(game, pid):
        if r0 <= r < r1 and c0 <= c < c1:
            score -= 60.0
    return score


def _recently_lost(game: SplixGame, pid: int) -> np.ndarray:
    """My land that someone else owns now, from the observation memory (if any)."""
    mem = (getattr(game, "obs_memory", None) or {}).get(pid)
    if not mem or not mem.get("maps"):
        return np.zeros_like(game.owner, dtype=bool)
    old_own = mem["maps"][0][2]
    if old_own.shape != game.owner.shape:
        return np.zeros_like(game.owner, dtype=bool)
    return old_own & (game.owner != pid) & (game.owner != EMPTY)


def _enemy_edge(game: SplixGame, pid: int, h: int, lost: np.ndarray, reach: int = 8) -> float:
    """How good direction h is for biting into enemy land: enemy-owned tiles
    (land just taken from me counts triple) in a strip next to me, minus enemies."""
    p = game.players[pid]
    n = game.size
    dr, dc = int(DIRS[h, 0]), int(DIRS[h, 1])
    cr, cc = p.pos[0] + dr * (reach // 2 + 2), p.pos[1] + dc * (reach // 2 + 2)
    half = reach // 2 + 2
    r0, r1, c0, c1 = max(0, cr - half), min(n, cr + half + 1), max(0, cc - half), min(n, cc + half + 1)
    if r0 >= r1 or c0 >= c1:
        return -1e9
    window = game.owner[r0:r1, c0:c1]
    enemy = (window != EMPTY) & (window != pid)
    score = float(enemy.sum()) + 2.0 * float((lost[r0:r1, c0:c1] & enemy).sum())
    for r, c in _enemy_heads(game, pid):
        if abs(r - cr) + abs(c - cc) <= reach + 4:
            score -= 80.0
    return score


def _free_run(game: SplixGame, pid: int, h: int, limit: int = 20) -> int:
    """Free tiles straight ahead in direction h, starting past my own land."""
    p = game.players[pid]
    r, c = p.pos
    dr, dc = int(DIRS[h, 0]), int(DIRS[h, 1])
    run, seen_free = 0, False
    for _ in range(limit + 30):
        r, c = r + dr, c + dc
        if not (0 <= r < game.size and 0 <= c < game.size):
            break
        if game.owner[r, c] == pid and not seen_free:
            continue  # still crossing my own land
        if game.trail[r, c] != EMPTY or (game.owner[r, c] != EMPTY and game.owner[r, c] != pid):
            break
        seen_free = True
        run += 1
        if run >= limit:
            break
    return run

MAX_OPTION_STEPS = 90
DEFEND_RADIUS = 15  # cut an enemy trail lying on my land when it is this close
VIEW = 25  # options only look this far (the server's range)
ORDER = [STRAIGHT, TURN_LEFT, TURN_RIGHT]


def _enemy_heads(game: SplixGame, pid: int) -> np.ndarray:
    p = game.players[pid]
    heads = [q.pos for q in game.players
             if q.alive and q.pid != pid and max(abs(q.pos[0] - p.pos[0]), abs(q.pos[1] - p.pos[1])) <= VIEW]
    return np.array(heads, dtype=np.int64).reshape(-1, 2)


MEMORY_TRUST = 30  # remembered enemies older than this many steps are ignored
MEMORY_SPEED = 0.5  # an unseen enemy is assumed to have closed in this many tiles per step since seen
# Loops are sized to fit the safety budget before they start (out + along <=
# BUDGET_SHARE * (nearest enemy reach - cushion)), so they finish as rectangles
# instead of being aborted halfway, which captured almost nothing.
BUDGET_SHARE = 0.8


def _enemy_reach(game: SplixGame, pid: int, exposed: np.ndarray) -> int | None:
    """Fewest steps any enemy could need to touch ``exposed`` cells.

    Visible heads count at their distance. Enemies seen recently but now out of
    sight count too: last seen ``age`` steps ago at P, they could be ``age``
    tiles closer by now (the observation memory, see obs.memory_planes).
    """
    best = None
    heads = _enemy_heads(game, pid)
    if len(heads):
        best = int(np.abs(exposed[:, None, :] - heads[None, :, :]).sum(2).min())
    mem = (getattr(game, "obs_memory", None) or {}).get(pid)
    if mem:
        clock = int(getattr(game, "clock", game.t))
        for pos, t in mem.get("seen", {}).values():
            age = clock - t
            if 0 < age <= MEMORY_TRUST:
                d = int(np.abs(exposed - np.array(pos)).sum(1).min() - MEMORY_SPEED * age)
                d = max(0, d)
                best = d if best is None else min(best, d)
    return best


def escape_margin(game: SplixGame, pid: int) -> float:
    """(fewest steps an enemy -- visible, or seen recently -- needs to reach my
    trail or head) - (my distance home)."""
    p = game.players[pid]
    exposed = np.array([*p.trail, p.pos], dtype=np.int64).reshape(-1, 2)
    reach = _enemy_reach(game, pid, exposed)
    if reach is None:
        return 99.0
    own = game.owner == pid
    if own[p.pos]:
        home = 0
    else:
        cells = np.argwhere(own)
        home = int(np.abs(cells - np.array(p.pos)).sum(1).min()) if len(cells) else 99
    return float(reach - home)


def enemy_near(game: SplixGame, pid: int, radius: int) -> bool:
    heads = _enemy_heads(game, pid)
    p = game.players[pid]
    return bool(len(heads)) and int(np.abs(heads - np.array(p.pos)).sum(1).min()) <= radius


class OptionRunner:
    """Carries out one option for one player. ``act`` returns a primitive
    action each step, or None once the option is over."""

    def __init__(self, option: int, rng: np.random.Generator, lag: float = 0.0, can_pause: bool = True,
                 safety: float = 6.0, lag_slack: float = 30.0, defend: bool = True):
        self.kind, *self.params = OPTIONS[option]
        self.option = option
        self.rng = rng
        self.can_pause = can_pause
        self.threshold = safety + lag_slack * lag
        self.steps = 0
        self.phase: str | None = None
        self.left_home = False
        self.done = False
        self.aborted = False  # the safety rule fired
        self.face: int | None = None  # world direction an adaptive loop heads out in
        self.defend = defend
        self.idle_guard = False  # if on: never stand around on my land with nobody in view (hurt in simulation, see SUMMARY)
        self.idled_off = False
        self.defended = 0  # steps spent on the defend reflex

    # -- helpers ----------------------------------------------------------------
    def _near(self, game, pid, radius=2):
        n = game.size
        m = np.zeros((n, n), bool)
        for r, c in _enemy_heads(game, pid):
            m[max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1] = True
        return m

    def _route(self, game, pid, goal, blocked):
        near = self._near(game, pid)
        own = game.owner == pid
        a = bfs_first_action(game, pid, goal & ~near, blocked | (near & ~own), ORDER)
        if a is None:
            a = bfs_first_action(game, pid, goal, blocked, ORDER)
        return a

    def _go_home(self, game, pid):
        self.phase = "home"
        a = self._route(game, pid, game.owner == pid, game.trail == pid)
        if a is None:
            opts = safe_actions(game, pid)
            return int(self.rng.choice(opts)) if opts else STRAIGHT
        return a

    def _checked(self, game, pid, a):
        """Use action a unless it is fatal (wall / own trail); then head home."""
        if a is None or not game.is_safe(pid, a):
            return self._go_home(game, pid)
        return a

    def _step_inside(self, game, pid):
        """One safe step that stays on my land (used when an option has nothing to do)."""
        self.done = True
        for a in ORDER:
            if game.is_safe(pid, a) and game.owner[game.next_cell(pid, a)[0]] == pid:
                return a
        opts = safe_actions(game, pid)
        return opts[0] if opts else STRAIGHT

    # -- main -------------------------------------------------------------------
    def act(self, game: SplixGame, pid: int) -> int | None:
        if self.done:
            return None
        p = game.players[pid]
        if not p.alive:
            self.done = True
            return None
        at_home = bool(game.owner[p.pos] == pid)
        if not at_home:
            self.left_home = True
        elif self.left_home:  # back on my land after a trip: the loop closed
            self.done = True
            return None
        if self.steps >= MAX_OPTION_STEPS and at_home:
            self.done = True
            return None
        self.steps += 1
        if not at_home and (self.steps > MAX_OPTION_STEPS or
                            (p.trail and escape_margin(game, pid) <= self.threshold)):
            if self.phase != "home":
                self.aborted = True
            return self._go_home(game, pid)
        if self.defend:
            invader = self._invasion(game, pid, at_home)
            if invader is not None:
                self.defended += 1
                if not at_home:  # out on a plan: get home first, then cut it from my own land
                    return self._go_home(game, pid)
                a = bfs_first_action(game, pid, invader, (game.owner != pid) | (game.trail == pid), ORDER)
                if a is not None and game.is_safe(pid, a):
                    return a
        if self.phase == "home":
            return self._go_home(game, pid)
        if at_home and self.idle_guard and self.kind in ("home", "ambush", "attack") and self.phase is None \
                and not self._threat_near(game, pid):
            # nothing to wait for or attack: expand instead of idling
            self.kind = "open_loop"
            self.idled_off = True
        return getattr(self, "_opt_" + self.kind)(game, pid, at_home)

    def _threat_near(self, game, pid, radius: int = VIEW) -> bool:
        p = game.players[pid]
        reach = _enemy_reach(game, pid, np.array([p.pos], dtype=np.int64))
        enemy_trail = (game.trail != EMPTY) & (game.trail != pid)
        if enemy_trail.any():
            d = int(np.abs(np.argwhere(enemy_trail) - np.array(p.pos)).sum(1).min())
            if d <= VIEW:
                return True  # something to attack
        return reach is not None and reach <= radius

    def _invasion(self, game, pid, at_home):
        """Enemy trail cells on my land worth going for, or None.

        A trail across my land is either a free kill or a big land grab
        against me. Go for it when it is close (DEFEND_RADIUS) or when I can
        win the race: my way to it (home first, if I'm out) is shorter than
        its owner's way back to their own land."""
        own = game.owner == pid
        invader = (game.trail != EMPTY) & (game.trail != pid) & own
        if not invader.any():
            return None
        p = game.players[pid]
        me = np.array(p.pos)
        detour = 0
        if not at_home:
            cells = np.argwhere(own)
            detour = int(np.abs(cells - me).sum(1).min()) if len(cells) else 99
        worth = np.zeros_like(invader)
        for q in game.players:
            if not q.alive or q.pid == pid:
                continue
            theirs = invader & (game.trail == q.pid)
            if not theirs.any():
                continue
            mine_d = detour + int(np.abs(np.argwhere(theirs) - me).sum(1).min())
            their_land = np.argwhere(game.owner == q.pid)
            home_d = int(np.abs(their_land - np.array(q.pos)).sum(1).min()) if len(their_land) else 99
            if mine_d <= DEFEND_RADIUS or mine_d <= home_d + 2:
                worth |= theirs
        return worth if worth.any() else None

    def _defend_old(self, game, pid):  # kept for reference; replaced by _invasion
        own = game.owner == pid
        invader = (game.trail != EMPTY) & (game.trail != pid) & own
        if not invader.any():
            return None
        p = game.players[pid]
        cells = np.argwhere(invader)
        if int(np.abs(cells - np.array(p.pos)).sum(1).min()) > DEFEND_RADIUS:
            return None
        # stay on my land all the way (enemy trail cells on my land are the goal)
        a = bfs_first_action(game, pid, invader, ~own | (game.trail == pid), ORDER)
        return a if a is not None and game.is_safe(pid, a) else None

    # -- the options -----------------------------------------------------------
    def _plan_adaptive(self, game, pid):
        p = game.players[pid]
        scores = [_open_space(game, pid, h) for h in range(4)]
        d = int(np.argmax(scores))
        left_h, right_h = (d + 3) % 4, (d + 1) % 4
        turn = TURN_LEFT if scores[left_h] >= scores[right_h] else TURN_RIGHT
        run = _free_run(game, pid, d)
        side_run = _free_run(game, pid, left_h if turn == TURN_LEFT else right_h)
        if self.kind == "open_loop":
            out = int(np.clip(0.6 * run, 3, 14))
        else:  # safe_loop: stay ahead of the nearest enemy
            heads = _enemy_heads(game, pid)
            e = int(np.abs(heads - np.array(p.pos)).sum(1).min()) if len(heads) else 99
            out = int(np.clip((e - 4) // 2, 2, 10))
            out = min(out, max(2, int(0.6 * run)))
        side = int(np.clip(min(0.8 * out, max(3, 0.6 * side_run)), 2, 10))
        self.kind, self.params, self.face = "loop", [out, side, turn], d

    def _opt_open_loop(self, game, pid, at_home):
        self._plan_adaptive(game, pid)
        return self._opt_loop(game, pid, at_home)

    _opt_safe_loop = _opt_open_loop

    def _opt_steal_edge(self, game, pid, at_home):
        """A short, wide loop into the enemy land next to mine: in 3-4 tiles,
        along the border, back. Prefers land that was just taken from me."""
        lost = _recently_lost(game, pid)
        scores = [_enemy_edge(game, pid, h, lost) for h in range(4)]
        d = int(np.argmax(scores))
        if scores[d] <= 3:  # no enemy land worth biting nearby: a small safe loop instead
            self.kind = "safe_loop"
            return self._opt_open_loop(game, pid, at_home)
        left_h, right_h = (d + 3) % 4, (d + 1) % 4
        turn = TURN_LEFT if _enemy_edge(game, pid, left_h, lost) >= _enemy_edge(game, pid, right_h, lost) else TURN_RIGHT
        self.kind, self.params, self.face = "loop", [4, 7, turn], d
        return self._opt_loop(game, pid, at_home)

    def _opt_loop(self, game, pid, at_home):
        out, side, turn = self.params
        p = game.players[pid]
        if at_home:  # leave on the side away from enemies (or toward the chosen direction)
            outside = (game.owner != pid) & (game.trail < 0)
            if self.face is not None:
                dr, dc = int(DIRS[self.face, 0]), int(DIRS[self.face, 1])
                rr, cc = np.mgrid[0 : game.size, 0 : game.size]
                ahead = (rr - p.pos[0]) * dr + (cc - p.pos[1]) * dc >= 1
                a = self._route(game, pid, outside & ahead, np.zeros_like(outside))
                if a is not None:
                    return self._checked(game, pid, a)
            return self._checked(game, pid, self._route(game, pid, outside, np.zeros_like(outside)))
        if self.phase is None and self.face is not None and p.heading != self.face:
            diff = (self.face - p.heading) % 4  # turn to face the chosen direction first
            if not p.trail or len(p.trail) <= 2:
                return self._checked(game, pid, TURN_RIGHT if diff in (1, 2) else TURN_LEFT)
        if self.phase is None:  # first tile outside: plan the rectangle from here
            out, side = self._fit_budget(game, pid, out, side)
            self.params = [out, side, turn]
            self.phase, self.left = "out", out - 1
        if self.phase == "out":
            if self.left > 0:
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
            self.phase, self.left = "side", side - 1
            return self._checked(game, pid, turn)
        if self.phase == "side":
            if self.left > 0:
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
        return self._go_home(game, pid)

    def _fit_budget(self, game, pid, out, side):
        """Shrink a loop (keeping its shape) so it fits the time before an enemy could arrive."""
        p = game.players[pid]
        reach = _enemy_reach(game, pid, np.array([p.pos], dtype=np.int64))
        if reach is None:
            return out, side
        budget = BUDGET_SHARE * (reach - self.threshold)
        if out + side <= budget:
            return out, side
        scale = max(0.0, budget) / (out + side)
        return max(1, int(out * scale)), max(1, int(side * scale))

    def _opt_home(self, game, pid, at_home):
        return self._step_inside(game, pid) if at_home else self._go_home(game, pid)

    def _opt_reclaim(self, game, pid, at_home):
        from .obs import cheap_land

        p = game.players[pid]
        if self.phase is None:
            from scipy.ndimage import binary_fill_holes

            own = game.owner == pid
            cheap = cheap_land(game, own, p.pos)
            # enemy pockets enclosed by my land are cheap too: one short trip fills them
            pockets = binary_fill_holes(own) & ~own & (game.owner != EMPTY)
            r, c = p.pos
            near = np.zeros_like(own)
            near[max(0, r - VIEW) : r + VIEW + 1, max(0, c - VIEW) : c + VIEW + 1] = True
            cheap = cheap | (pockets & near)
            if not cheap.any():  # nothing cheap nearby: a small loop instead
                self.kind, self.params = "loop", [3, 3, TURN_LEFT]
                return self._opt_loop(game, pid, at_home)
            self.target, self.phase = cheap, "go"
        if self.phase == "go":
            if not at_home and self.target[p.pos]:
                return self._go_home(game, pid)
            return self._checked(game, pid, self._route(game, pid, self.target, game.trail == pid))
        return self._go_home(game, pid)

    def _opt_attack(self, game, pid, at_home):
        p = game.players[pid]
        enemy_trail = (game.trail != EMPTY) & (game.trail != pid)
        if enemy_trail.any() and self.steps <= 40:
            cells = np.argwhere(enemy_trail)
            if int(np.abs(cells - np.array(p.pos)).sum(1).min()) <= VIEW:
                a = bfs_first_action(game, pid, enemy_trail, game.trail == pid, ORDER)
                if a is not None and game.is_safe(pid, a):
                    return a
        return self._step_inside(game, pid) if at_home else self._go_home(game, pid)

    def _opt_ambush(self, game, pid, at_home):
        # wait on my land while an enemy is close: one step per decision
        if not at_home:
            return self._go_home(game, pid)
        if self.can_pause and enemy_near(game, pid, 8):
            self.done = True
            return PAUSE
        return self._step_inside(game, pid)


# ---------------------------------------------------------------- env and actor
import gymnasium as gym  # noqa: E402


class OptionsEnv(gym.Wrapper):
    """SplixEnv where each action is an option (index into OPTIONS).

    ``step`` runs the option to its end (or the episode's) and returns the
    discounted sum of rewards; ``info["option_disc"]`` is gamma ** k for the
    k steps it took, the discount to bootstrap with.
    """

    def __init__(self, env, gamma: float = 0.995, safety: float = 6.0, lag_slack: float = 30.0):
        super().__init__(env)
        self.gamma = gamma
        self.safety = safety
        self.lag_slack = lag_slack
        self.action_space = gym.spaces.Discrete(N_OPTIONS)
        self._rng = np.random.default_rng(0)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._rng = np.random.default_rng(int(self.env.np_random.integers(2**31)))
        return obs, info

    def step(self, option):
        base = self.env.unwrapped
        run = OptionRunner(int(option), self._rng, lag=getattr(base, "lag_prob", 0.0),
                           can_pause=getattr(base, "allow_pause", False), safety=self.safety, lag_slack=self.lag_slack)
        total, disc, k = 0.0, 1.0, 0
        obs, term, trunc, info = None, False, False, {}
        base.lazy_obs = True  # full observation only once, when the plan ends
        while True:
            a = run.act(base.game, 0)
            if a is None:
                if k > 0:
                    break
                a = STRAIGHT if base.game.is_safe(0, STRAIGHT) else (safe_actions(base.game, 0) or [STRAIGHT])[0]
            obs, r, term, trunc, info = self.env.step(a)
            total += disc * r
            disc *= self.gamma
            k += 1
            if term or trunc:
                break
        base.lazy_obs = False
        obs = base._obs()
        info = dict(info)
        info.update(option_disc=disc, option_steps=k, option_aborted=run.aborted)
        return obs, total, term, trunc, info


class OptionsActor:
    """Plays an options agent tile by tile, so evaluate / watch / live / frozen
    opponents can use it like any other actor."""

    def __init__(self, agent, name: str = "options", deterministic: bool = True):
        self.agent = agent
        self.name = name
        self.deterministic = deterministic
        cfg = agent.options or {}
        self.safety = cfg.get("safety", 6.0)
        self.lag_slack = cfg.get("lag_slack", 30.0)
        self.run: OptionRunner | None = None
        self.rng = np.random.default_rng(0)
        self.chosen: list[int] = []

    def reset(self, env, seed: int) -> None:
        self.run = None
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs=None, pid: int = 0) -> int:
        from .obs import build_obs

        game = env.game
        if self.run is not None:
            a = self.run.act(game, pid)
            if a is not None:
                from .obs import update_memory

                update_memory(game, pid, self.agent.obs_config)
                return a
        o = build_obs(game, pid, self.agent.obs_config)
        option = int(self.agent.act(o, deterministic=self.deterministic))
        self.chosen.append(option)
        lag = float(getattr(game, "lag_estimate", 0.0))
        self.run = OptionRunner(option, self.rng, lag=lag, can_pause=getattr(env, "allow_pause", True),
                                safety=self.safety, lag_slack=self.lag_slack)
        a = self.run.act(game, pid)
        if a is None:
            a = STRAIGHT if game.is_safe(pid, STRAIGHT) else (safe_actions(game, pid) or [STRAIGHT])[0]
        return a


def make_options_env(gamma: float = 0.995, safety: float = 6.0, lag_slack: float = 30.0, **env_kwargs):
    """Picklable factory for vector envs."""
    from .env import SplixEnv

    return OptionsEnv(SplixEnv(**env_kwargs), gamma=gamma, safety=safety, lag_slack=lag_slack)
