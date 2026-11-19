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


def _enemy_heads(game: SplixGame, pid: int, exclude=None) -> np.ndarray:
    p = game.players[pid]
    heads = [q.pos for q in game.players
             if q.alive and q.pid != pid and getattr(q, "key", q.pid) != exclude
             and max(abs(q.pos[0] - p.pos[0]), abs(q.pos[1] - p.pos[1])) <= VIEW]
    return np.array(heads, dtype=np.int64).reshape(-1, 2)


MEMORY_TRUST = 30  # remembered enemies older than this many steps are ignored
MEMORY_SPEED = 0.5  # an unseen enemy is assumed to have closed in this many tiles per step since seen
# Loops are sized to fit the safety budget before they start (out + along <=
# BUDGET_SHARE * (nearest enemy reach - cushion)), so they finish as rectangles
# instead of being aborted halfway, which captured almost nothing.
BUDGET_SHARE = 0.8
IDLE_BUDGET = 12  # expand instead of waiting at home when a strip this big fits (None: only with nobody on screen)
LAG_CAP = 0.1  # the most lag the safety cushion accounts for (training and tests used 0 - 0.1)
# Unseen enemies are assumed to be at least this many tiles inside the edge of my view (lag, and the server's view is not exactly square).
UNSEEN_MARGIN = 5
# Don't start a loop the budget can't fit (it would be turned back at once).
NO_DOOMED_LOOPS = True
MIN_LOOP = 4
# Turn a loop's side leg toward my land so it lies along the edge.
HUG_EDGE = True
# If all my land is a hole inside one enemy's land, their next loop anywhere
# takes all of it (and I die with no land left): cut a strip out to the
# open first.
BREAK_OUT = False  # under the real capture rule it adds nothing (see SUMMARY)
# Turn one tile early when the tile after next is a wall or my own trail: a
# turn that lands a tile late (lag) then still lands safely. Most deaths in
# simulation were wall hits from last-tile turns.
LAG_GUARD = True
# Attack only races I win: I reach the victim's trail at least RACE_SLACK
# steps before it gets home and before it could reach my trail. While such a
# race is on, the victim doesn't count in the safety rule; everyone else does.
SAFE_ATTACK = True
RACE_SLACK = 2
# From my own land, take any winnable race within KILL_REACH, whatever plan I'm on.
KILL_REFLEX = False  # more kills but almost twice the deaths in testing
KILL_REACH = 12
# Expand with strips: when a loop is due, lay a d x L rectangle flush along
# the straightest stretch of my edge nearby instead (staying within d tiles of
# my land the whole way), picking the most land per step of the trip.
STRIPS = True
# Only see enemy trails on my land that are on screen, as the real client does.
VIEW_DEFEND = True
# My tile count dropping with no enemy trail on my land in view means land is
# being taken where I can't see: walk over my land to where it likely is --
# border other land has grown toward lately (minimap), else border other land
# touches, else my land out of view. At most one patrol per ALARM_COOLDOWN steps.
LAND_ALARM = False  # off: walking to a far border arrives after the raid (see SUMMARY)
ALARM_TILES = 5   # tiles lost within ALARM_WINDOW steps
ALARM_WINDOW = 20
ALARM_COOLDOWN = 60  # steps between patrols
PATROL_STEPS = 80  # a patrol keeps going this long to reach the threatened border (big territories)
GROWTH_AGE = 30  # "approaching": other land that appeared within this many steps
MAX_STRIP = 24
MAX_DEPTH = 8
STRIP_SKIP_HOLES = True  # no strips into holes in my land (enclosed enemy land is for the cut)
STRIP_WINDOW = 25  # strips may start this far away (50 tested: no better, more deaths)
STRIP_MAX_WALK = 50
STRIP_WALK_WEIGHT = 1.0  # how much a step walking over my land to the strip's start counts against it
# Cut a line through enemy land enclosed by mine, away from its owner's head:
# the server gives me every piece the owner isn't standing in. Only when the
# owner can't reach the line before I'm across it.
CUT_REFLEX = True
# Close dents: a straight trail across the mouth of a dent in my land, from my
# land to my land, takes the whole dent. Chosen over a strip when it gains
# more land per step.
CLOSE_DENTS = True
MAX_CLOSE = 30  # longest line across a dent
# The line stays on screen (where trails and heads are visible), but what it
# would capture is worked out over CLOSE_WINDOW tiles each way, from my
# remembered land and the minimap, so a dent bigger than the screen counts whole.
CLOSE_WINDOW = 60
CLOSE_BUDGET_FACTOR = 1.0  # a line may be this many safety budgets long
CUT_SLACK = 1  # the owner must need more than (tiles left to cross + this) steps to reach my line
BREAK_OUT_THRESHOLD = 2.0  # escape cushion while breaking out (normal: safety + lag slack)


def _enemy_reach(game: SplixGame, pid: int, exposed: np.ndarray, exclude=None) -> int | None:
    """Fewest steps any enemy could need to touch ``exposed`` cells.

    Visible heads count at their distance. Enemies seen recently but now out of
    sight count too: last seen ``age`` steps ago at P, they could be ``age``
    tiles closer by now (the observation memory, see obs.memory_planes).
    """
    best = None
    heads = _enemy_heads(game, pid, exclude)
    if len(heads):
        best = int(np.abs(exposed[:, None, :] - heads[None, :, :]).sum(2).min())
    mem = (getattr(game, "obs_memory", None) or {}).get(pid)
    if mem:
        clock = int(getattr(game, "clock", game.t))
        # An enemy I can't see now is outside my view, so it needs at least
        # VIEW + 1 - (how far the cell is from my head) steps to reach that cell.
        head = np.array(game.players[pid].pos)
        unseen = np.maximum(0, VIEW + 1 - UNSEEN_MARGIN - np.abs(exposed - head).max(1))
        for key, (pos, t) in mem.get("seen", {}).items():
            if exclude is not None and key == exclude:
                continue
            age = clock - t
            if 0 < age <= MEMORY_TRUST:
                guess = np.abs(exposed - np.array(pos)).sum(1) - MEMORY_SPEED * age
                d = max(0, int(np.maximum(guess, unseen).min()))
                best = d if best is None else min(best, d)
    return best


def enclosure(game: SplixGame, pid: int) -> tuple[int, np.ndarray] | None:
    """(enemy pid, cells outside its closure) if all my land lies inside one
    enemy's land (a hole in it), else None. Only land I know about counts."""
    from scipy.ndimage import binary_fill_holes

    own = game.owner == pid
    if not own.any():
        return None
    cells = np.argwhere(own)
    r, c = cells[0]
    for q in game.players:
        if not q.alive or q.pid == pid:
            continue
        theirs = (game.owner == q.pid) | (game.trail == q.pid)
        # quick test first: is my first tile walled in by their land on all four sides?
        if not (theirs[:r, c].any() and theirs[r + 1:, c].any() and theirs[r, :c].any() and theirs[r, c + 1:].any()):
            continue
        closed = binary_fill_holes(theirs)
        if closed[own].all():
            return q.pid, ~closed
    return None


def escape_margin(game: SplixGame, pid: int, exclude=None) -> float:
    """(fewest steps an enemy -- visible, or seen recently -- needs to reach my
    trail or head) - (my distance home). ``exclude``: an enemy I'm about to
    kill, whose race I've already checked."""
    p = game.players[pid]
    exposed = np.array([*p.trail, p.pos], dtype=np.int64).reshape(-1, 2)
    reach = _enemy_reach(game, pid, exposed, exclude)
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
        # lag above LAG_CAP was never trained or tested: don't let a pessimistic
        # estimate shrink every plan to a few tiles
        self.threshold = safety + lag_slack * min(lag, LAG_CAP)
        self.steps = 0
        self.phase: str | None = None
        self.left_home = False
        self.done = False
        self.aborted = False  # the safety rule fired
        self.face: int | None = None  # world direction an adaptive loop heads out in
        self.defend = defend
        self.idle_guard = True  # never stand around on my land with nobody in view (helps against human-like bots, see SUMMARY)
        self.idled_off = False
        self.defended = 0  # steps spent on the defend reflex
        self.broke_out = False  # switched to breaking out of an enclosure
        self.base_threshold = self.threshold
        self.prey = None  # (key, trail cells) of an enemy whose race I'm winning
        self.cutting = None  # (start, h, gain, line cells, owner key) of a cut through enemy land
        self.alarmed = False  # went to look for land being taken out of view
        self.target = None
        self.patrol_kind = None

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

    def _late_turn_fatal(self, game, pid, heading=None, pos=None):
        """Would going straight one more tile leave no room for a late turn?"""
        p = game.players[pid]
        h = p.heading if heading is None else heading
        r, c = p.pos if pos is None else pos
        r2, c2 = r + 2 * int(DIRS[h, 0]), c + 2 * int(DIRS[h, 1])
        if not (0 <= r2 < game.size and 0 <= c2 < game.size):
            return True
        return bool(game.trail[r2, c2] == pid)

    def _after(self, game, pid, a):
        p = game.players[pid]
        h = p.heading if a == STRAIGHT else (p.heading + (3 if a == TURN_LEFT else 1)) % 4
        return h, (p.pos[0] + int(DIRS[h, 0]), p.pos[1] + int(DIRS[h, 1]))

    def _lag_safe(self, game, pid, a):
        """Backstop for every move: don't end up facing a wall or my own trail
        from right next to it, where the next turn landing late is fatal."""
        if not LAG_GUARD or a not in (STRAIGHT, TURN_LEFT, TURN_RIGHT):
            return a
        h, nxt = self._after(game, pid, a)
        if not self._late_turn_fatal(game, pid, h, nxt):
            return a
        own = np.argwhere(game.owner == pid)
        best, best_d = a, None
        for t in (STRAIGHT, TURN_LEFT, TURN_RIGHT):
            if t == a or not game.is_safe(pid, t):
                continue
            h, nxt = self._after(game, pid, t)
            if self._late_turn_fatal(game, pid, h, nxt):
                continue
            d = int(np.abs(own - np.array(nxt)).sum(1).min()) if len(own) else 0
            if best_d is None or d < best_d:
                best, best_d = t, d
        return best

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
        a = self._act(game, pid)
        return a if a is None else self._lag_safe(game, pid, a)

    def _act(self, game: SplixGame, pid: int) -> int | None:
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
        if SAFE_ATTACK and self.kind == "attack":
            self.prey = self._race(game, pid)
        prey_key = self.prey[0] if self.prey is not None else None
        if self.kind == "cut" and self.cutting is not None:
            prey_key = self.cutting[4] if self._cut_race_ok(game, pid) else None
        if not at_home and (self.steps > MAX_OPTION_STEPS or
                            (p.trail and escape_margin(game, pid, prey_key) <= self.threshold)):
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
        if LAND_ALARM and at_home and self.kind not in ("attack", "break_out", "patrol") and self.phase is None \
                and self._land_alarm(game, pid):
            self.kind, self.alarmed = "patrol", True
        if KILL_REFLEX and SAFE_ATTACK and at_home and self.kind not in ("attack", "break_out"):
            prey = self._race(game, pid)
            if prey is not None and int(np.abs(np.argwhere(prey[1]) - np.array(p.pos)).sum(1).min()) <= KILL_REACH:
                self.kind, self.prey, self.phase, self.steps = "attack", prey, None, 1
                self.killer = True
        if CUT_REFLEX and at_home and self.phase is None and self.kind not in ("attack", "cut", "break_out"):
            plan = self._plan_cut(game, pid)
            if plan is not None:
                # worth it only if it beats the strip I'd lay instead, per step of the trip
                start, h, gain, line, key = plan
                walk = int(abs(start[0] - p.pos[0]) + abs(start[1] - p.pos[1]))
                strip = self._plan_strip(game, pid) if STRIPS else None
                strip_rate = 0.0
                if strip is not None:
                    sh, sS, st, sL, sd = strip
                    strip_rate = sL * sd / (sL + 2 * sd + abs(sS[0] - p.pos[0]) + abs(sS[1] - p.pos[1]))
                if gain / (len(line) + 1 + walk) >= strip_rate:
                    self.kind, self.cutting = "cut", plan
                else:
                    self.cutting = None
        if BREAK_OUT and at_home and self.phase is None and self.kind != "break_out":
            enc = enclosure(game, pid)
            if enc is not None:
                self.kind, self.target, self.broke_out = "break_out", enc[1], True
                # all my land is lost at the encloser's next loop anyway: accept
                # more risk than usual to get out
                self.threshold = min(self.base_threshold, BREAK_OUT_THRESHOLD)
        if STRIPS and at_home and self.phase is None and self.kind in ("loop", "open_loop", "safe_loop"):
            plan = self._plan_strip(game, pid)
            if plan is not None:
                self.kind, self.strip = "strip", plan
            if CLOSE_DENTS:
                close = self._plan_close(game, pid)
                if close is not None:
                    start, h, gain, line, _ = close
                    walk = int(abs(start[0] - p.pos[0]) + abs(start[1] - p.pos[1]))
                    rate = gain / (len(line) + 1 + walk)
                    strip_rate = 0.0
                    if plan is not None:
                        sh, sS, st, sL, sd = plan
                        strip_rate = sL * sd / (sL + 2 * sd + abs(sS[0] - p.pos[0]) + abs(sS[1] - p.pos[1]))
                    if rate > strip_rate:
                        self.kind, self.cutting = "cut", close
        if at_home and self.idle_guard and self.kind in ("home", "ambush", "attack") and self.phase is None \
                and self._nothing_to_wait_for(game, pid):
            # nothing to wait for or attack: expand instead of idling
            self.kind = "open_loop"
            self.idled_off = True
        return getattr(self, "_opt_" + self.kind)(game, pid, at_home)

    def _nothing_to_wait_for(self, game, pid) -> bool:
        """Waiting only pays when an enemy could come within reach soon: with
        room for a real strip (budget >= IDLE_BUDGET) and no enemy trail in
        view to go for, expand instead. IDLE_BUDGET = None: wait whenever
        anyone is on screen (the old rule)."""
        if IDLE_BUDGET is None:
            return not self._threat_near(game, pid)
        enemy_trail = (game.trail != EMPTY) & (game.trail != pid)
        if enemy_trail.any():
            p = game.players[pid]
            if int(np.abs(np.argwhere(enemy_trail) - np.array(p.pos)).sum(1).min()) <= VIEW:
                return False
        return self._budget(game, pid) >= IDLE_BUDGET

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
        p = game.players[pid]
        if VIEW_DEFEND and invader.any():
            r, c = p.pos
            sight = np.zeros_like(invader)
            sight[max(0, r - VIEW) : r + VIEW + 1, max(0, c - VIEW) : c + VIEW + 1] = True
            invader &= sight
        if not invader.any():
            return None
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
            if NO_DOOMED_LOOPS and self._budget(game, pid) < MIN_LOOP:
                # the safety rule would turn me back after a step or two, leaving a
                # 1-tile bump: stay in and decide again
                return self._step_inside(game, pid)
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
            if self.left > 0 and not (LAG_GUARD and self._late_turn_fatal(game, pid)):
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
            self.phase, self.left = "side", side - 1
            if HUG_EDGE:
                turn = self._hugging_turn(game, pid, side, turn)
            return self._checked(game, pid, turn)
        if self.phase == "side":
            if self.left > 0 and not (LAG_GUARD and self._late_turn_fatal(game, pid)):
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
        return self._go_home(game, pid)

    def _budget(self, game, pid):
        p = game.players[pid]
        reach = _enemy_reach(game, pid, np.array([p.pos], dtype=np.int64))
        return 99.0 if reach is None else BUDGET_SHARE * (reach - self.threshold)

    def _hugging_turn(self, game, pid, side, turn):
        """Turn the side leg toward my land, so the loop lies flush along the
        edge instead of jutting out past a corner: of the two turns, the one
        whose end is closer to home (the default on a tie)."""
        p = game.players[pid]
        own = np.argwhere(game.owner == pid)
        if not len(own):
            return turn
        best, best_d = turn, None
        for t in (turn, TURN_LEFT if turn == TURN_RIGHT else TURN_RIGHT):
            h = (p.heading + (3 if t == TURN_LEFT else 1)) % 4
            end = np.array(p.pos) + side * DIRS[h]
            if not (0 <= end[0] < game.size and 0 <= end[1] < game.size) or not game.is_safe(pid, t):
                continue
            d = int(np.abs(own - end).sum(1).min())
            if best_d is None or d < best_d:
                best, best_d = t, d
        return best

    def _fit_budget(self, game, pid, out, side):
        """Shrink a loop (keeping its shape) so it fits the time before an enemy could arrive."""
        p = game.players[pid]
        reach = _enemy_reach(game, pid, np.array([p.pos], dtype=np.int64))
        if reach is None:
            return out, side
        budget = BUDGET_SHARE * (reach - self.threshold)
        if out + side <= budget:
            return out, side
        # Same perimeter, as close to square as fits: a square captures the most
        # land per tile of exposed trail; thin strips capture almost nothing.
        b = int(max(0.0, budget))
        side2 = min(side, max(2, b // 2))
        return min(out, max(2, b - side2)), side2

    def _opt_break_out(self, game, pid, at_home):
        """Cut a strip through the enclosing land to the open, then go home:
        the capture joins my land to the outside, so it's no longer a hole."""
        p = game.players[pid]
        if at_home and self.phase is None:
            if self._budget(game, pid) < 1:
                return self._step_inside(game, pid)  # the encloser is right here: wait for a gap
            self.phase = "go"
        if self.phase == "go":
            if self.target[p.pos]:
                return self._go_home(game, pid)
            return self._checked(game, pid, self._route(game, pid, self.target, game.trail == pid))
        return self._go_home(game, pid)

    def _plan_strip(self, game, pid):
        """(h, start S, turn, length L, depth d) for the best strip, or None.

        Every straight stretch of my edge in view is a candidate. Score: new
        land per step of the trip (walk to the start + out + along + back),
        times how compact my land is afterwards (area / bounding box, squared),
        so strips that fill notches beat ones that stretch a peninsula."""
        p = game.players[pid]
        n = game.size
        own = game.owner == pid
        budget = self._budget(game, pid)
        if budget < MIN_LOOP:
            return None
        d_cap = int(np.clip(budget // 2, 2, MAX_DEPTH))
        me = np.array(p.pos)
        R = max(VIEW, STRIP_WINDOW)  # edges this far away count too (walking there is on my land)
        r0, r1 = max(0, me[0] - R), min(n, me[0] + R + 1)
        c0, c1 = max(0, me[1] - R), min(n, me[1] + R + 1)
        local = own[r0:r1, c0:c1]
        if not local.any():
            return None
        lc = np.argwhere(local) + np.array([r0, c0])
        area0 = len(lc)
        inside = None
        if STRIP_SKIP_HOLES:
            # holes in my land (often an enemy sitting inside it): don't nibble at
            # them with strips next to their owner; grow outward, and leave
            # enclosed enemy land to the cut
            from scipy.ndimage import binary_fill_holes
            inside = binary_fill_holes(own) & ~own
        lo0, hi0 = lc.min(0), lc.max(0)
        rr, cc = np.mgrid[0:n, 0:n]
        window = np.zeros_like(own)
        window[r0:r1, c0:c1] = True
        busy = game.trail != EMPTY
        for q in game.players:
            if q.alive and q.pid != pid:
                busy[q.pos] = True
        best, best_score = None, 0.0
        for h in range(4):
            dr, dc = int(DIRS[h, 0]), int(DIRS[h, 1])
            nb_in = (rr + dr >= 0) & (rr + dr < n) & (cc + dc >= 0) & (cc + dc < n)
            nb_own = np.roll(own, (-dr, -dc), axis=(0, 1)) & nb_in  # own[r + dr, c + dc]
            frontier = own & nb_in & ~nb_own & window
            if not frontier.any():
                continue
            axis = 1 if dr != 0 else 0  # position along the edge: column for vertical h, row otherwise
            cells = np.argwhere(frontier)
            for fixed in np.unique(cells[:, 1 - axis]):
                along = np.sort(cells[cells[:, 1 - axis] == fixed][:, axis])
                runs, start = [], along[0]
                for a, b in zip(along[:-1], along[1:]):
                    if b != a + 1:
                        runs.append((start, a)); start = b
                runs.append((start, along[-1]))
                for lo, hi in runs:
                    L = int(hi - lo + 1)
                    if L < 3:
                        continue
                    near_lo = abs(int(lo) - int(me[axis])) <= abs(int(hi) - int(me[axis]))
                    cap = int(np.clip(budget - 2, 3, MAX_STRIP))
                    if L > cap:
                        lo, hi = (lo, lo + cap - 1) if near_lo else (hi - cap + 1, hi)
                        L = cap
                    s_along, step = (int(lo), 1) if near_lo else (int(hi), -1)
                    S = (int(fixed), s_along) if axis == 1 else (s_along, int(fixed))
                    pdir = (0, step) if axis == 1 else (step, 0)
                    d = 0
                    for k in range(1, d_cap + 1):
                        band = [(S[0] + k * dr + j * pdir[0], S[1] + k * dc + j * pdir[1]) for j in range(L)]
                        # one tile off the wall, so the far turn can land late
                        if not all(1 <= x < n - 1 and 1 <= y < n - 1 for x, y in band) or any(busy[x, y] for x, y in band):
                            break
                        d = k
                    if d < 2:
                        continue
                    if inside is not None and inside[S[0] + dr, S[1] + dc]:
                        continue  # this edge faces a hole in my land
                    far = (S[0] + d * dr + (L - 1) * pdir[0], S[1] + d * dc + (L - 1) * pdir[1])
                    lo1 = np.minimum(np.minimum(lo0, S), far)
                    hi1 = np.maximum(np.maximum(hi0, S), far)
                    corners = np.array([S, far])
                    band_lo, band_hi = corners.min(0), corners.max(0)
                    gained = L * d - int(np.count_nonzero(own[band_lo[0] : band_hi[0] + 1, band_lo[1] : band_hi[1] + 1]))
                    if gained < 4:
                        continue
                    fill = (area0 + gained) / float(np.prod(hi1 - lo1 + 1))
                    walk = int(np.abs(np.array(S) - me).sum())
                    if walk > STRIP_MAX_WALK:
                        continue
                    score = gained / (L + 2 * d + STRIP_WALK_WEIGHT * walk) * fill * fill
                    if score > best_score:
                        ph = [i for i in range(4) if (int(DIRS[i, 0]), int(DIRS[i, 1])) == pdir][0]
                        turn = TURN_RIGHT if ph == (h + 1) % 4 else TURN_LEFT
                        best, best_score = (h, S, turn, L, d), score
        return best

    def _opt_strip(self, game, pid, at_home):
        h, S, turn, L, d = self.strip
        p = game.players[pid]
        if at_home:
            if self.phase not in (None, "to_start"):
                return self._go_home(game, pid)
            self.phase = "to_start"
            T = (S[0] + int(DIRS[h, 0]), S[1] + int(DIRS[h, 1]))
            goal = np.zeros_like(game.owner, dtype=bool)
            goal[T] = True
            blocked = game.owner != pid
            for side in ((h + 1) % 4, (h + 3) % 4):  # step out from S itself, facing h
                x, y = T[0] + int(DIRS[side, 0]), T[1] + int(DIRS[side, 1])
                if 0 <= x < game.size and 0 <= y < game.size:
                    blocked[x, y] = True
            blocked[T] = False
            blocked |= game.trail == pid
            if tuple(p.pos) == tuple(S) and self._budget(game, pid) < max(MIN_LOOP, d + 2):
                return self._step_inside(game, pid)  # at the start, but it's no longer safe here: decide again
            a = bfs_first_action(game, pid, goal, blocked, ORDER)
            if a is None or self.steps > STRIP_MAX_WALK + 5:  # can't get there: an ordinary loop instead
                self.kind = "open_loop"
                return self._opt_open_loop(game, pid, at_home)
            return self._checked(game, pid, a)
        if self.phase == "to_start":  # first tile out
            self.phase, self.left = "out", d - 1
        if self.phase == "out":
            if self.left > 0 and not (LAG_GUARD and self._late_turn_fatal(game, pid)):
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
            self.phase, self.left = "side", L - 2  # the corner tile and the turn's own tile are the first two
            return self._checked(game, pid, turn)
        if self.phase == "side":
            if self.left > 0 and not (LAG_GUARD and self._late_turn_fatal(game, pid)):
                self.left -= 1
                return self._checked(game, pid, STRAIGHT)
        return self._go_home(game, pid)

    def _land_alarm(self, game, pid) -> bool:
        """Has my tile count dropped lately with nothing in view to explain it?"""
        count = getattr(game, "land_count", None)
        if count is None:
            count = int(np.count_nonzero(game.owner == pid))
        store = getattr(game, "obs_memory", None)
        if store is None:
            store = game.obs_memory = {}
        mem = store.setdefault(pid, {"seen": {}, "maps": []})
        hist = mem.setdefault("land", [])
        clock = int(getattr(game, "clock", game.t))
        if not hist or hist[-1][0] != clock:
            hist.append((clock, count))
        del hist[: max(0, len(hist) - ALARM_WINDOW)]
        if clock % 10 == 0:
            self._other_land(game, pid)  # keep the growth snapshots current
        if len(hist) < 2 or max(c for _, c in hist) - count < ALARM_TILES:
            return False
        if self._invasion(game, pid, True) is not None or len(_enemy_heads(game, pid)):
            return False  # the taker is in view: the defend reflex and the safety rule have it
        if clock - mem.get("patrol_t", -10**9) < ALARM_COOLDOWN:
            return False
        mem["patrol_t"] = clock
        return True

    def _other_land(self, game, pid):
        """Land that isn't mine, as far as I know it: what I've seen, plus the
        minimap (which shows land but not whose). Snapshots kept for growth."""
        own = game.owner == pid
        land = game.owner != EMPTY
        hint = getattr(game, "land_hint", None)
        if hint is not None and hint.shape == land.shape:
            land = land | hint
        other = land & ~own
        mem = game.obs_memory.setdefault(pid, {"seen": {}, "maps": []})
        snaps = mem.setdefault("other", [])
        clock = int(getattr(game, "clock", game.t))
        if not snaps or clock - snaps[-1][0] >= 10:
            snaps.append((clock, other.copy()))
        while len(snaps) > 1 and clock - snaps[1][0] >= GROWTH_AGE:
            snaps.pop(0)
        old = snaps[0][1] if snaps[0][1].shape == other.shape else other
        return other, other & ~old

    def _opt_patrol(self, game, pid, at_home):
        """Walk over my land to where it's likely being taken, out of view:
        border where other land has grown toward me lately, else border that
        other land touches, else just my land out of view."""
        from scipy.ndimage import binary_dilation

        p = game.players[pid]
        if not at_home or self.steps > PATROL_STEPS:
            return self._go_home(game, pid) if not at_home else self._step_inside(game, pid)
        own = game.owner == pid
        r, c = p.pos
        out_of_view = np.ones_like(own)
        out_of_view[max(0, r - VIEW + 3) : r + VIEW - 2, max(0, c - VIEW + 3) : c + VIEW - 2] = False
        if self.target is None:
            other, grown = self._other_land(game, pid)
            far = own & out_of_view
            approaching = far & binary_dilation(grown, iterations=3)
            touching = far & binary_dilation(other, iterations=1)
            self.target = next((m for m in (approaching, touching, far) if m.any()), None)
            self.patrol_kind = ("approaching" if approaching.any() else "touching" if touching.any() else "inside") \
                if self.target is not None else None
        if self.target is None or self.target[p.pos]:
            return self._step_inside(game, pid)
        a = bfs_first_action(game, pid, self.target, ~own | (game.trail == pid), ORDER)
        if a is None or not game.is_safe(pid, a):
            return self._step_inside(game, pid)
        return a

    def _plan_close(self, game, pid):
        """Best straight line across a dent in my land, as (start, h, gain, line, None)."""
        from scipy.ndimage import binary_fill_holes, label

        p = game.players[pid]
        n = game.size
        budget = self._budget(game, pid)
        if budget < MIN_LOOP:
            return None
        me = np.array(p.pos)
        # lines: within view. Captures: over the wider window.
        R = max(VIEW, CLOSE_WINDOW)
        r0, r1 = max(0, me[0] - R), min(n, me[0] + R + 1)
        c0, c1 = max(0, me[1] - R), min(n, me[1] + R + 1)
        own = (game.owner == pid)[r0:r1, c0:c1]
        if not own.any():
            return None
        busy = (game.trail != EMPTY)[r0:r1, c0:c1].copy()
        heads = []
        for q in game.players:
            if q.alive and q.pid != pid:
                if r0 <= q.pos[0] < r1 and c0 <= q.pos[1] < c1:
                    busy[q.pos[0] - r0, q.pos[1] - c0] = True
                for cell in [q.pos] + ([q.trail[0]] if q.trail else []):
                    heads.append((cell[0] - r0, cell[1] - c0))
        vr0, vr1 = max(0, me[0] - VIEW) - r0, min(n, me[0] + VIEW + 1) - r0  # the view, in window coordinates
        vc0, vc1 = max(0, me[1] - VIEW) - c0, min(n, me[1] + VIEW + 1) - c0
        # the whole line is exposed: it must fit in about twice the budget
        # (halfway across, either end is equally close)
        max_len = int(min(MAX_CLOSE, CLOSE_BUDGET_FACTOR * budget))
        H, W = own.shape
        best, best_rate = None, 0.0
        def on_screen(cell):
            return vr0 <= cell[0] < vr1 and vc0 <= cell[1] < vc1

        for axis in (0, 1):
            # lines start on screen (the screen moves along as I cross); their far end can be off it
            for k in range(vr0, vr1) if axis == 0 else range(vc0, vc1):
                row = own[k, :] if axis == 0 else own[:, k]
                idx = np.flatnonzero(row)
                for a, b in zip(idx[:-1], idx[1:]):
                    gap = int(b - a - 1)
                    if gap < 2 or gap > max_len:
                        continue
                    end_a, end_b = ((k, int(a)), (k, int(b))) if axis == 0 else ((int(a), k), (int(b), k))
                    if not (on_screen(end_a) or on_screen(end_b)):
                        continue
                    cells = [(k, j) if axis == 0 else (j, k) for j in range(a + 1, b)]
                    if any(busy[c] for c in cells):
                        continue
                    mask = own.copy()
                    for c in cells:
                        mask[c] = True
                    enclosed = binary_fill_holes(mask) & ~mask
                    if enclosed.any() and heads:
                        labels, _ = label(enclosed)
                        prot = {labels[x + dx, y + dy] for x, y in heads for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                if 0 <= x + dx < H and 0 <= y + dy < W}
                        prot.discard(0)
                        if prot:
                            enclosed &= ~np.isin(labels, list(prot))
                    gain = int(np.count_nonzero(enclosed)) + gap
                    if gain < 2 * gap + 4:  # a line that takes little more than itself isn't a dent
                        continue
                    ends = [e for e in (end_a, end_b) if on_screen(e)] or [end_a, end_b]
                    ends = [(x + r0, y + c0) for x, y in ends]
                    start = min(ends, key=lambda e: abs(e[0] - me[0]) + abs(e[1] - me[1]))
                    ends = [(x + r0, y + c0) for x, y in (end_a, end_b)]
                    other = ends[1] if start == ends[0] else ends[0]
                    walk = int(abs(start[0] - me[0]) + abs(start[1] - me[1]))
                    rate = gain / (gap + 1 + walk)
                    if rate > best_rate:
                        h = [i for i in range(4) if (int(np.sign(other[0] - start[0])), int(np.sign(other[1] - start[1])))
                             == (int(DIRS[i, 0]), int(DIRS[i, 1]))][0]
                        line = [(x + r0, y + c0) for x, y in cells]
                        if start != ends[0]:
                            line = line[::-1]
                        best, best_rate = (start, h, gain, line, None), rate
        return best

    def _plan_cut(self, game, pid):
        from .opponents.bots import plan_cut

        def winnable(line, q):  # the owner needs longer to reach the line than I need to cross it
            d = int(np.abs(np.array(line) - np.array(q.pos)).sum(1).min())
            return d > len(line) + 1 + CUT_SLACK

        plan = plan_cut(game, pid, reach=VIEW, ok=winnable)
        if plan is None:
            return None
        start, h, gain, line, owner_pid = plan
        q = game.players[owner_pid]
        self.cutting = (start, h, gain, line, getattr(q, "key", q.pid))
        if not self._cut_race_ok(game, pid, at_start=True):
            self.cutting = None
            return None
        return self.cutting

    def _cut_race_ok(self, game, pid, at_start=False):
        """Can the patch's owner reach my line before I'm across it?"""
        start, h, gain, line, key = self.cutting
        q = next((q for q in game.players if q.alive and getattr(q, "key", q.pid) == key), None)
        if q is None:
            return True
        p = game.players[pid]
        cells = np.array(line if at_start else [*p.trail, p.pos, *line[len(p.trail):]], dtype=np.int64).reshape(-1, 2)
        # the race starts when I step off my land (the walk to the start is on it,
        # and this is re-checked every step of the walk)
        left = len(line) + 1 - (0 if at_start else len(p.trail))
        d = int(np.abs(cells - np.array(q.pos)).sum(1).min())
        return d > left + CUT_SLACK

    def _opt_cut(self, game, pid, at_home):
        start, h, gain, line, key = self.cutting
        p = game.players[pid]
        if at_home:
            if self.phase not in (None, "to_start"):
                return self._go_home(game, pid)
            if not self._cut_race_ok(game, pid, at_start=True) or self.steps > 60:
                return self._step_inside(game, pid)  # the owner came close: not now
            if key is None and self._budget(game, pid) < (len(line) + 1) / 2:
                return self._step_inside(game, pid)  # closing a dent: someone came closer while I walked
            self.phase = "to_start"
            T = (start[0] + int(DIRS[h, 0]), start[1] + int(DIRS[h, 1]))
            goal = np.zeros_like(game.owner, dtype=bool)
            goal[T] = True
            blocked = game.owner != pid
            for side in ((h + 1) % 4, (h + 3) % 4):  # step out from the start cell itself, facing across
                x, y = T[0] + int(DIRS[side, 0]), T[1] + int(DIRS[side, 1])
                if 0 <= x < game.size and 0 <= y < game.size:
                    blocked[x, y] = True
            blocked[T] = False
            blocked |= game.trail == pid
            a = bfs_first_action(game, pid, goal, blocked, ORDER)
            if a is None:
                return self._step_inside(game, pid)
            return self._checked(game, pid, a)
        self.phase = "across"
        if p.heading == h and game.is_safe(pid, STRAIGHT):
            return STRAIGHT
        return self._go_home(game, pid)

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

    def _race(self, game, pid):
        """The closest enemy trail I can cut first, as (key, cells), or None."""
        p = game.players[pid]
        me = np.array(p.pos)
        mine = np.array([*p.trail, p.pos], dtype=np.int64).reshape(-1, 2)
        best, best_d = None, None
        for q in game.players:
            if not q.alive or q.pid == pid or max(abs(q.pos[0] - p.pos[0]), abs(q.pos[1] - p.pos[1])) > VIEW:
                continue
            theirs = np.argwhere(game.trail == q.pid)
            if not len(theirs):
                continue
            my_d = int(np.abs(theirs - me).sum(1).min())
            land = np.argwhere(game.owner == q.pid)
            home = int(np.abs(land - np.array(q.pos)).sum(1).min()) if len(land) else 99
            hit = int(np.abs(mine - np.array(q.pos)).sum(1).min())
            if my_d > VIEW or my_d + RACE_SLACK > min(home, hit):
                continue
            if best_d is None or my_d < best_d:
                best, best_d = (getattr(q, "key", q.pid), game.trail == q.pid), my_d
        return best

    def _opt_attack(self, game, pid, at_home):
        p = game.players[pid]
        if SAFE_ATTACK:
            if self.prey is not None and self.steps <= 40:
                a = bfs_first_action(game, pid, self.prey[1], game.trail == pid, ORDER)
                if a is not None and game.is_safe(pid, a):
                    return a
            return self._step_inside(game, pid) if at_home else self._go_home(game, pid)
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
        self.explain = False  # print one line per plan: what, how big, and why

    def _explain(self, game, pid, option):
        r = self.run
        p = game.players[pid]
        reach = _enemy_reach(game, pid, np.array([p.pos], dtype=np.int64))
        size = ""
        if r.kind == "strip":
            size = f" {r.strip[3]} long x {r.strip[4]} deep"
        elif r.kind in ("loop", "open_loop", "safe_loop") and r.params:
            size = f" {r.params[0]} out x {r.params[1]} along"
        elif r.kind == "cut":
            size = f" across {len(r.cutting[3])}, takes {r.cutting[2]}"
        picked = OPTION_NAMES[option] if option < len(OPTION_NAMES) else option
        print(f"[plan] {r.kind}{size} (picked {picked}) | budget {r._budget(game, pid):.0f} tiles, "
              f"cushion {r.threshold:.1f} (lag {float(getattr(game, 'lag_estimate', 0.0)):.2f}), "
              f"nearest known enemy {'none' if reach is None else reach} tiles, "
              f"on screen {len(_enemy_heads(game, pid))}", flush=True)

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
        if self.explain:
            self._explain(game, pid, option)
        if a is None:
            a = STRAIGHT if game.is_safe(pid, STRAIGHT) else (safe_actions(game, pid) or [STRAIGHT])[0]
        return a


def make_options_env(gamma: float = 0.995, safety: float = 6.0, lag_slack: float = 30.0, **env_kwargs):
    """Picklable factory for vector envs."""
    from .env import SplixEnv

    return OptionsEnv(SplixEnv(**env_kwargs), gamma=gamma, safety=safety, lag_slack=lag_slack)
