import numpy as np
import pytest

from splixrl.game import (
    DOWN,
    EMPTY,
    LEFT,
    PAUSE,
    RIGHT,
    STRAIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    UP,
    Player,
    SplixGame,
    turn,
)


def make_game(size=12, n=2):
    g = SplixGame(size=size, n_players=n, spawn_radius=1)
    g.reset(np.random.default_rng(0))
    # replace random spawns with a known layout
    g.owner.fill(EMPTY)
    g.trail.fill(EMPTY)
    g.players = []
    return g


def place(g, pid, r, c, heading, square=1):
    g.owner[r - square : r + square + 1, c - square : c + square + 1] = pid
    g.players.append(Player(pid=pid, pos=(r, c), heading=heading))


def test_turn():
    assert turn(UP, TURN_LEFT) == LEFT
    assert turn(UP, TURN_RIGHT) == RIGHT
    assert turn(LEFT, TURN_LEFT) == DOWN
    assert turn(DOWN, STRAIGHT) == DOWN


def test_spawn_is_square_and_separate():
    g = SplixGame(size=20, n_players=3, spawn_radius=2)
    g.reset(np.random.default_rng(1))
    for p in g.players:
        assert g.territory(p.pid) == 25
        r, c = p.pos
        assert (g.owner[r - 2 : r + 3, c - 2 : c + 3] == p.pid).all()


def test_spawn_too_small_board_raises():
    with pytest.raises(ValueError):
        SplixGame(size=6, n_players=4, spawn_radius=2).reset(np.random.default_rng(0))


def test_wall_kills():
    g = make_game()
    place(g, 0, 1, 5, UP)  # head at row 1, one step from the top edge
    g.step([STRAIGHT])  # row 0, still on the board
    assert g.players[0].alive
    g.step([STRAIGHT])
    assert not g.players[0].alive and g.players[0].death_cause == "wall"
    assert g.territory(0) == 0


def test_trail_grows_outside_and_capture_fills_loop():
    g = make_game(size=12, n=1)
    place(g, 0, 5, 5, UP)  # owns rows 4-6, cols 4-6
    # out of the top: (3,5), (2,5); right: (2,6),(2,7); down: (3,7),(4,7); left back into (4,6)
    moves = [STRAIGHT, STRAIGHT, STRAIGHT, TURN_RIGHT, STRAIGHT, TURN_RIGHT, STRAIGHT, TURN_RIGHT]
    for a in moves[:-1]:
        g.step([a])
    p = g.players[0]
    assert p.alive
    assert len(p.trail) == 6 and (g.trail == 0).sum() == 6
    ev = g.step([moves[-1]])
    assert p.pos == (4, 6) and g.owner[4, 6] == 0
    assert (g.trail == 0).sum() == 0 and p.trail == []
    # captured the 6 trail cells plus the enclosed (3,6)
    assert ev.captured[0] == 7
    assert g.owner[3, 6] == 0
    assert g.territory(0) == 9 + 7


def test_own_trail_kills_self():
    g = make_game(size=12, n=1)
    place(g, 0, 8, 5, UP)  # owns rows 7-9
    p = g.players[0]
    # trail going up col 5 then right, head at (4,6) facing down
    for cell in [(6, 5), (5, 5), (4, 5), (4, 6)]:
        g.trail[cell] = 0
        p.trail.append(cell)
    p.pos, p.heading = (4, 6), DOWN
    g.step([TURN_RIGHT])  # facing down, turning right = heading left -> (4,5): own trail
    assert not p.alive and p.death_cause == "own_trail"


def test_cutting_trail_kills_owner_and_credits_killer():
    g = make_game(size=14)
    place(g, 0, 7, 3, RIGHT)  # head (7,3), owns cols 2-4
    place(g, 1, 2, 10, LEFT)
    p0, p1 = g.players
    p0.pos = (7, 4)
    for cell in [(5, 8), (6, 8), (7, 8), (8, 8)]:
        g.trail[cell] = 1
        p1.trail.append(cell)
    p1.pos, p1.heading = (8, 8), DOWN
    # p0 walks right along row 7: (7,5), (7,6), (7,7), (7,8) = p1's trail
    for _ in range(4):
        g.step([STRAIGHT, TURN_LEFT if p1.heading == DOWN else STRAIGHT])
        if not p1.alive:
            break
    assert not p1.alive and p1.death_cause == "trail_cut"
    assert p1.killed_by == 0 and p0.kills == 1 and p0.alive
    assert g.territory(1) == 0 and (g.trail == 1).sum() == 0


def test_head_on_both_outside_both_die():
    g = make_game(size=14)
    place(g, 0, 7, 2, RIGHT)
    place(g, 1, 7, 8, LEFT)
    # heads at col 2 and 8; walk toward each other: meet at col 5
    for _ in range(3):
        g.step([STRAIGHT, STRAIGHT])
    assert not g.players[0].alive and not g.players[1].alive
    assert g.players[0].death_cause == "head_on" == g.players[1].death_cause


def test_head_on_on_own_land_survives():
    g = make_game(size=14)
    place(g, 0, 7, 4, RIGHT, square=2)  # owns rows 5-9, cols 2-6; head (7,4)
    g.players[0].pos = (7, 5)
    place(g, 1, 2, 11, LEFT)
    p1 = g.players[1]
    p1.pos = (7, 7)
    for cell in [(7, 8), (7, 7)]:
        g.trail[cell] = 1
        p1.trail.append(cell)
    g.step([STRAIGHT, STRAIGHT])  # both enter (7,6), which p0 owns
    assert g.players[0].alive
    assert not p1.alive and p1.death_cause == "head_on" and p1.killed_by == 0
    assert g.players[0].kills == 1


def test_swap_outside_both_die():
    g = make_game(size=14)
    place(g, 0, 7, 2, RIGHT)
    place(g, 1, 7, 7, LEFT)
    g.step([STRAIGHT, STRAIGHT])  # p0 (7,3) own, p1 (7,6) own
    g.step([STRAIGHT, STRAIGHT])  # p0 (7,4) out, p1 (7,5) out -> adjacent
    assert g.players[0].alive and g.players[1].alive
    g.step([STRAIGHT, STRAIGHT])  # swap cells -> both die (each also hits the other's trail)
    assert not g.players[0].alive and not g.players[1].alive


def _build_loop_around(g):
    """p0 owns rows 5-7 cols 2-4 and has a trail around rows 4-8, cols 3-8,
    head at (8,3) facing left: turning right steps into (7,3) and closes it."""
    place(g, 0, 6, 3, LEFT)
    p0 = g.players[0]
    cells = [(4, c) for c in range(3, 9)] + [(r, 8) for r in range(5, 9)] + [(8, c) for c in range(7, 2, -1)]
    for cell in cells:
        g.trail[cell] = 0
        p0.trail.append(cell)
    p0.pos = (8, 3)
    return p0


def test_capture_steals_enclosed_enemy_land():
    g = make_game(size=14)
    p0 = _build_loop_around(g)
    place(g, 1, 11, 11, UP)  # p1 home far away
    g.owner[6, 7] = 1  # ...plus one cell inside p0's loop
    ev = g.step([TURN_RIGHT, STRAIGHT])
    assert p0.pos == (7, 3) and p0.trail == []
    assert g.owner[6, 7] == 0 and ev.lost[1] == 1
    # enclosed interior: rows 5-7, cols 5-7 (9 cells) + trail (15 cells)
    assert ev.captured[0] == 9 + 15
    assert g.players[1].alive


def test_render_follows_a_large_board():
    """A public-server arena must not become a multi-thousand-pixel picture."""
    from splixrl.render import render_rgb

    g = SplixGame(size=398, n_players=1, spawn_radius=0)
    g.players = [Player(pid=0, pos=(10, 10), heading=UP)]
    focused = render_rgb(g, 16, focus=(10, 10), radius=22)
    assert focused.shape == (45 * 16 + 4, 45 * 16 + 4, 3)
    # the same call with no focus still draws every cell
    small = SplixGame(size=20, n_players=1, spawn_radius=0)
    assert render_rgb(small, 8).shape[0] == 20 * 8 + 4


def test_losing_all_territory_eliminates():
    g = make_game(size=14)
    p0 = _build_loop_around(g)
    g.players.append(Player(pid=1, pos=(12, 12), heading=UP))
    g.owner[6, 7] = 1  # p1's only cell is inside the loop
    g.step([TURN_RIGHT, STRAIGHT])
    p1 = g.players[1]
    assert not p1.alive and p1.death_cause == "territory_lost" and p1.killed_by == 0
    assert p0.kills == 1



def _enclave_game():
    g = SplixGame(size=40, n_players=2)
    g.reset(np.random.default_rng(0))
    me, en = g.players
    g.owner.fill(EMPTY)
    g.trail.fill(EMPTY)
    g.owner[5:30, 5:30] = 1  # enemy territory
    g.owner[14:20, 12:24] = 0  # my 6 x 12 enclave inside it
    me.pos, me.heading, me.trail = (16, 14), 0, []  # I'm in its left half
    return g, me, en


def test_surrounding_my_land_takes_none_of_it_while_i_stand_in_it():
    g, me, en = _enclave_game()
    en.pos, en.heading, en.trail = (6, 29), 1, []
    for a in [STRAIGHT, TURN_RIGHT, TURN_RIGHT]:  # a tiny loop on its east edge
        g.step([PAUSE, a])
    assert (g.owner == 0).sum() == 72 and me.alive


def test_a_line_through_my_land_takes_the_piece_im_not_in():
    g, me, en = _enclave_game()
    # the enemy runs a line down through the middle of my enclave (column 18)
    en.pos, en.heading, en.trail = (13, 18), 2, []
    for _ in range(7):
        g.step([PAUSE, STRAIGHT])
    assert g.owner[20, 18] == 1  # back on its own land below: the loop closed
    assert (g.owner[14:20, 19:24] == 1).all()  # the right piece, without my head, is theirs
    assert (g.owner[14:20, 12:18] == 0).all() and me.alive  # the piece I'm in is still mine


def test_encircling_an_enemy_takes_nothing_while_its_head_is_inside():
    g = SplixGame(size=30, n_players=2)
    g.reset(np.random.default_rng(0))
    me, en = g.players
    g.owner.fill(EMPTY)
    g.trail.fill(EMPTY)
    g.owner[2:5, 2:20] = 0
    g.owner[12:15, 12:15] = 1
    en.pos, en.heading, en.trail = (13, 13), 0, []
    # my trail: down from my land, left, and back up, around the enemy's land
    me.trail = [(r, 18) for r in range(5, 19)] + [(18, c) for c in range(17, 8, -1)] + [(r, 9) for r in range(17, 4, -1)]
    for cell in me.trail:
        g.trail[cell] = 0
    me.pos, me.heading = (5, 9), 0
    g.step([STRAIGHT, PAUSE])  # up onto my land: the loop closes
    assert g.owner[4, 9] == 0 and not me.trail
    assert (g.owner[12:15, 12:15] == 1).all() and en.alive  # it stood inside: nothing taken
    assert g.owner[10, 15] == EMPTY  # nor the empty land around it
    assert g.owner[18, 12] == 0  # the trail itself is mine



def test_cutter_bot_takes_the_part_of_an_enclave_its_owner_isnt_in():
    from splixrl.opponents import make_opponent

    g, me, en = _enclave_game()
    en.pos, en.heading, en.trail = (8, 20), 2, []
    bot = make_opponent("cutter")
    bot.reset(g, 1, np.random.default_rng(0))
    for _ in range(40):
        g.step([PAUSE, bot.act(g, 1)])
    mine = g.owner == 0
    assert me.alive and mine[16, 14] and 0 < mine.sum() < 72  # kept the piece I'm in, lost the rest
