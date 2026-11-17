"""Opponent specs.

A spec is a string (or an Opponent factory). String forms:

    "random"            uniform random moves
    "random_safe"       random, but avoids walls and its own trail
    "loop"              LoopBot: small rectangular loops, runs home when threatened
    "chaser"            ChaserBot: hunts nearby enemy trails, otherwise loops
    "greedy"            LoopBot drawing big loops and ignoring danger until late:
                        lots of land, and long trails to cut (kill practice)
    "invader"           InvaderBot: raids the nearest enemy land and takes it home
    "loop?max_out=8&danger_dist=2"   bot with keyword arguments
    "frozen:path/to/agent.pt"        a saved RL agent, frozen (never updated)

Pass a list of specs as the env's opponent pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .base import Opponent, bfs_first_action, safe_actions
from .bots import ChaserBot, InvaderBot, LoopBot, RandomBot

BOTS: dict[str, Callable[..., Opponent]] = {
    "random": lambda **kw: RandomBot(safe=False, **kw),
    "random_safe": lambda **kw: RandomBot(safe=True, **kw),
    "loop": LoopBot,
    "chaser": ChaserBot,
    "invader": InvaderBot,
    "greedy": lambda **kw: LoopBot(**{
        "min_out": 5, "max_out": 12, "min_side": 5, "max_side": 14,
        "danger_dist": 1, "max_trail": 70, "keep_away": 1, **kw,
    }),
}

# the fixed opponent set used for the main comparison (train + eval)
DEFAULT_POOL = ("random_safe", "loop", "chaser")


def _parse_value(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def make_opponent(spec: Any) -> Opponent:
    if isinstance(spec, Opponent):
        raise TypeError("pass a spec string or a factory, not an Opponent instance (they hold per-episode state)")
    if callable(spec) and not isinstance(spec, str):
        return spec()
    if not isinstance(spec, str):
        raise TypeError(f"bad opponent spec {spec!r}")
    if spec.startswith("frozen:"):
        from .frozen import FrozenOpponent

        return FrozenOpponent(spec[len("frozen:") :])
    name, _, query = spec.partition("?")
    if name not in BOTS:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(BOTS)} or frozen:<path>")
    kwargs = {}
    if query:
        for part in query.split("&"):
            k, _, v = part.partition("=")
            kwargs[k] = _parse_value(v)
    return BOTS[name](**kwargs)


def opponent_label(spec: Any) -> str:
    """Short name for logs and tables."""
    if isinstance(spec, str):
        if spec.startswith("frozen:"):
            p = Path(spec[len("frozen:") :])
            return "frozen:" + "/".join(p.with_suffix("").parts[-3:])
        return spec
    return getattr(spec, "__name__", repr(spec))


__all__ = [
    "BOTS",
    "DEFAULT_POOL",
    "ChaserBot",
    "InvaderBot",
    "LoopBot",
    "Opponent",
    "RandomBot",
    "bfs_first_action",
    "make_opponent",
    "opponent_label",
    "safe_actions",
]
