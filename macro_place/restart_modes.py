"""Lever R — Restart-mode parameter dispatch.

Refactors the per-mode parameter selection out of ``cd_lns_placer._run_restart``
into a pure function, and adds a new ``"exploratory"`` mode designed to land in
a genuinely different basin from the existing 3 modes.

Existing modes:
  - ``"conservative"`` — tight radius (1/32), 5 sweeps, no LNS, sigma=0.
  - ``"light"``        — medium radius (1/16), 10 sweeps, no LNS, sigma=0.02.
  - ``"aggressive"``   — config-driven radius/sweeps, LNS on, config sigma.

New mode:
  - ``"exploratory"``  — config sweeps, LNS on, **larger** initial radius (0.40)
                         and **higher** warm-start sigma (0.20) so the starting
                         basin is meaningfully different from the conservative-
                         light-aggressive cluster.

Default ``restart_modes`` in the placer config is unchanged. Users opt in by
setting ``restart_modes = ("conservative", "light", "aggressive", "exploratory")``
or any other tuple containing the new mode.
"""
from __future__ import annotations

from typing import Any, Mapping


def mode_params(mode: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return per-mode CD/LNS parameters as a plain dict.

    Unknown modes fall back to ``"aggressive"`` (matches the prior
    behavior of the ``else`` branch in ``_run_restart``).
    """
    if mode == "conservative":
        return {
            "radius_init_ratio": 1.0 / 32.0,
            "max_sweeps": 5,
            "do_lns": False,
            "warm_sigma": 0.0,
        }
    if mode == "light":
        return {
            "radius_init_ratio": 1.0 / 16.0,
            "max_sweeps": 10,
            "do_lns": False,
            "warm_sigma": 0.02,
        }
    if mode == "exploratory":
        # New: high-sigma warm start + larger initial CD radius to land in a
        # basin separated from the conservative/light/aggressive cluster.
        return {
            "radius_init_ratio": 0.40,
            "max_sweeps": int(cfg.get("max_sweeps", 30)),
            "do_lns": True,
            "warm_sigma": 0.20,
        }
    # Default: aggressive (matches the `else` branch in _run_restart).
    return {
        "radius_init_ratio": float(cfg.get("radius_init_ratio", 0.25)),
        "max_sweeps": int(cfg.get("max_sweeps", 30)),
        "do_lns": True,
        "warm_sigma": float(cfg.get("warm_start_sigma", 0.05)),
    }
