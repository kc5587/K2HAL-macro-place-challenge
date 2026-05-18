"""Self-containment audit run at submission import time.

Verifies every macro_place.* module the placer needs resolves to a file
inside this repository. If any required module is missing or resolves
outside the repo (e.g. a stale system-installed copy), raise ImportError
loudly so the submission fails-fast rather than silently DQ'ing on the
eval server.
"""
from __future__ import annotations

import importlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED = (
    "macro_place.fast_proxy",
    "macro_place.objective",
    "macro_place.legality",
    "macro_place.adapter",
    "macro_place.benchmark",
    "macro_place.nets",
    "macro_place.cd",
    "macro_place.lns_v2",
    "macro_place.hessian_escape",
    "macro_place.spatial_lns",
)


def assert_self_contained() -> None:
    for name in _REQUIRED:
        mod = importlib.import_module(name)
        origin = Path(getattr(mod, "__file__", "") or "")
        if _REPO_ROOT not in origin.parents:
            raise ImportError(
                f"submission requires {name} but it resolves to {origin} "
                f"outside the repo root {_REPO_ROOT}. Submission would DQ."
            )
