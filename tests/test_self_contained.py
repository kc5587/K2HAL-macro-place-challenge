"""CI test that scans placer.py's imports and asserts every macro_place.*
or submissions.* import resolves to a file inside the repo."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLACER = ROOT / "submissions" / "macro_placer" / "cd_lns_placer.py"


def _resolve_module(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or spec.origin == "built-in":
        return None
    return Path(spec.origin)


@pytest.mark.unit
def test_placer_imports_resolve_inside_repo() -> None:
    tree = ast.parse(PLACER.read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name)
    project_imports = {
        m for m in seen if m.startswith(("macro_place", "submissions"))
    }
    for mod in project_imports:
        path = _resolve_module(mod)
        assert path is not None, f"{mod} does not resolve to a file"
        assert ROOT in path.parents, (
            f"{mod} resolves outside repo: {path} (would DQ on eval server)"
        )


@pytest.mark.unit
def test_assert_self_contained_runs() -> None:
    """Smoke: assert_self_contained() should not raise for the current tree."""
    from submissions.macro_placer._audit import assert_self_contained
    assert_self_contained()
