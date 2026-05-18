"""One-off probe: run smoke_one_bench.run_bench with hessian_escape_enabled=True
to validate whether E12 finds escapes on a real benchmark.

Single-bench by design — small probe before committing to a full smoke.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from smoke_one_bench import run_bench  # noqa: E402  (in scripts/ alongside this file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="ibm06")
    parser.add_argument("--time-budget", type=float, default=1800.0)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    overrides = {
        "hessian_escape_enabled": True,
    }
    t0 = time.perf_counter()
    result = run_bench(
        bench_name=args.bench,
        time_budget_s=args.time_budget,
        num_restarts=args.num_restarts,
        config_overrides=overrides,
    )
    result["wall_s"] = time.perf_counter() - t0
    result["overrides"] = overrides

    out_dir = args.out_dir or (_REPO / "output" / f"hessian_probe_{args.bench}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bench}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
