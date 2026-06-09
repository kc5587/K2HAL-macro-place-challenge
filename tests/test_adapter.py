from __future__ import annotations

import torch

from macro_place.benchmark import Benchmark


def _benchmark(name: str = "ibm01") -> Benchmark:
    return Benchmark(
        name=name,
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=1,
        macro_positions=torch.zeros((1, 2), dtype=torch.float32),
        macro_sizes=torch.ones((1, 2), dtype=torch.float32),
        macro_fixed=torch.zeros((1,), dtype=torch.bool),
        macro_names=["m0"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros((0,), dtype=torch.float32),
        grid_rows=1,
        grid_cols=1,
        num_hard_macros=1,
        num_soft_macros=0,
    )


def test_resolve_plc_finds_challenge_benchmark_from_self_contained_submission(
    monkeypatch,
    tmp_path,
) -> None:
    """Self-contained swag harness runs our code from /submission but data is in cwd."""
    from macro_place import adapter

    submission_root = tmp_path / "submission"
    challenge_root = tmp_path / "challenge"
    ibm_dir = (
        challenge_root
        / "external"
        / "MacroPlacement"
        / "Testcases"
        / "ICCAD04"
        / "ibm01"
    )
    ibm_dir.mkdir(parents=True)
    (ibm_dir / "netlist.pb.txt").write_text("netlist", encoding="utf-8")

    expected_plc = object()
    loaded_dirs: list[str] = []

    def fake_load_benchmark_from_dir(path: str):
        loaded_dirs.append(path)
        return _benchmark(), expected_plc

    monkeypatch.setattr(adapter, "_REPO_ROOT", submission_root)
    monkeypatch.setattr(
        adapter,
        "_IBM_ROOT",
        submission_root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
    )
    monkeypatch.setattr(
        adapter,
        "_NG45_ROOT",
        submission_root / "external" / "MacroPlacement" / "Flows" / "NanGate45",
    )
    monkeypatch.chdir(challenge_root)
    monkeypatch.setattr(adapter, "load_benchmark_from_dir", fake_load_benchmark_from_dir)

    assert adapter.resolve_plc(_benchmark()) is expected_plc
    assert loaded_dirs == [str(ibm_dir)]
