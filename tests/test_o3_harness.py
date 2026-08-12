"""Tests for the external -O3 runtime baseline harness (pure logic only).

These tests cover the deterministic pieces of evaluation/o3_runtime_harness.py:
bootstrap CIs, arm statistics, summary aggregation, and URI slugging. The
actual compiler/execution protocol is validated end-to-end separately (it
requires CompilerGym and a built benchmark binary).
"""

from __future__ import annotations

import math

from evaluation.o3_runtime_harness import (
    INPUT_INDEX_LARGEST,
    arm_stats,
    bootstrap_ci,
    _baseline_median_sec,
    build_native,
    geo_mean,
    _slug,
    parse_inputs,
    resolve_input,
    summarize_results,
)


def test_slug_is_filesystem_safe():
    slug = _slug("benchmark://cbench-v1/qsort")
    assert "/" not in slug
    assert ":" not in slug
    assert slug == "benchmark__cbench-v1_qsort"


def test_bootstrap_ci_is_seeded_and_contains_median():
    samples = [0.10, 0.12, 0.11, 0.13, 0.09, 0.11, 0.14, 0.10, 0.12, 0.11]
    first = bootstrap_ci(samples, seed=42)
    again = bootstrap_ci(samples, seed=42)
    assert first == again  # reproducible
    assert first[0] <= first[1]
    assert first[0] <= sorted(samples)[len(samples) // 2] <= first[1]
    # CI should be tight for near-constant samples.
    tight = bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5], seed=1)
    assert tight[1] - tight[0] < 1e-3


def test_bootstrap_ci_single_sample():
    assert bootstrap_ci([0.7], seed=0) == (0.7, 0.7)


def test_arm_stats_shape():
    stats = arm_stats([0.1, 0.2, 0.3], seed=42)
    assert stats["n"] == 3
    assert stats["median_sec"] == 0.2
    assert abs(stats["mean_sec"] - 0.2) < 1e-12
    assert stats["std_sec"] is not None
    assert stats["ci95_lo_sec"] <= 0.2 <= stats["ci95_hi_sec"]
    empty = arm_stats([], seed=42)
    assert empty["n"] == 0
    assert empty["median_sec"] is None


def test_geo_mean():
    assert geo_mean([1.0, 4.0]) == 2.0
    assert geo_mean([]) is None


def test_baseline_median_sec_prefers_clang_o3():
    entry = {
        "o3": {"median_sec": 0.2},
        "clang_o3": {"median_sec": 0.3},
        "hybrid": {"median_sec": 0.1},
    }
    assert abs(_baseline_median_sec(entry) - 0.3) < 1e-12
    # Older rows without clang_o3 fall back to the opt-o3 median.
    legacy = {"o3": {"median_sec": 0.2}, "hybrid": {"median_sec": 0.1}}
    assert abs(_baseline_median_sec(legacy) - 0.2) < 1e-12
    assert _baseline_median_sec({}) is None


def test_build_native_emits_opt_level_argv(tmp_path, monkeypatch):
    import subprocess

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        proc = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        (tmp_path / "a.out").write_bytes(b"\x7fELF")
        return proc

    monkeypatch.setattr("evaluation.o3_runtime_harness.subprocess.run", fake_run)
    monkeypatch.setattr(
        "evaluation.o3_runtime_harness.compiler_gym_clang_path",
        lambda: tmp_path / "clang",
    )
    exe = build_native(
        b"BC", tmp_path, ["$CC", "$IN", "-o", "a.out", "-lm"], "a.out", 60
    )
    assert exe.name == "a.out"
    # $CC must expand to [clang, -O3] as separate argv entries, never
    # a single "clang -O3" token (which would fail to exec).
    assert captured["cmd"][0] == str(tmp_path / "clang")
    assert captured["cmd"][1] == "-O3"
    assert captured["cmd"][2].endswith("module.bc")
    assert captured["cmd"][3:] == ["-o", "a.out", "-lm"]


def test_build_native_fallback_uses_opt_level(tmp_path, monkeypatch):
    import subprocess

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        proc = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        (kwargs["cwd"] / "a.out").write_bytes(b"\x7fELF")
        return proc

    monkeypatch.setattr("evaluation.o3_runtime_harness.subprocess.run", fake_run)
    monkeypatch.setattr(
        "evaluation.o3_runtime_harness.compiler_gym_clang_path",
        lambda: tmp_path / "clang",
    )
    build_native(b"BC", tmp_path / "sub", [], "a.out", 60)
    assert captured["cmd"][0] == str(tmp_path / "clang")
    assert captured["cmd"][1] == "-O3"


def test_summarize_results_aggregation():
    rows = [
        {
            "benchmark_uri": "benchmark://cbench-v1/a",
            "suite": "cbench-v1",
            "protocol": "native",
            "o3": {"median_sec": 0.2, "ci95_lo_sec": 0.19, "ci95_hi_sec": 0.21},
            "clang_o3": {"median_sec": 0.3, "ci95_lo_sec": 0.29, "ci95_hi_sec": 0.31},
            "hybrid": {"median_sec": 0.1, "ci95_lo_sec": 0.09, "ci95_hi_sec": 0.11},
            "outputs_match": True,
            "pass_sequence": ["-gvn"],
            "hybrid_vs_o3_ir_pct": 10.0,
        },
        {
            "benchmark_uri": "benchmark://cbench-v1/b",
            "suite": "cbench-v1",
            "protocol": "native",
            "o3": {"median_sec": 0.1, "ci95_lo_sec": 0.09, "ci95_hi_sec": 0.11},
            "clang_o3": {"median_sec": 0.1, "ci95_lo_sec": 0.09, "ci95_hi_sec": 0.11},
            "hybrid": {"median_sec": 0.2, "ci95_lo_sec": 0.19, "ci95_hi_sec": 0.21},
            "outputs_match": True,
            "pass_sequence": ["-licm"],
            "hybrid_vs_o3_ir_pct": -5.0,
        },
        {
            "benchmark_uri": "benchmark://cbench-v1/c",
            "suite": "cbench-v1",
            "protocol": "native",
            "o3": {"median_sec": 0.4, "ci95_lo_sec": 0.39, "ci95_hi_sec": 0.41},
            "clang_o3": {"median_sec": 0.4, "ci95_lo_sec": 0.39, "ci95_hi_sec": 0.41},
            "hybrid": {"median_sec": 0.4, "ci95_lo_sec": 0.39, "ci95_hi_sec": 0.41},
            "outputs_match": True,
            "pass_sequence": ["-dce"],
            "hybrid_vs_o3_ir_pct": 0.0,
        },
    ]
    summary = summarize_results(rows)
    assert summary["benchmarks_evaluated"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    # geo-mean of speedups 2.0, 0.5, 1.0 == 1.0
    assert abs(summary["geo_mean_speedup"] - 1.0) < 1e-12
    # vs clang-o3: 3.0, 0.5, 1.0 -> geo-mean = (1.5)^(1/3)
    expected_c3 = math.exp(math.log(1.5) / 3)
    assert abs(summary["geo_mean_speedup_vs_clang_o3"] - expected_c3) < 1e-9
    row_a = next(r for r in summary["rows"] if r["benchmark_uri"].endswith("/a"))
    assert math.isclose(row_a["speedup_hybrid_vs_o3"], 2.0)
    assert math.isclose(row_a["speedup_hybrid_vs_clang_o3"], 3.0)
    assert summary["wins_vs_clang_o3"] == 1


def test_resolve_input_swaps_file_and_syncs_finfo(tmp_path):
    for name, size in (("1.dat", 100), ("2.dat", 200), ("3.dat", 50)):
        (tmp_path / name).write_bytes(b"x" * size)
    (tmp_path / "convert_tool").write_bytes(b"x" * 10)  # non-numeric: ignored
    run_args = ["./a.out", str(tmp_path / "1.dat")]
    pre = ["echo 1 >_finfo_dataset"]
    new_run, new_pre, info = resolve_input(run_args, pre, 1)
    assert new_run[1].endswith("2.dat")
    assert new_pre == ["echo 2 >_finfo_dataset"]
    assert info["input_index"] == 1
    assert info["input_candidates"] == 3


def test_resolve_input_largest_by_size(tmp_path):
    for name, size in (("1.dat", 100), ("2.dat", 300), ("3.dat", 50)):
        (tmp_path / name).write_bytes(b"x" * size)
    new_run, new_pre, info = resolve_input(
        ["./a.out", str(tmp_path / "1.dat")], ["echo 1 >_finfo_dataset"], INPUT_INDEX_LARGEST
    )
    assert new_run[1].endswith("2.dat")
    assert new_pre == ["echo 2 >_finfo_dataset"]


def test_resolve_input_prefers_same_suffix(tmp_path):
    (tmp_path / "1.txt").write_bytes(b"a" * 10)
    (tmp_path / "2.txt").write_bytes(b"b" * 10)
    (tmp_path / "3.enc").write_bytes(b"c" * 999)  # biggest, but wrong suffix
    new_run, _, info = resolve_input(
        ["./a.out", str(tmp_path / "1.txt")], [], 1
    )
    assert new_run[1].endswith("2.txt")
    assert info["input_candidates"] == 2


def test_resolve_input_swaps_entire_dataset_family(tmp_path):
    """stringsearch takes 1.txt (text) + 1.s.txt (needles): both must swap."""
    for name, size in (
        ("1.txt", 100),
        ("1.s.txt", 20),
        ("2.txt", 200),
        ("2.s.txt", 30),
        ("3.txt", 400),
        ("3.s.txt", 40),
    ):
        (tmp_path / name).write_bytes(b"x" * size)
    run_args = ["./a.out", str(tmp_path / "1.txt"), str(tmp_path / "1.s.txt"), "output.txt"]
    new_run, new_pre, info = resolve_input(
        run_args, ["echo 1 >_finfo_dataset"], INPUT_INDEX_LARGEST
    )
    assert new_run[1].endswith("3.txt")  # text file -> largest dataset
    assert new_run[2].endswith("3.s.txt")  # family member swapped too
    assert new_run[3] == "output.txt"  # non-existent output file untouched
    assert new_pre == ["echo 3 >_finfo_dataset"]
    assert info["input_candidates"] == 6  # 3 datasets x (text + needles)
    assert info["input_file"].endswith("3.txt")


def test_resolve_input_largest_preserves_chosen_file_extension(tmp_path):
    """tiff2rgba-style: 1.tif -> largest is 17.nocomp.tif, NOT 17.tif."""
    for name, size in (
        ("1.tif", 10),
        ("1.nocomp.tif", 20),
        ("2.tif", 30),
        ("2.nocomp.tif", 40),
        ("17.nocomp.tif", 9999),  # largest in the family
        ("17.tif", 50),
    ):
        (tmp_path / name).write_bytes(b"x" * size)
    new_run, new_pre, info = resolve_input(
        ["./a.out", str(tmp_path / "1.tif"), "output.tif"],
        ["echo 1 >_finfo_dataset"],
        INPUT_INDEX_LARGEST,
    )
    assert new_run[1].endswith("17.nocomp.tif")  # the actual chosen file
    assert new_run[2] == "output.tif"
    assert new_pre == ["echo 17 >_finfo_dataset"]
    assert info["input_file"].endswith("17.nocomp.tif")


def test_resolve_input_unchanged_without_input_file():
    run_args = ["./a.out", "1125000"]  # e.g. bitcount: no file argument
    new_run, new_pre, info = resolve_input(run_args, ["echo 1 >_finfo_dataset"], INPUT_INDEX_LARGEST)
    assert new_run == run_args
    assert new_pre == ["echo 1 >_finfo_dataset"]
    assert info["input_file"] is None


def test_parse_inputs():
    assert parse_inputs("0") == [0]
    assert parse_inputs("0,largest") == [0, INPUT_INDEX_LARGEST]
    assert parse_inputs("largest,2") == [INPUT_INDEX_LARGEST, 2]


def test_summarize_deduplicates_benchmarks_across_waves():
    """A benchmark measured with default AND large inputs appears once."""
    default_row = {
        "benchmark_uri": "benchmark://cbench-v1/x",
        "suite": "cbench-v1",
        "o3": {"median_sec": 0.01, "ci95_lo_sec": 0.009, "ci95_hi_sec": 0.011},
        "clang_o3": {"median_sec": 0.01, "ci95_lo_sec": 0.009, "ci95_hi_sec": 0.011},
        "hybrid": {"median_sec": 0.02, "ci95_lo_sec": 0.019, "ci95_hi_sec": 0.021},
        "inputs": [
            {
                "input_file": "1.dat",
                "o3": {"median_sec": 0.01, "ci95_lo_sec": 0.009, "ci95_hi_sec": 0.011},
                "clang_o3": {"median_sec": 0.01, "ci95_lo_sec": 0.009, "ci95_hi_sec": 0.011},
                "hybrid": {"median_sec": 0.02, "ci95_lo_sec": 0.019, "ci95_hi_sec": 0.021},
                "speedup_hybrid_vs_o3": 0.5,
                "speedup_hybrid_vs_clang_o3": 0.5,
            }
        ],
    }
    large_row = {
        "benchmark_uri": "benchmark://cbench-v1/x",
        "suite": "cbench-v1",
        "o3": {"median_sec": 1.0, "ci95_lo_sec": 0.9, "ci95_hi_sec": 1.1},
        "clang_o3": {"median_sec": 0.9, "ci95_lo_sec": 0.85, "ci95_hi_sec": 0.95},
        "hybrid": {"median_sec": 0.5, "ci95_lo_sec": 0.45, "ci95_hi_sec": 0.55},
        "inputs": [
            {
                "input_file": "9.dat",
                "o3": {"median_sec": 1.0, "ci95_lo_sec": 0.9, "ci95_hi_sec": 1.1},
                "clang_o3": {"median_sec": 0.9, "ci95_lo_sec": 0.85, "ci95_hi_sec": 0.95},
                "hybrid": {"median_sec": 0.5, "ci95_lo_sec": 0.45, "ci95_hi_sec": 0.55},
                "speedup_hybrid_vs_o3": 2.0,
                "speedup_hybrid_vs_clang_o3": 1.8,
            }
        ],
    }
    summary = summarize_results([default_row, large_row])
    assert summary["benchmarks_evaluated"] == 1  # not 2
    assert summary["rows"][0]["input_file"] == "9.dat"
    assert abs(summary["rows"][0]["speedup_hybrid_vs_o3"] - 2.0) < 1e-12
    assert abs(summary["rows"][0]["speedup_hybrid_vs_clang_o3"] - 1.8) < 1e-12
    assert summary["wins"] == 1
    assert summary["wins_vs_clang_o3"] == 1


def test_summarize_skips_rows_without_medians():
    summary = summarize_results(
        [
            {
                "benchmark_uri": "x",
                "o3": {"median_sec": None},
                "hybrid": {"median_sec": 0.1},
            },
            {
                "benchmark_uri": "y",
                "status": "failed",
                "reason": "build error",
            },
        ]
    )
    assert summary["benchmarks_evaluated"] == 0
