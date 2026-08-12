"""Tests for the external -O3 runtime baseline harness (pure logic only).

These tests cover the deterministic pieces of evaluation/o3_runtime_harness.py:
bootstrap CIs, arm statistics, summary aggregation, and URI slugging. The
actual compiler/execution protocol is validated end-to-end separately (it
requires CompilerGym and a built benchmark binary).
"""

from __future__ import annotations

import math

from evaluation.o3_runtime_harness import (
    arm_stats,
    bootstrap_ci,
    geo_mean,
    _slug,
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


def test_summarize_results_aggregation():
    rows = [
        {
            "benchmark_uri": "benchmark://cbench-v1/a",
            "suite": "cbench-v1",
            "protocol": "native",
            "o3": {"median_sec": 0.2, "ci95_lo_sec": 0.19, "ci95_hi_sec": 0.21},
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
    assert math.isclose(
        summary["rows"][0]["speedup_hybrid_vs_o3"], 2.0
    )


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
