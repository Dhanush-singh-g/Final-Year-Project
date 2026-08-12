"""Tests for the per-benchmark z-scored runtime reward (research fix for the
cross-program-incomparable raw runtime target)."""

from __future__ import annotations

from scripts.reward import (
    add_zscored_runtime_column,
    per_benchmark_zscore,
)


def _rows():
    # Two benchmarks with different scales: A's raw improvements are ~10x B's.
    rows = []
    for name, base in (("a", 10.0), ("b", 1.0)):
        for i, delta in enumerate((-1, 0, 1)):
            rows.append(
                {
                    "benchmark_uri": f"benchmark://cbench-v1/{name}",
                    "pass_flag": f"-pass{i}",
                    "runtime_improvement_pct": str(base * delta),
                }
            )
    return rows


def test_per_benchmark_zscore_groups_by_benchmark():
    stats = per_benchmark_zscore(_rows())
    assert set(stats) == {"benchmark://cbench-v1/a", "benchmark://cbench-v1/b"}
    # Each group's mean is 0 and std is its own scale.
    assert abs(stats["benchmark://cbench-v1/a"][0]) < 1e-9
    assert abs(stats["benchmark://cbench-v1/b"][0]) < 1e-9
    # Population std of {-10, 0, 10} is sqrt(200/3); of {-1, 0, 1}, sqrt(2/3).
    assert abs(stats["benchmark://cbench-v1/a"][1] - (200.0 / 3.0) ** 0.5) < 1e-9
    assert abs(stats["benchmark://cbench-v1/b"][1] - (2.0 / 3.0) ** 0.5) < 1e-9


def test_zscore_makes_scales_comparable():
    stats = per_benchmark_zscore(_rows())
    # A's best pass (+10) and B's best pass (+1) must map to the same z.
    z_a_best = (10.0 - stats["benchmark://cbench-v1/a"][0]) / stats["benchmark://cbench-v1/a"][1]
    z_b_best = (1.0 - stats["benchmark://cbench-v1/b"][0]) / stats["benchmark://cbench-v1/b"][1]
    # A's best (+10) and B's best (+1) are the same number of stds above
    # their group mean (population std): identical z, so the two programs'
    # rewards are comparable.
    assert abs(z_a_best - z_b_best) < 1e-9
    assert z_a_best > 1.0 and z_a_best < 1.5
    assert z_a_best == z_a_best  # finite


def test_add_zscored_runtime_column_outputs_blanks_for_missing():
    rows = _rows()
    rows.append({"benchmark_uri": "benchmark://cbench-v1/solo", "pass_flag": "-p", "runtime_improvement_pct": ""})
    out = add_zscored_runtime_column(rows)
    assert out[-1]["z_runtime_improvement_pct"] == ""
    scored = [r for r in out if r["z_runtime_improvement_pct"]]
    assert len(scored) == 6
    # Mean of z-scores across a benchmark is ~0.
    za = [float(r["z_runtime_improvement_pct"]) for r in out if r["benchmark_uri"].endswith("/a")]
    assert abs(sum(za)) < 1e-9


def test_single_candidate_group_is_skipped():
    rows = [{"benchmark_uri": "b1", "pass_flag": "-p", "runtime_improvement_pct": "5.0"}]
    assert per_benchmark_zscore(rows) == {}
    out = add_zscored_runtime_column(rows)
    assert out[0]["z_runtime_improvement_pct"] == ""
