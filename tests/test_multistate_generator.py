"""Tests for the multi-state pass-quality dataset generator (pure logic).

The generator's measurement loop requires CompilerGym and native builds, so
these tests cover the deterministic helpers and the design invariants that
matter for dataset quality: state signatures are stable, and the state-builder
pass set must be disjoint from the no-op-prone loop passes (a random
loop-pass prefix collapses states — the failure mode that produced duplicate
state_ids and identical pre-state features in the first version).
"""

from __future__ import annotations

import hashlib

from scripts.generate_multistate_dataset import (
    LOOP_PASSES,
    REPLAY_FIELDS,
    STATE_BUILDER_PASSES,
    STATE_DIVERSITY_THRESHOLD,
    _feature_distance,
    _replay_episode_id,
    _state_signature,
    build_replay_row,
)


def test_state_signature_is_deterministic_and_distinct():
    a = b"module one"
    b = b"module two"
    assert _state_signature(a) == _state_signature(a)
    assert _state_signature(a) != _state_signature(b)
    # 16 hex chars, derived from sha256 of the bitcode bytes.
    assert len(_state_signature(a)) == 16
    assert _state_signature(a) == hashlib.sha256(a).hexdigest()[:16]


def _replay_feats(ir):
    return {
        "pre_ir_instruction_count": str(ir),
        "pre_object_text_size_bytes": "100",
        "pre_total_basic_blocks": "3",
        "pre_total_functions": "1",
        "pre_total_instructions": str(ir),
        "pre_total_memory_instructions": "2",
        "pre_autophase_TotalInsts": str(ir),
    }


def test_replay_episode_id_is_deterministic_per_state():
    a = _replay_episode_id("benchmark://cbench-v1/gsm", "sig1")
    b = _replay_episode_id("benchmark://cbench-v1/gsm", "sig1")
    c = _replay_episode_id("benchmark://cbench-v1/gsm", "sig2")
    assert a == b
    assert a != c
    assert len(a) == 24


def test_replay_row_schema_and_relabeling():
    row = build_replay_row(
        "benchmark://cbench-v1/gsm",
        state_index=1,
        step_index=2,
        pass_flag="-licm",
        state_id="sig_pre",
        post_state_id="sig_post",
        state_features=_replay_feats(100),
        post_features=_replay_feats(90),
        state_ir=100,
        post_ir=90,
        state_med=1.0,
        post_med=0.9,
        improvement=10.0,
    )
    # Every non-autophase schema field is present (autophase columns are
    # filled by the real extractor; the synthetic fixture provides one).
    core_fields = [f for f in REPLAY_FIELDS if "autophase" not in f]
    missing = [f for f in core_fields if f not in row]
    assert not missing, missing
    # Post features are relabeled copies of the post-state pre_* row.
    assert row["post_ir_instruction_count"] == "90"
    assert row["post_autophase_TotalInsts"] == "90"
    assert row["pre_autophase_TotalInsts"] == "100"
    # One-step episode semantics: done, raw runtime reward, ids wired.
    assert row["done"] == "True"
    assert row["hybrid_reward"] == "10.000000"
    assert row["pre_state_id"] == "sig_pre"
    assert row["post_state_id"] == "sig_post"
    assert row["episode_id"] == _replay_episode_id("benchmark://cbench-v1/gsm", "sig_pre")


def test_loop_passes_are_not_state_builders():
    """The no-op-prone loop passes (verified no-ops at O0 on cBench, e.g.
    -loop-unroll on dijkstra) must never be used to construct the non-O0
    states, or states collapse into duplicates. -licm is an exception: it is
    loop-related but performs real IR transformation, so it may build states."""
    noop_loop = set(LOOP_PASSES) - {"-licm"}
    assert noop_loop.isdisjoint(STATE_BUILDER_PASSES)
    assert len(LOOP_PASSES) == 8
    assert "-loop-unroll" in LOOP_PASSES
    assert "-sroa" in STATE_BUILDER_PASSES  # IR-reducing builder passes


def _feature_row(autophase: list, ir: int) -> dict:
    """Minimal pre-state feature row with the keys the distance metric reads."""
    row = {f"pre_autophase_{i}": str(v) for i, v in enumerate(autophase)}
    row["pre_ir_instruction_count"] = str(ir)
    return row


def test_feature_distance_detects_near_duplicates():
    """The diversity guard must reject the verified near-duplicate failure
    mode (a pass like -memcpyopt changes the bitcode but leaves the
    model-visible features identical) while accepting genuinely different
    states."""
    o0 = _feature_row([10.0, 90.0], 450)
    near_duplicate = _feature_row([10.0, 90.0], 450)  # memcpyopt-style: same features
    different = _feature_row([70.0, 30.0], 358)  # newgvn-style: new histogram + IR
    # Near-duplicates measure far below the acceptance threshold.
    assert _feature_distance(o0, near_duplicate) < STATE_DIVERSITY_THRESHOLD
    # Genuinely different states measure far above it.
    assert _feature_distance(o0, different) > STATE_DIVERSITY_THRESHOLD


def test_feature_distance_is_symmetric_and_ir_sensitive():
    same_hist = _feature_row([50.0, 50.0], 1000)
    same_hist_other_ir = _feature_row([50.0, 50.0], 700)
    d1 = _feature_distance(same_hist, same_hist_other_ir)
    d2 = _feature_distance(same_hist_other_ir, same_hist)
    assert d1 == d2  # symmetric
    # Relative IR change alone must push past the threshold.
    assert d1 > STATE_DIVERSITY_THRESHOLD
