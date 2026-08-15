"""Tests for the scale-free state representation (episode-independent ratio
features) used by the OOD RL experiment, and the --feature-cols override.

The representation change: replace the 6 absolute IR/size/block/function
counts (which carry benchmark-size information and shift out of range across
benchmark families) with 5 per-state ratios that are size-agnostic.
"""

from training.common import (
    SCALE_FREE_RATIO_COLS,
    derive_ratio_features,
)


def _row(**overrides):
    row = {
        "pre_ir_instruction_count": "1000",
        "pre_total_functions": "10",
        "pre_total_memory_instructions": "300",
        "pre_total_instructions": "800",
        "pre_object_text_size_bytes": "4000",
        "pre_total_basic_blocks": "40",
    }
    row.update(overrides)
    return row


def test_derive_ratio_features_matches_expected_math():
    out = derive_ratio_features(_row())
    assert out["pre_ir_per_func"] == 100.0  # 1000 / 10
    assert out["pre_mem_frac"] == 300 / 800
    assert out["pre_size_per_inst"] == 4000 / 800
    assert out["pre_blocks_per_func"] == 40 / 10
    assert out["pre_insts_per_block"] == 800 / 40
    assert set(out.keys()) == set(SCALE_FREE_RATIO_COLS)


def test_derive_ratio_features_post_prefix():
    post_row = {k.replace("pre_", "post_", 1): v for k, v in _row().items()}
    out = derive_ratio_features(post_row, prefix="post_")
    assert "post_ir_per_func" in out
    assert out["post_ir_per_func"] == 100.0
    assert "pre_ir_per_func" not in out


def test_derive_ratio_features_zero_denominator_guard():
    out = derive_ratio_features(_row(pre_total_functions="0", pre_total_instructions="0"))
    assert out["pre_ir_per_func"] == 0.0
    assert out["pre_mem_frac"] == 0.0
    assert out["pre_insts_per_block"] == 0.0


def test_derive_ratio_features_missing_values():
    out = derive_ratio_features({"pre_ir_instruction_count": "100"})
    # missing denoms -> 0.0, no exception
    assert out["pre_ir_per_func"] == 0.0
    assert isinstance(out["pre_ir_per_func"], float)


def test_ratio_features_are_size_invariant():
    # A 10x larger program with identical shape yields identical ratios.
    small = derive_ratio_features(_row())
    big = derive_ratio_features(_row(
        pre_ir_instruction_count="10000",
        pre_total_functions="100",
        pre_total_memory_instructions="3000",
        pre_total_instructions="8000",
        pre_object_text_size_bytes="40000",
        pre_total_basic_blocks="400",
    ))
    for col in SCALE_FREE_RATIO_COLS:
        assert small[col] == big[col], col


def test_feature_cols_override_names_align_with_derive_output():
    # The names a user passes to train_rl --feature-cols must match exactly
    # what derive_ratio_features emits so online inference can inject them.
    assert SCALE_FREE_RATIO_COLS == [
        "pre_ir_per_func",
        "pre_mem_frac",
        "pre_size_per_inst",
        "pre_blocks_per_func",
        "pre_insts_per_block",
    ]
