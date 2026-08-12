"""Tests for the learned STOP action in the RL trainer.

STOP is a real member of the agent's action vocabulary: fitted-Q learns
Q(state, STOP) from synthetic terminal transitions (reward 0, done=True), so
the agent stops exactly when every available pass is expected to be
net-negative.
"""

from __future__ import annotations

from training.train_rl import STOP_FLAG, synthesize_stop_transitions


def _episode_rows():
    return [
        {
            "episode_id": "ep1",
            "benchmark_uri": "benchmark://cbench-v1/qsort",
            "pass_flag": "-gvn",
            "hybrid_reward": "5.0",
            "done": "False",
            "pre_state_id": "s0",
            "post_state_id": "s1",
            "pre_ir_instruction_count": "100",
            "post_ir_instruction_count": "90",
        },
        {
            "episode_id": "ep1",
            "benchmark_uri": "benchmark://cbench-v1/qsort",
            "pass_flag": "-licm",
            "hybrid_reward": "2.0",
            "done": "True",
            "pre_state_id": "s1",
            "post_state_id": "s2",
            "pre_ir_instruction_count": "90",
            "post_ir_instruction_count": "85",
        },
        {
            "episode_id": "ep2",
            "benchmark_uri": "benchmark://cbench-v1/gsm",
            "pass_flag": "-sroa",
            "hybrid_reward": "1.0",
            "done": "True",
            "pre_state_id": "t0",
            "post_state_id": "t1",
            "pre_ir_instruction_count": "50",
            "post_ir_instruction_count": "48",
        },
    ]


def test_stop_transitions_only_from_terminal_rows():
    rows = _episode_rows()
    stop = synthesize_stop_transitions(rows)
    # ep1 and ep2 have a done=True row; ep1's first row is not terminal.
    assert len(stop) == 2
    assert {r["episode_id"] for r in stop} == {"ep1", "ep2"}


def test_stop_transition_has_zero_reward_and_terminal_flag():
    rows = _episode_rows()
    stop = synthesize_stop_transitions(rows)
    for r in stop:
        assert r["pass_flag"] == STOP_FLAG
        assert r["hybrid_reward"] == "0.0"
        assert r["raw_step_reward"] == "0.0"
        assert r["done"].lower() == "true"


def test_stop_transition_is_terminal_state_self_loop():
    rows = _episode_rows()
    stop = synthesize_stop_transitions(rows)
    for r in stop:
        # Stopping applies no pass: post state == pre state.
        assert r["post_state_id"] == r["pre_state_id"]
        assert r["post_ir_instruction_count"] == r["pre_ir_instruction_count"]


def test_stop_transition_preserves_state_features():
    rows = _episode_rows()
    stop = synthesize_stop_transitions(rows)
    by_ep = {r["episode_id"]: r for r in stop}
    ep1 = by_ep["ep1"]
    assert ep1["pre_ir_instruction_count"] == "90"  # terminal state of ep1
    assert ep1["post_ir_instruction_count"] == "90"


def test_no_terminal_rows_produces_no_stop():
    rows = [dict(r, done="False") for r in _episode_rows()]
    assert synthesize_stop_transitions(rows) == []
