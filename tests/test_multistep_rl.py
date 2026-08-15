"""Unit tests for the multi-step RL redesign (Phases 5-6).

Covers the 7 required properties:
  1. terminal target: y = r
  2. non-terminal target: y = r + gamma * max Q(s', a')
  3. STOP is available as an action
  4. Q(STOP) is handled as a terminal action
  5. self-loop/no-op behavior is handled correctly (no bootstrap through s'==s)
  6. state chaining is correct (post(t) == pre(t+1) within an episode)
  7. evaluation benchmarks cannot enter the RL buffer
plus an end-to-end check that fitted-Q actually propagates future value
(the property the old all-done=True buffers never exercised).
"""

from __future__ import annotations

import argparse
import json

import pytest

from scripts.inspect_rl_buffer import verify_episode_chaining
from scripts.verify_split_leakage import (
    EVAL_BENCHMARKS,
    verify_no_eval_leakage,
)
from training.train_rl import (
    STOP_FLAG,
    _parse_available_actions,
    encode_state_action,
    fqi_target,
    train_sklearn_dqn,
)


# --- 1. terminal target ---------------------------------------------------

def test_terminal_target_is_immediate_reward():
    # done=True: future value must NOT enter the target.
    assert fqi_target(reward=2.5, gamma=0.95, qmax_next=99.0, done=True, is_self_loop=False) == 2.5
    assert fqi_target(reward=0.0, gamma=0.95, qmax_next=-5.0, done=True, is_self_loop=False) == 0.0


# --- 2. non-terminal target ----------------------------------------------

def test_nonterminal_target_includes_future_value():
    y = fqi_target(reward=1.0, gamma=0.95, qmax_next=2.0, done=False, is_self_loop=False)
    assert y == pytest.approx(1.0 + 0.95 * 2.0)


# --- 3. STOP is available as an action -----------------------------------

def test_stop_is_a_real_action_member():
    assert STOP_FLAG == "-stop"
    # available_actions parsed from a buffer row always include STOP (the
    # option to stop is never masked).
    avail = _parse_available_actions({"available_actions": json.dumps(["-licm", "-gvn"])})
    assert "-licm" in avail and "-gvn" in avail
    assert STOP_FLAG in avail
    # Legacy buffers without the column fall back to the full vocabulary.
    assert _parse_available_actions({"available_actions": ""}) is None


# --- 4. STOP handled as a terminal action --------------------------------

def test_stop_row_is_terminal():
    # A STOP transition carries r=0 and done=True; its target is 0.
    assert fqi_target(reward=0.0, gamma=0.95, qmax_next=None, done=True, is_self_loop=False) == 0.0
    # Even if a STOP row were mislabeled non-terminal, a self-loop (s'==s, the
    # STOP row's post state is its pre state) must clamp the target to r.
    assert fqi_target(reward=0.0, gamma=0.95, qmax_next=10.0, done=False, is_self_loop=True) == 0.0


# --- 5. self-loop / no-op handling ---------------------------------------

def test_self_loop_never_bootstraps():
    # s' == s: no bootstrap, no matter how large the (illegitimate) qmax is.
    assert fqi_target(reward=0.3, gamma=0.95, qmax_next=50.0, done=False, is_self_loop=True) == 0.3
    # Missing next state is also treated as terminal.
    assert fqi_target(reward=0.3, gamma=0.95, qmax_next=None, done=False, is_self_loop=False) == 0.3


# --- 6. state chaining ----------------------------------------------------

def _row(eid, step, flag, done, pre, post, avail=None, stop=False):
    r = {
        "episode_id": eid,
        "benchmark_uri": "benchmark://cbench-v1/qsort",
        "step_index": str(step),
        "pass_flag": "-stop" if stop else flag,
        "done": "True" if done else "False",
        "hybrid_reward": "0.0" if stop else "1.0",
        "pre_state_id": pre,
        "post_state_id": pre if stop else post,
    }
    if not stop:
        r["available_actions"] = json.dumps(avail or [])
    return r


def test_chaining_verifier_accepts_chained_buffer():
    rows = [
        _row("e1", 0, "-p0", False, "s0", "s1", avail=["-p1"]),
        _row("e1", 1, "-p1", True, "s1", "s2", avail=[]),
        _row("e2", 0, "-p0", False, "s0", "s1", avail=["-p1"]),
        _row("e2", 1, "-stop", True, "s1", "s1", stop=True),
    ]
    assert verify_episode_chaining(rows) == []


def test_chaining_verifier_flags_independent_rows():
    # The old failure mode: rows measured from unrelated copies of S0 whose
    # post states never feed the next pre state.
    rows = [
        _row("e1", 0, "-p0", False, "s0", "sX", avail=["-p1"]),
        _row("e1", 1, "-p1", True, "s0", "sY", avail=[]),  # pre should be sX
    ]
    violations = verify_episode_chaining(rows)
    assert len(violations) == 1
    assert "sX" in violations[0]


# --- 7. evaluation benchmarks cannot enter the buffer ---------------------

def test_eval_benchmark_leakage_is_detected():
    clean = [{"benchmark_uri": "benchmark://cbench-v1/qsort"}]
    violations, counts = verify_no_eval_leakage(clean)
    assert violations == []
    assert counts["benchmark://cbench-v1/qsort"] == 1

    leaked = [{"benchmark_uri": EVAL_BENCHMARKS[0]}]
    violations, _ = verify_no_eval_leakage(leaked)
    assert len(violations) == 1
    assert EVAL_BENCHMARKS[0] in violations[0]


# --- end-to-end: fitted-Q must propagate future value ---------------------

def _chained_rows(n_copies=40):
    """Chain: s0 -p0-> s1 -p1-> s2 (terminal) + STOP rows at s1.

    Immediate rewards 1.0 per pass. Myopic Q(s0,-p0) ~ 1.0; bootstrapping
    Q(s0,-p0) ~ 1 + gamma*1 = 1.95. Replicated so HistGB's default
    min_samples_leaf=20 can actually split.
    """
    def state(n):
        return {"pre_f0": str(n), "pre_f1": "0", "pre_f2": "0"}

    def post_features(n):
        return {"post_f0": str(n), "post_f1": "0", "post_f2": "0"}

    rows = []
    for i in range(n_copies):
        eid = f"e{i}"
        rows += [
            {
                "episode_id": eid, "benchmark_uri": "benchmark://cbench-v1/qsort",
                "step_index": "0", "pass_flag": "-p0", "done": "False",
                "hybrid_reward": "1.0", "pre_state_id": "s0", "post_state_id": "s1",
                "available_actions": json.dumps(["-p1"]),
                **state(0), **post_features(1),
            },
            {
                "episode_id": eid, "benchmark_uri": "benchmark://cbench-v1/qsort",
                "step_index": "1", "pass_flag": "-p1", "done": "True",
                "hybrid_reward": "1.0", "pre_state_id": "s1", "post_state_id": "s2",
                "available_actions": json.dumps([]),
                **state(1), **post_features(2),
            },
            {
                "episode_id": eid, "benchmark_uri": "benchmark://cbench-v1/qsort",
                "step_index": "0", "pass_flag": "-stop", "done": "True",
                "hybrid_reward": "0.0", "pre_state_id": "s1", "post_state_id": "s1",
                "available_actions": "",
                **state(1), **post_features(1),
            },
        ]
    return rows


def test_fitted_q_propagates_future_value():
    rows = _chained_rows()
    args = argparse.Namespace(gamma=0.95, q_iterations=5, action_encoding="one_hot")
    model = train_sklearn_dqn(
        rows, ["pre_f0", "pre_f1", "pre_f2"],
        {"-p0": 0, "-p1": 1, STOP_FLAG: 2}, args,
    )

    def q(state_row, flag):
        x = encode_state_action(
            state_row, flag,
            ["pre_f0", "pre_f1", "pre_f2"],
            {"-p0": 0, "-p1": 1, STOP_FLAG: 2}, "one_hot",
        )
        return float(model.predict([x])[0])

    q_s0_p0 = q({"pre_f0": "0", "pre_f1": "0", "pre_f2": "0"}, "-p0")
    q_s0_p1 = q({"pre_f0": "0", "pre_f1": "0", "pre_f2": "0"}, "-p1")
    q_s0_stop = q({"pre_f0": "0", "pre_f1": "0", "pre_f2": "0"}, STOP_FLAG)
    q_s1_p1 = q({"pre_f0": "1", "pre_f1": "0", "pre_f2": "0"}, "-p1")

    # Bootstrapping must make (s0, -p0) worth more than its immediate reward
    # (1.0) AND more than the terminal pass (s1, -p1), which has no future.
    assert q_s0_p0 > 1.3, f"Q(s0,-p0)={q_s0_p0} — future value did not propagate"
    assert q_s0_p0 > q_s1_p1 + 0.3, f"{q_s0_p0} vs terminal {q_s1_p1}"
    # And the one-step alternative / STOP at s0 must rank below it.
    assert q_s0_p0 > q_s0_p1
    assert q_s0_p0 > q_s0_stop
