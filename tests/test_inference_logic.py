"""Regression tests for the inference episode-control fixes.

Covers the review findings:
- ``training/inference.py`` no longer ignores no-op actions (the old code
  printed "terminating" but kept optimizing and could repeat the same pass).
- Actions are masked per (state_id, action); a no-op action is never
  re-selected in the same state.
- Inference selection is deterministic by default.
- ``scripts/collect_rl_transitions.py`` episodes are reproducible across
  resume (per-episode RNG instead of a shifted global stream).
"""

from __future__ import annotations

from training.inference import STOP_FLAG, select_action, step_is_no_op
from scripts.collect_rl_transitions import episode_id_for, episode_rng_for


def test_step_is_no_op_detects_all_signals():
    # Environment flag: action had no effect.
    assert step_is_no_op(True, True, 10, 1.0)
    # State did not move.
    assert step_is_no_op(None, False, 10, 1.0)
    # No IR change AND zero hybrid reward.
    assert step_is_no_op(None, True, 0, 0.0)
    # Real progress is never a no-op.
    assert not step_is_no_op(None, True, 5, 1.0)


def test_select_action_returns_none_when_nothing_available():
    assert select_action([], [("-gvn", 1.0, 1.0)]) is None


def test_select_action_masks_tried_actions():
    sl_ranked = [("-gvn", 5.0, 0.5), ("-licm", 3.0, 0.3), ("-sroa", 1.0, 0.2)]
    # -gvn already tried in this state: it must not be re-selected.
    picked = select_action(["-licm", "-sroa"], sl_ranked)
    assert picked in ("-licm", "-sroa")


def test_select_action_rl_prefers_available_pick():
    sl = [("-a", 1.0, 1.0)]
    q = {"-a": 0.5, "-b": 9.0, "-c": 8.0}
    # Agent's own pick was masked: fall back to the masked Q argmax.
    picked = select_action(["-b", "-c"], sl, rl_q_values=q, rl_best="-a")
    assert picked == "-b"
    # Agent's pick still available: honor it.
    picked = select_action(["-a", "-b"], sl, rl_q_values=q, rl_best="-b")
    assert picked == "-b"


def test_select_action_stop_sentinel():
    sl = [("-gvn", 1.0, 1.0)]
    assert select_action(["-gvn"], sl, stop_prior=2.0) == STOP_FLAG
    assert select_action(["-gvn"], sl, stop_prior=0.5) == "-gvn"


def test_select_action_learned_rl_stop():
    sl = [("-gvn", 1.0, 0.4), ("-licm", 0.5, 0.3)]
    available = ["-gvn", "-licm"]
    # Agent's learned Q(STOP) beats every available pass -> stop.
    q_stop_high = {"-gvn": 0.2, "-licm": -0.4, STOP_FLAG: 0.5}
    assert select_action(available, sl, rl_q_values=q_stop_high, rl_best="-gvn") == STOP_FLAG
    # Agent's learned Q(STOP) is worse than the best pass -> keep optimizing.
    q_stop_low = {"-gvn": 3.0, "-licm": 0.1, STOP_FLAG: 0.0}
    assert select_action(available, sl, rl_q_values=q_stop_low, rl_best="-gvn") == "-gvn"
    # STOP is selected even when the agent's own pass pick is masked.
    q_masked_pick = {"-a": 9.0, "-gvn": 0.1, "-licm": 0.0, STOP_FLAG: 5.0}
    assert select_action(["-gvn", "-licm"], sl, rl_q_values=q_masked_pick, rl_best="-a") == STOP_FLAG
    # Without STOP in the Q dict, learned STOP is inert (SL fallback).
    q_no_stop = {"-gvn": 2.0, "-licm": 1.0}
    assert select_action(available, sl, rl_q_values=q_no_stop, rl_best="-gvn") == "-gvn"


def test_select_action_is_deterministic_by_default():
    sl = [("-a", 5.0, 1.0), ("-b", 3.0, 0.0)]
    for _ in range(5):
        assert select_action(["-a", "-b"], sl) == "-a"


def test_select_action_seeded_explore_is_reproducible():
    import random
    sl = [("-a", 5.0, 1.0), ("-b", 3.0, 0.0), ("-c", 1.0, 0.0)]
    first = select_action(["-a", "-b", "-c"], sl, explore=True, rng=random.Random(42))
    again = select_action(["-a", "-b", "-c"], sl, explore=True, rng=random.Random(42))
    assert first == again


def test_noop_action_is_masked_and_episode_terminates():
    """Acceptance criterion from the review: a selected action is never
    repeated in an unchanged state, and with no_op_limit=1 the episode stops
    after the first no-op instead of burning the pass budget."""
    available = ["-gvn", "-licm", "-sroa"]
    sl_ranked = [("-gvn", 5.0, 0.6), ("-licm", 3.0, 0.3), ("-sroa", 1.0, 0.1)]
    tried: set = set()
    consecutive_no_op = 0
    no_op_limit = 1
    steps = []
    for _ in range(5):  # bounded emulation of the inference loop
        candidates = [flag for flag in available if flag not in tried]
        if not candidates:
            break
        chosen = select_action(candidates, sl_ranked)
        steps.append(chosen)
        no_op = chosen == "-gvn"  # only -gvn has no effect here
        if no_op:
            consecutive_no_op += 1
            tried.add(chosen)
            if consecutive_no_op >= no_op_limit:
                break
        else:
            consecutive_no_op = 0
    # The episode terminates after the single no-op action; it is masked.
    assert steps == ["-gvn"]
    assert "-gvn" not in [flag for flag in available if flag not in tried]


def test_episode_rng_is_resume_reproducible():
    flags = ["-gvn", "-licm", "-sroa"]
    rng_a = episode_rng_for("benchmark://cbench-v1/qsort", 3, 42, flags, 8)
    rng_b = episode_rng_for("benchmark://cbench-v1/qsort", 3, 42, flags, 8)
    # Same episode identity -> identical stream (resume reproduces content).
    assert [rng_a.choice(flags) for _ in range(10)] == [
        rng_b.choice(flags) for _ in range(10)
    ]


def test_episode_rng_differs_across_episodes():
    flags = ["-gvn", "-licm", "-sroa", "-dce", "-instcombine"]
    rng_a = episode_rng_for("benchmark://cbench-v1/qsort", 3, 42, flags, 8)
    rng_b = episode_rng_for("benchmark://cbench-v1/qsort", 4, 42, flags, 8)
    seq_a = [rng_a.choice(flags) for _ in range(20)]
    seq_b = [rng_b.choice(flags) for _ in range(20)]
    assert seq_a != seq_b


def test_episode_id_is_stable_and_deterministic():
    flags = ["-gvn"]
    assert episode_id_for("benchmark://cbench-v1/qsort", 0, 42, flags, 8) == episode_id_for(
        "benchmark://cbench-v1/qsort", 0, 42, flags, 8
    )
    assert len(episode_id_for("benchmark://cbench-v1/qsort", 0, 42, flags, 8)) == 24
