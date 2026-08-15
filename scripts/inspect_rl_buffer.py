#!/usr/bin/env python3
"""Phase 12 — inspect a multi-step RL replay buffer.

Reports the health statistics of a ``generate_rl_episodes.py`` buffer and
PROVES the transitions are genuinely chained:

    for every episode: post_state_id(row t) == pre_state_id(row t+1)

Also surfaces how many transitions are non-terminal (done=False), how many
episodes really ended by STOP, the start-state split (O0 vs deeper), action
and reward distributions, and prints concrete trajectory examples with state
identifiers so the chaining is visible.

Usage:
  python scripts/inspect_rl_buffer.py --input datasets/replay_buffer/rl_experiences_multistep_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STOP_FLAG = "-stop"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def episode_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        out[r.get("episode_id", "")].append(r)
    for ep in out.values():
        ep.sort(key=lambda r: int(r.get("step_index", 0) or 0))
    return out


def verify_episode_chaining(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return a list of chaining violations (empty == perfectly chained).

    Within each episode, consecutive transitions must satisfy
    ``post_state_id(t) == pre_state_id(t+1)`` (STOP rows are always last and
    carry no next state, so they are exempt).
    """
    violations: List[str] = []
    episodes = episode_rows(rows)
    for eid, ep in episodes.items():
        for i in range(len(ep) - 1):
            cur, nxt = ep[i], ep[i + 1]
            if cur.get("pass_flag") == STOP_FLAG:
                continue
            if cur.get("post_state_id") != nxt.get("pre_state_id"):
                violations.append(
                    f"{eid}: row {i} post={cur.get('post_state_id')} "
                    f"!= row {i+1} pre={nxt.get('pre_state_id')}"
                )
    return violations


def buffer_report(rows: Sequence[Dict[str, str]]) -> Dict:
    episodes = episode_rows(rows)
    lengths = [len(ep) for ep in episodes.values()]
    done = Counter(r.get("done", "").lower() in ("true", "1", "yes") for r in rows)
    stop_rows = [r for r in rows if r.get("pass_flag") == STOP_FLAG]
    stops_by_reason = Counter(
        ep[-1].get("terminal_reason", "") for ep in episodes.values()
    )
    starts = Counter(
        ep[0].get("start_state", "o0") for ep in episodes.values()
    )
    actions = Counter(r.get("pass_flag", "") for r in rows)
    noop = Counter(r.get("no_op", "").lower() in ("true", "1", "yes") for r in rows)
    self_loops = sum(
        1 for r in rows
        if r.get("post_state_id") and r.get("post_state_id") == r.get("pre_state_id")
    )
    rewards = [
        float(r["hybrid_reward"]) for r in rows if (r.get("hybrid_reward") or "").strip()
    ]
    per_benchmark = Counter(r.get("benchmark_uri", "") for r in rows)
    unique_states = len({r.get("pre_state_id") for r in rows if r.get("pre_state_id")})
    violations = verify_episode_chaining(rows)

    return {
        "episodes": len(episodes),
        "transitions": len(rows),
        "avg_episode_length": (sum(lengths) / len(lengths)) if lengths else None,
        "max_episode_length": max(lengths) if lengths else None,
        "done_true_fraction": done.get(True, 0) / len(rows) if rows else None,
        "done_false_fraction": done.get(False, 0) / len(rows) if rows else None,
        "genuine_stop_transitions": len(stop_rows),
        "terminal_reasons": dict(stops_by_reason),
        "start_state_distribution": dict(starts),
        "action_distribution": dict(actions),
        "no_op_fraction": noop.get(True, 0) / len(rows) if rows else None,
        "self_loop_transitions": self_loops,
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "unique_pre_states": unique_states,
        "unique_programs": len(per_benchmark),
        "transitions_per_benchmark": dict(per_benchmark),
        "chaining_violations": len(violations),
        "chaining_violation_examples": violations[:5],
    }


def format_trajectories(rows: Sequence[Dict[str, str]], n: int = 3) -> str:
    """Render a few episodes as S0 -> A -> S1 -> ... chains with state ids."""
    episodes = episode_rows(rows)
    lines = []
    for eid in sorted(episodes)[:n]:
        ep = episodes[eid]
        uri = ep[0].get("benchmark_uri", "")
        reason = ep[-1].get("terminal_reason", "")
        start = ep[0].get("start_state", "")
        chain = []
        for i, r in enumerate(ep):
            sid = (r.get("pre_state_id") or "?")[:10]
            flag = r.get("pass_flag", "")
            chain.append(f"S{i}({sid}) --{flag}-->")
            if i == len(ep) - 1:
                chain.append(f"S{i+1}({(r.get('post_state_id') or '?')[:10]})")
        lines.append(
            f"  {uri.split('/')[-1]} [{start}, end={reason}]: "
            + " ".join(chain)
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--trajectories", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    report = buffer_report(rows)

    print("=== Phase 12 RL buffer report ===")
    for key in (
        "episodes", "transitions", "avg_episode_length", "max_episode_length",
        "done_true_fraction", "done_false_fraction", "genuine_stop_transitions",
        "no_op_fraction", "self_loop_transitions",
        "unique_pre_states", "unique_programs",
    ):
        print(f"  {key}: {report[key]}")
    print("  terminal_reasons:", report["terminal_reasons"])
    print("  start_state_distribution:", report["start_state_distribution"])
    print("  reward mean/min/max:", report["reward_mean"], report["reward_min"], report["reward_max"])
    print("  action_distribution:", report["action_distribution"])
    print("  chaining_violations:", report["chaining_violations"])
    if report["chaining_violation_examples"]:
        print("  violation examples:", report["chaining_violation_examples"][:3])

    print("\n=== Trajectory examples (state ids truncated to 10 chars) ===")
    print(format_trajectories(rows, args.trajectories))

    print("\n=== Per-benchmark transitions ===")
    for uri, count in sorted(report["transitions_per_benchmark"].items()):
        print(f"  {uri}: {count}")

    ok = report["chaining_violations"] == 0
    print(f"\nCHAINING: {'PASS' if ok else 'FAIL'} ({report['chaining_violations']} violations)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
