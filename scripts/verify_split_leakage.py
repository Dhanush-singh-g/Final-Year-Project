#!/usr/bin/env python3
"""Program-level leakage verification for NeuroCompiler training buffers.

Phase 2/4 requirement: the FINAL TEST programs must never appear in training,
at any granularity (feature rows, states, transitions, rewards). This script
proves, programmatically:

1. No evaluation-program URI appears in the buffer (full-URI identity).
2. No row carries an evaluation benchmark_uri.
3. Near-duplicate slugs across families are reported (e.g. cBench ``sha`` vs
   CHStone ``sha`` are DIFFERENT programs with different URIs — the checks
   must use full URIs, never slugs).
4. Duplicate (benchmark, pre_state_id, pass_flag) rows are reported
   (redundant measurements of the same transition).

Exit code 0 = all checks pass.

Usage:
  python scripts/verify_split_leakage.py \
      --buffer datasets/replay_buffer/rl_experiences_multistep_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# The 8 large-input cBench FINAL TEST programs (held out of ALL training).
EVAL_BENCHMARKS = [
    "benchmark://cbench-v1/bitcount",
    "benchmark://cbench-v1/bzip2",
    "benchmark://cbench-v1/dijkstra",
    "benchmark://cbench-v1/gsm",
    "benchmark://cbench-v1/jpeg-c",
    "benchmark://cbench-v1/stringsearch",
    "benchmark://cbench-v1/tiff2bw",
    "benchmark://cbench-v1/tiff2rgba",
]


def verify_no_eval_leakage(
    rows: Sequence[Dict[str, str]],
    eval_uris: Sequence[str] = EVAL_BENCHMARKS,
) -> Tuple[List[str], Dict[str, int]]:
    """Return (violations, per-benchmark row counts)."""
    eval_set = set(eval_uris)
    violations: List[str] = []
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        uri = r.get("benchmark_uri", "")
        counts[uri] += 1
        if uri in eval_set:
            violations.append(f"eval program row in training buffer: {uri}")
    return violations, dict(counts)


def slug_collisions(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Slugs shared across families (informational: different URIs = different
    programs, but flag them so a human confirms no accidental merge)."""
    slug_families: Dict[str, set] = defaultdict(set)
    for r in rows:
        uri = r.get("benchmark_uri", "")
        if not uri:
            continue
        suite = uri.split("://")[1].split("/")[0]
        slug = uri.split("/")[-1]
        slug_families[slug].add(suite)
    return [
        f"slug '{slug}' appears in families {sorted(fams)}"
        for slug, fams in sorted(slug_families.items())
        if len(fams) > 1
    ]


def duplicate_transitions(rows: Sequence[Dict[str, str]]) -> List[str]:
    dup = Counter(
        (r.get("benchmark_uri", ""), r.get("pre_state_id", ""), r.get("pass_flag", ""))
        for r in rows
    )
    return [
        f"{k[0]} state {k[1][:10]} pass {k[2]} measured {n}x"
        for k, n in dup.items()
        if n > 1
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--eval-uris", default=None,
                        help="Comma-separated eval URIs (default: the 8 cBench test programs)")
    args = parser.parse_args()

    with Path(args.buffer).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    eval_uris = (
        [u.strip() for u in args.eval_uris.split(",") if u.strip()]
        if args.eval_uris else EVAL_BENCHMARKS
    )

    violations, counts = verify_no_eval_leakage(rows, eval_uris)
    collisions = slug_collisions(rows)
    dups = duplicate_transitions(rows)

    print(f"Buffer: {len(rows)} rows, {len(counts)} programs")
    print("Programs:", sorted(counts))
    if collisions:
        print("\nSlug collisions across families (informational):")
        for c in collisions:
            print("  ", c)
    if dups:
        print(f"\nDuplicate (program, state, pass) measurements ({len(dups)}):")
        for d in dups[:10]:
            print("  ", d)
    print(f"\nEval-leakage violations: {len(violations)}")
    for v in violations[:10]:
        print("  ", v)

    ok = not violations
    print(f"LEAKAGE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
