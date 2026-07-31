#!/usr/bin/env python3
"""Curated LLVM pass set for NeuroCompiler.

LLVM has 100+ passes. We use 27 that actually mutate IR, grouped as:
- Scalar
- Loop
- Interprocedural
- Memory
- Miscellaneous

This file is the single source of truth for Phase 2 pass selection.
"""

from __future__ import annotations

from typing import List, Tuple

# (name, flag, description, category)
CURATED_PASSES: List[Tuple[str, str, str, str]] = [
    # Scalar
    ("adce", "-adce", "Aggressive DCE", "scalar"),
    ("dce", "-dce", "Dead Code Elimination", "scalar"),
    ("early-cse", "-early-cse", "Early CSE", "scalar"),
    ("gvn", "-gvn", "Global Value Numbering", "scalar"),
    ("newgvn", "-newgvn", "New GVN", "scalar"),
    ("instcombine", "-instcombine", "InstCombine", "scalar"),
    ("aggressive-instcombine", "-aggressive-instcombine", "Aggressive InstCombine", "scalar"),
    ("sroa", "-sroa", "Scalar Replacement Of Aggregates", "scalar"),
    ("reassociate", "-reassociate", "Reassociate", "scalar"),
    ("simplifycfg", "-simplifycfg", "Simplify CFG", "scalar"),
    ("constmerge", "-constmerge", "Constant Merge", "scalar"),
    ("correlated-propagation", "-correlated-propagation", "Correlated Propagation", "scalar"),
    # Loop
    ("licm", "-licm", "Loop Invariant Code Motion", "loop"),
    ("loop-rotate", "-loop-rotate", "Loop Rotate", "loop"),
    ("loop-unroll", "-loop-unroll", "Loop Unroll", "loop"),
    ("loop-vectorize", "-loop-vectorize", "Loop Vectorize", "loop"),
    ("loop-deletion", "-loop-deletion", "Loop Deletion", "loop"),
    ("loop-unswitch", "-loop-unswitch", "Loop Unswitch", "loop"),
    ("loop-distribute", "-loop-distribute", "Loop Distribute", "loop"),
    ("indvars", "-indvars", "Induction Variable Simplification", "loop"),
    # Interprocedural
    ("inline", "-inline", "Function Inlining", "ipo"),
    ("partial-inliner", "-partial-inliner", "Partial Inliner", "ipo"),
    ("deadargelim", "-deadargelim", "Dead Argument Elimination", "ipo"),
    ("argpromotion", "-argpromotion", "Argument Promotion", "ipo"),
    ("globalopt", "-globalopt", "Global Optimizer", "ipo"),
    ("globaldce", "-globaldce", "Global DCE", "ipo"),
    ("functionattrs", "-functionattrs", "Function Attrs", "ipo"),
    # Memory
    ("dse", "-dse", "Dead Store Elimination", "memory"),
    ("memcpyopt", "-memcpyopt", "Memcpy Optimization", "memory"),
    # Misc
    ("jump-threading", "-jump-threading", "Jump Threading", "misc"),
    ("tailcallelim", "-tailcallelim", "Tail Call Elimination", "misc"),
]

CURATED_FLAGS = [flag for _, flag, _, _ in CURATED_PASSES]
CURATED_NAMES = [name for name, _, _, _ in CURATED_PASSES]

def get_curated_flags() -> List[str]:
    return list(CURATED_FLAGS)

def get_curated_names() -> List[str]:
    return list(CURATED_NAMES)

def describe() -> str:
    lines = ["Recommended 27-30 passes by category:"]
    by_cat = {}
    for name, flag, desc, cat in CURATED_PASSES:
        by_cat.setdefault(cat, []).append(f"{flag:<25} {desc}")
    for cat in ["scalar", "loop", "ipo", "memory", "misc"]:
        lines.append(f"\n{cat.upper()}:")
        lines.extend(f"  {x}" for x in by_cat.get(cat, []))
    return "\n".join(lines)

if __name__ == "__main__":
    print(describe())
    print(f"\nTotal: {len(CURATED_PASSES)} passes")
