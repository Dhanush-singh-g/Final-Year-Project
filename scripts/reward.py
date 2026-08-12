#!/usr/bin/env python3
"""
Reward computation for NeuroCompiler.

Phase 5 reward combines multiple objectives:

Reward = 0.6 * Runtime Improvement + 0.3 * IR Reduction + 0.1 * Code Size Reduction

When runtime is unavailable (cBench many non-runnable), falls back to IR + Object size.

This is used by both SL dataset (for labeling immediate quality) and RL replay buffer (long-term).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RewardWeights:
    runtime: float = 0.6
    ir: float = 0.3
    code_size: float = 0.1

    def __post_init__(self):
        total = self.runtime + self.ir + self.code_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

DEFAULT_WEIGHTS = RewardWeights()

def safe_ratio(delta: float, base: float) -> float:
    if base is None or base == 0:
        return 0.0
    return delta / base

def compute_ir_improvement(pre_ir: Optional[int], post_ir: Optional[int]) -> float:
    """Positive if IR shrank."""
    if pre_ir is None or post_ir is None or pre_ir == 0:
        return 0.0
    # (pre - post)/pre  -> positive means reduced instructions
    return (pre_ir - post_ir) / pre_ir

def compute_size_improvement(pre_size: Optional[int], post_size: Optional[int]) -> float:
    if pre_size is None or post_size is None or pre_size == 0:
        return 0.0
    return (pre_size - post_size) / pre_size

def compute_runtime_improvement(pre_rt: Optional[float], post_rt: Optional[float]) -> float:
    """Positive if runtime decreased."""
    if pre_rt is None or post_rt is None or pre_rt <= 0:
        return 0.0
    return (pre_rt - post_rt) / pre_rt

def compute_hybrid_reward(
    pre_ir: Optional[int],
    post_ir: Optional[int],
    pre_size: Optional[int] = None,
    post_size: Optional[int] = None,
    pre_runtime: Optional[float] = None,
    post_runtime: Optional[float] = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> dict:
    """
    Returns dict with breakdown and final weighted reward.
    Final reward * 100 to be in percentage scale, matches CompilerGym reward magnitude.
    """
    ir_imp = compute_ir_improvement(pre_ir, post_ir)
    size_imp = compute_size_improvement(pre_size, post_size)
    rt_imp = compute_runtime_improvement(pre_runtime, post_runtime)

    # If runtime is not available, renormalize weights to IR + size only
    if pre_runtime is None or post_runtime is None:
        # Distribute runtime weight proportionally to IR and size
        if weights.runtime >= 1.0:
            effective_ir_w = 0.75
            effective_size_w = 0.25
            effective_rt_w = 0.0
        else:
            remaining = 1.0 - weights.runtime
            # Keep ratio between ir and size
            ir_ratio = weights.ir / (weights.ir + weights.code_size) if (weights.ir + weights.code_size) > 0 else 0.75
            size_ratio = 1.0 - ir_ratio
            effective_ir_w = weights.ir + weights.runtime * ir_ratio
            effective_size_w = weights.code_size + weights.runtime * size_ratio
            effective_rt_w = 0.0
    else:
        effective_ir_w = weights.ir
        effective_size_w = weights.code_size
        effective_rt_w = weights.runtime

    hybrid = (
        effective_rt_w * rt_imp
        + effective_ir_w * ir_imp
        + effective_size_w * size_imp
    )

    # Scale to ~ reward magnitude used in training (x100)
    return {
        "runtime_improvement": rt_imp,
        "ir_improvement": ir_imp,
        "size_improvement": size_imp,
        "hybrid_reward": hybrid,
        "hybrid_reward_scaled": hybrid * 100.0,
        "weights_used": {
            "runtime": effective_rt_w,
            "ir": effective_ir_w,
            "size": effective_size_w,
        },
    }

