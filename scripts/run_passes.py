#!/usr/bin/env python3
"""Apply LLVM optimization passes and record state transitions.

Stage 2 of the NeuroCompiler dataset pipeline. Every Transition represents:
    (pre-state, action, step reward, post-state)
The module supports both independent single-pass experiments and ordered pass
sequences. Environment reset policy remains the caller's responsibility.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # Supports both `python scripts/x.py` and package imports.
    from .extract_features import (
        AUTOPHASE_FEATURE_NAMES,
        MeasurementConfig,
        ProgramFeatures,
        extract_features,
    )
except ImportError:
    from extract_features import (  # type: ignore
        AUTOPHASE_FEATURE_NAMES,
        MeasurementConfig,
        ProgramFeatures,
        extract_features,
    )

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionMetadata:
    """Stable metadata for one LLVM action-space entry."""

    action_id: int
    name: str
    flag: str
    description: str


@dataclass(frozen=True)
class Transition:
    """Result of applying one action to one LLVM program state."""

    benchmark_uri: str
    action: ActionMetadata
    reward_space: str
    pass_position: int
    previous_pass_sequence: Tuple[int, ...]
    pre: ProgramFeatures
    post: Optional[ProgramFeatures]
    step_reward: Optional[float]
    cumulative_reward: Optional[float]
    pass_success: bool
    action_had_no_effect: Optional[bool]
    done: bool
    step_walltime_sec: float
    error_type: str = ""
    error_message: str = ""

    def to_row(self) -> Dict[str, Any]:
        """Flatten the transition into a deterministic CSV-compatible row."""

        row: Dict[str, Any] = {
            "schema_version": self.pre.schema_version,
            "benchmark_uri": self.benchmark_uri,
            "pass_id": self.action.action_id,
            "pass_name": self.action.name,
            "pass_flag": self.action.flag,
            "pass_description": self.action.description,
            "pass_position": self.pass_position,
            "previous_pass_sequence": json.dumps(self.previous_pass_sequence),
            "reward_space": self.reward_space,
            "step_reward": self.step_reward,
            "cumulative_reward": self.cumulative_reward,
            "pass_success": self.pass_success,
            "action_had_no_effect": self.action_had_no_effect,
            "done": self.done,
            "step_walltime_sec": self.step_walltime_sec,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        row.update(_state_row(self.pre, "pre_"))
        if self.post is not None:
            row.update(_state_row(self.post, "post_"))
            row.update(_delta_row(self.pre, self.post))
        return row


def action_catalog(env: Any) -> List[ActionMetadata]:
    """Return all actions exposed by the active LLVM Commandline space."""

    space = env.action_space
    count = int(space.n)
    names = list(getattr(space, "names", []))
    flags = list(getattr(space, "flags", []))
    descriptions = list(getattr(space, "descriptions", []))
    if not (len(names) == len(flags) == len(descriptions) == count):
        raise RuntimeError(
            "LLVM action-space metadata is incomplete: "
            f"n={count}, names={len(names)}, flags={len(flags)}, "
            f"descriptions={len(descriptions)}"
        )
    return [
        ActionMetadata(i, str(names[i]), str(flags[i]), str(descriptions[i]))
        for i in range(count)
    ]


def resolve_actions(
    env: Any,
    requested: Optional[Sequence[Union[int, str]]] = None,
) -> List[ActionMetadata]:
    """Resolve IDs, action names, or command-line flags to action metadata."""

    catalog = action_catalog(env)
    if not requested:
        return catalog
    by_id = {item.action_id: item for item in catalog}
    by_text: Dict[str, ActionMetadata] = {}
    for item in catalog:
        by_text[item.name] = item
        by_text[item.flag] = item

    resolved: List[ActionMetadata] = []
    seen = set()
    for value in requested:
        item: Optional[ActionMetadata]
        if isinstance(value, int) or str(value).strip().isdigit():
            item = by_id.get(int(value))
        else:
            item = by_text.get(str(value).strip())
        if item is None:
            raise ValueError(f"Unknown LLVM action: {value!r}")
        if item.action_id not in seen:
            resolved.append(item)
            seen.add(item.action_id)
    return resolved


def run_pass_sequence(
    env: Any,
    actions: Sequence[ActionMetadata],
    *,
    reward_space: str,
    measurement: Optional[MeasurementConfig] = None,
    initial_features: Optional[ProgramFeatures] = None,
    timeout_sec: float = 300.0,
) -> List[Transition]:
    """Apply an ordered pass sequence and return one transition per action.

    The environment must already be reset with the requested benchmark and
    reward space. For independent single-pass experiments, reset before each
    call and pass a one-element action sequence.
    """

    if not getattr(env, "in_episode", False):
        raise RuntimeError("Environment must be reset before running passes")
    if not actions:
        return []

    config = measurement or MeasurementConfig()
    pre = initial_features or extract_features(env, config)
    previous: List[int] = []
    transitions: List[Transition] = []

    for position, action in enumerate(actions):
        started = time.perf_counter()
        post: Optional[ProgramFeatures] = None
        reward: Optional[float] = None
        done = False
        info: Dict[str, Any] = {}
        error_type = ""
        error_message = ""
        success = False

        try:
            _, raw_reward, done, raw_info = env.step(
                action.action_id,
                timeout=timeout_sec,
            )
            info = dict(raw_info or {})
            if raw_reward is not None:
                reward = float(raw_reward)
                if not math.isfinite(reward):
                    raise ValueError(f"Non-finite reward returned: {reward!r}")
            # A terminal step may still expose a valid final state. Try to read
            # it; if the service terminated with an error, extraction is caught.
            post = extract_features(env, config)
            success = not bool(info.get("error_details"))
        except Exception as error:  # Preserve failure as a dataset audit row.
            error_type = type(error).__name__
            error_message = str(error).replace("\n", " ")[:2000]
            LOGGER.warning(
                "Pass failed benchmark=%s action=%s: %s",
                pre.benchmark_uri,
                action.flag,
                error_message,
            )

        elapsed = time.perf_counter() - started
        cumulative = getattr(env, "episode_reward", None)
        if cumulative is not None:
            try:
                cumulative = float(cumulative)
            except (TypeError, ValueError):
                cumulative = None

        transition = Transition(
            benchmark_uri=pre.benchmark_uri,
            action=action,
            reward_space=reward_space,
            pass_position=position,
            previous_pass_sequence=tuple(previous),
            pre=pre,
            post=post,
            step_reward=reward,
            cumulative_reward=cumulative,
            pass_success=success,
            action_had_no_effect=(
                bool(info["action_had_no_effect"])
                if "action_had_no_effect" in info
                else None
            ),
            done=bool(done),
            step_walltime_sec=elapsed,
            error_type=error_type,
            error_message=error_message,
        )
        transitions.append(transition)

        if not success or post is None or done:
            break
        previous.append(action.action_id)
        pre = post

    return transitions


def _state_row(features: ProgramFeatures, prefix: str) -> Dict[str, Any]:
    row = features.flattened(prefix)
    row.pop("schema_version", None)
    row.pop("benchmark_uri", None)
    return row


def _delta_optional(pre: Optional[int], post: Optional[int]) -> Optional[int]:
    if pre is None or post is None:
        return None
    return post - pre


def _delta_row(pre: ProgramFeatures, post: ProgramFeatures) -> Dict[str, Any]:
    pre_runtime = pre.runtime_median_sec
    post_runtime = post.runtime_median_sec
    runtime_reduction = (
        pre_runtime - post_runtime
        if pre_runtime is not None and post_runtime is not None
        else None
    )
    runtime_speedup = (
        pre_runtime / post_runtime
        if pre_runtime is not None
        and post_runtime is not None
        and post_runtime > 0
        else None
    )
    runtime_improvement_pct = (
        100.0 * runtime_reduction / pre_runtime
        if runtime_reduction is not None and pre_runtime and pre_runtime > 0
        else None
    )
    row: Dict[str, Any] = {
        # Positive runtime reduction/improvement means the pass was faster.
        "runtime_reduction_sec": runtime_reduction,
        "runtime_speedup": runtime_speedup,
        "runtime_improvement_pct": runtime_improvement_pct,
        "delta_ir_instruction_count": (
            post.ir_instruction_count - pre.ir_instruction_count
        ),
        "delta_object_text_size_bytes": _delta_optional(
            pre.object_text_size_bytes, post.object_text_size_bytes
        ),
        "delta_total_basic_blocks": post.total_basic_blocks - pre.total_basic_blocks,
        "delta_total_functions": post.total_functions - pre.total_functions,
        "delta_total_instructions": post.total_instructions - pre.total_instructions,
        "delta_total_memory_instructions": (
            post.total_memory_instructions - pre.total_memory_instructions
        ),
    }
    for name in AUTOPHASE_FEATURE_NAMES:
        row[f"delta_autophase_{name}"] = post.autophase[name] - pre.autophase[name]
    return row


def state_fieldnames(prefix: str) -> List[str]:
    """Return the exact field order emitted for one flattened state."""

    return [
        f"{prefix}state_id",
        f"{prefix}ir_instruction_count",
        f"{prefix}object_text_size_bytes",
        f"{prefix}total_basic_blocks",
        f"{prefix}total_functions",
        f"{prefix}total_instructions",
        f"{prefix}total_memory_instructions",
        f"{prefix}is_runnable",
        f"{prefix}is_buildable",
        f"{prefix}runtime_measurement_count",
        f"{prefix}runtime_median_sec",
        f"{prefix}runtime_mean_sec",
        f"{prefix}runtime_std_sec",
        f"{prefix}runtime_samples_json",
        f"{prefix}buildtime_sec",
    ] + [f"{prefix}autophase_{name}" for name in AUTOPHASE_FEATURE_NAMES]


def delta_fieldnames() -> List[str]:
    return [
        "runtime_reduction_sec",
        "runtime_speedup",
        "runtime_improvement_pct",
        "delta_ir_instruction_count",
        "delta_object_text_size_bytes",
        "delta_total_basic_blocks",
        "delta_total_functions",
        "delta_total_instructions",
        "delta_total_memory_instructions",
    ] + [f"delta_autophase_{name}" for name in AUTOPHASE_FEATURE_NAMES]


def transition_fieldnames() -> List[str]:
    return [
        "schema_version",
        "benchmark_uri",
        "pass_id",
        "pass_name",
        "pass_flag",
        "pass_description",
        "pass_position",
        "previous_pass_sequence",
        "reward_space",
        "step_reward",
        "cumulative_reward",
        "pass_success",
        "action_had_no_effect",
        "done",
        "step_walltime_sec",
        "error_type",
        "error_message",
    ] + state_fieldnames("pre_") + state_fieldnames("post_") + delta_fieldnames()
