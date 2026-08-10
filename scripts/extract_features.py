#!/usr/bin/env python3
"""Extract reproducible LLVM program-state features from CompilerGym.

This module is Stage 1 of the NeuroCompiler dataset pipeline. Its primary API,
:func:`extract_features`, reads a snapshot of the *current* LLVM environment
state. It does not reset the environment or apply optimization passes; callers
therefore control whether the snapshot represents a pre-action or post-action
state.

CompilerGym's Runtime and Buildtime observations are experimental and costly.
They are disabled by default and must be requested through MeasurementConfig.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# CompilerGym 0.2.5 Autophase order. Keeping the order explicit makes CSV
# schemas deterministic and lets us reject incompatible environment responses.
AUTOPHASE_FEATURE_NAMES: Tuple[str, ...] = (
    "BBNumArgsHi",
    "BBNumArgsLo",
    "onePred",
    "onePredOneSuc",
    "onePredTwoSuc",
    "oneSuccessor",
    "twoPred",
    "twoPredOneSuc",
    "twoEach",
    "twoSuccessor",
    "morePreds",
    "BB03Phi",
    "BBHiPhi",
    "BBNoPhi",
    "BeginPhi",
    "BranchCount",
    "returnInt",
    "CriticalCount",
    "NumEdges",
    "const32Bit",
    "const64Bit",
    "numConstZeroes",
    "numConstOnes",
    "UncondBranches",
    "binaryConstArg",
    "NumAShrInst",
    "NumAddInst",
    "NumAllocaInst",
    "NumAndInst",
    "BlockMid",
    "BlockLow",
    "NumBitCastInst",
    "NumBrInst",
    "NumCallInst",
    "NumGetElementPtrInst",
    "NumICmpInst",
    "NumLShrInst",
    "NumLoadInst",
    "NumMulInst",
    "NumOrInst",
    "NumPHIInst",
    "NumRetInst",
    "NumSExtInst",
    "NumSelectInst",
    "NumShlInst",
    "NumStoreInst",
    "NumSubInst",
    "NumTruncInst",
    "NumXorInst",
    "NumZExtInst",
    "TotalBlocks",
    "TotalInsts",
    "TotalMemInst",
    "TotalFuncs",
    "ArgsPhi",
    "testUnary",
)

if len(AUTOPHASE_FEATURE_NAMES) != 56:  # Defensive invariant.
    raise RuntimeError("The Autophase schema must contain exactly 56 features")


class FeatureExtractionError(RuntimeError):
    """Raised when a required program-state observation cannot be extracted."""


@dataclass(frozen=True)
class MeasurementConfig:
    """Controls optional, platform-dependent measurements.

    Args:
        measure_runtime: Query the Runtime observation if the benchmark is
            runnable.
        runtime_count: Number of timed benchmark executions.
        runtime_warmup_count: Number of untimed warm-up executions.
        measure_buildtime: Query the Buildtime observation if buildable.
        collect_object_text_size: Query the lowered object .TEXT size. This is
            more expensive and platform-dependent, but deterministic per host.
    """

    measure_runtime: bool = False
    runtime_count: int = 5
    runtime_warmup_count: int = 1
    measure_buildtime: bool = False
    collect_object_text_size: bool = True

    def __post_init__(self) -> None:
        if self.runtime_count < 1:
            raise ValueError("runtime_count must be at least 1")
        if self.runtime_warmup_count < 0:
            raise ValueError("runtime_warmup_count cannot be negative")


@dataclass(frozen=True)
class ProgramFeatures:
    """A snapshot of one LLVM program state.

    Autophase values describe the current state. Stage 2 will label snapshots
    as pre- or post-action when flattening transitions into dataset rows.
    """

    schema_version: str
    benchmark_uri: str
    state_id: str
    autophase: Mapping[str, int]
    ir_instruction_count: int
    object_text_size_bytes: Optional[int]
    total_basic_blocks: int
    total_functions: int
    total_instructions: int
    total_memory_instructions: int
    is_runnable: bool
    is_buildable: bool
    runtime_samples_sec: Tuple[float, ...] = ()
    runtime_median_sec: Optional[float] = None
    runtime_mean_sec: Optional[float] = None
    runtime_std_sec: Optional[float] = None
    buildtime_sec: Optional[float] = None

    def flattened(self, prefix: str = "") -> Dict[str, Any]:
        """Return a deterministic, CSV-compatible dictionary.

        Args:
            prefix: Prefix for state-dependent fields, normally ``pre_`` or
                ``post_``. Identity fields remain unprefixed.
        """

        row: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "benchmark_uri": self.benchmark_uri,
            f"{prefix}state_id": self.state_id,
            f"{prefix}ir_instruction_count": self.ir_instruction_count,
            f"{prefix}object_text_size_bytes": self.object_text_size_bytes,
            f"{prefix}total_basic_blocks": self.total_basic_blocks,
            f"{prefix}total_functions": self.total_functions,
            f"{prefix}total_instructions": self.total_instructions,
            f"{prefix}total_memory_instructions": self.total_memory_instructions,
            f"{prefix}is_runnable": self.is_runnable,
            f"{prefix}is_buildable": self.is_buildable,
            f"{prefix}runtime_measurement_count": len(self.runtime_samples_sec),
            f"{prefix}runtime_median_sec": self.runtime_median_sec,
            f"{prefix}runtime_mean_sec": self.runtime_mean_sec,
            f"{prefix}runtime_std_sec": self.runtime_std_sec,
            f"{prefix}runtime_samples_json": json.dumps(self.runtime_samples_sec),
            f"{prefix}buildtime_sec": self.buildtime_sec,
        }
        for name in AUTOPHASE_FEATURE_NAMES:
            row[f"{prefix}autophase_{name}"] = self.autophase[name]
        return row

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for diagnostics."""

        result = asdict(self)
        result["runtime_samples_sec"] = list(self.runtime_samples_sec)
        return result


def _observation(env: Any, name: str, *, required: bool) -> Any:
    """Read one named observation with consistent error handling."""

    try:
        return env.observation[name]
    except Exception as error:  # CompilerGym service exceptions vary by failure.
        if required:
            raise FeatureExtractionError(
                f"Required observation {name!r} failed: {error}"
            ) from error
        LOGGER.warning("Optional observation %s failed: %s", name, error)
        return None


def _scalar_int(value: Any, observation_name: str) -> int:
    """Convert a scalar or one-element CompilerGym array to int."""

    if value is None:
        raise FeatureExtractionError(f"Observation {observation_name!r} is missing")
    if isinstance(value, (str, bytes)):
        raise FeatureExtractionError(
            f"Observation {observation_name!r} is not numeric: {value!r}"
        )
    try:
        if hasattr(value, "reshape"):
            flat = value.reshape(-1)
            if len(flat) != 1:
                raise ValueError(f"expected one value, got {len(flat)}")
            return int(flat[0])
        if isinstance(value, Sequence):
            if len(value) != 1:
                raise ValueError(f"expected one value, got {len(value)}")
            return int(value[0])
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureExtractionError(
            f"Invalid scalar observation {observation_name!r}: {value!r}"
        ) from error


def _benchmark_uri(env: Any) -> str:
    benchmark = getattr(env, "benchmark", None)
    if benchmark is None:
        raise FeatureExtractionError(
            "Environment has no active benchmark; call env.reset() first"
        )
    uri = getattr(benchmark, "uri", benchmark)
    return str(uri)


def _autophase_dict(env: Any) -> Dict[str, int]:
    """Extract and validate all 56 named Autophase features."""

    raw = _observation(env, "AutophaseDict", required=True)
    if not isinstance(raw, Mapping):
        raise FeatureExtractionError(
            f"AutophaseDict returned {type(raw).__name__}, expected a mapping"
        )

    expected = set(AUTOPHASE_FEATURE_NAMES)
    actual = set(raw.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FeatureExtractionError(
            "Autophase schema mismatch: "
            f"expected 56 keys, got {len(actual)}; missing={missing}, extra={extra}"
        )

    try:
        features = {name: int(raw[name]) for name in AUTOPHASE_FEATURE_NAMES}
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureExtractionError("Autophase contains a non-integer value") from error

    if any(value < 0 for value in features.values()):
        raise FeatureExtractionError("Autophase contains a negative feature value")
    return features


def _finite_float_samples(value: Any, observation_name: str) -> Tuple[float, ...]:
    """Validate a sequence of finite, non-negative timing values."""

    if value is None:
        return ()
    try:
        samples = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeatureExtractionError(
            f"Invalid timing observation {observation_name!r}: {value!r}"
        ) from error
    if any(not math.isfinite(item) or item < 0 for item in samples):
        raise FeatureExtractionError(
            f"Observation {observation_name!r} contains invalid timing values"
        )
    return samples


def _measure_runtime(env: Any, config: MeasurementConfig) -> Tuple[float, ...]:
    """Query Runtime while restoring the environment's measurement settings.

    The Runtime observation compiles and executes the benchmark executable. In
    some sandboxes the first attempt can transiently fail (e.g. a freshly
    rebuilt executable briefly reported as not executable). Retry briefly
    before giving up so one flaky measurement does not invalidate a pass.
    """

    old_count = getattr(env, "runtime_observation_count", None)
    old_warmups = getattr(env, "runtime_warmup_runs_count", None)
    try:
        env.runtime_observation_count = config.runtime_count
        env.runtime_warmup_runs_count = config.runtime_warmup_count
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                raw = _observation(env, "Runtime", required=True)
                samples = _finite_float_samples(raw, "Runtime")
                if not samples:
                    raise FeatureExtractionError(
                        "Runtime returned no samples for a benchmark marked runnable"
                    )
                return samples
            except Exception as error:  # transient build/exec failure
                last_error = error
                LOGGER.warning(
                    "Runtime observation attempt %d/3 failed, retrying: %s",
                    attempt + 1,
                    error,
                )
                time.sleep(1.0)
        raise FeatureExtractionError(
            f"Runtime observation failed after 3 attempts: {last_error}"
        ) from last_error
    finally:
        if old_count is not None:
            env.runtime_observation_count = old_count
        if old_warmups is not None:
            env.runtime_warmup_runs_count = old_warmups


def extract_features(
    env: Any,
    measurement: Optional[MeasurementConfig] = None,
) -> ProgramFeatures:
    """Extract a validated snapshot from an active LLVM environment.

    The caller must call ``env.reset(benchmark=...)`` before this function. No
    reset or optimization action is performed here, allowing the same function
    to capture both sides of a pass transition.

    Args:
        env: An active CompilerGym LLVM environment.
        measurement: Optional measurement policy. Defaults to inexpensive
            deterministic features plus object text size, with runtime and
            buildtime disabled.

    Returns:
        A validated :class:`ProgramFeatures` snapshot.

    Raises:
        FeatureExtractionError: If a required observation is unavailable or
            violates the expected CompilerGym 0.2.5 schema.
    """

    config = measurement or MeasurementConfig()
    if not getattr(env, "in_episode", False):
        raise FeatureExtractionError(
            "Environment is not in an episode; call env.reset() before extraction"
        )

    benchmark_uri = _benchmark_uri(env)
    autophase = _autophase_dict(env)
    state_id = str(_observation(env, "IrSha1", required=True))
    if not state_id:
        raise FeatureExtractionError("IrSha1 returned an empty state ID")

    ir_instruction_count = _scalar_int(
        _observation(env, "IrInstructionCount", required=True),
        "IrInstructionCount",
    )

    object_text_size: Optional[int] = None
    if config.collect_object_text_size:
        raw_object_size = _observation(env, "ObjectTextSizeBytes", required=False)
        if raw_object_size is not None:
            object_text_size = _scalar_int(raw_object_size, "ObjectTextSizeBytes")

    is_runnable = bool(
        _scalar_int(_observation(env, "IsRunnable", required=True), "IsRunnable")
    )
    is_buildable = bool(
        _scalar_int(_observation(env, "IsBuildable", required=True), "IsBuildable")
    )

    runtime_samples: Tuple[float, ...] = ()
    if config.measure_runtime:
        if is_runnable:
            runtime_samples = _measure_runtime(env, config)
        else:
            LOGGER.info("Skipping runtime for non-runnable benchmark %s", benchmark_uri)

    buildtime_sec: Optional[float] = None
    if config.measure_buildtime:
        if is_buildable:
            samples = _finite_float_samples(
                _observation(env, "Buildtime", required=True), "Buildtime"
            )
            if len(samples) != 1:
                raise FeatureExtractionError(
                    f"Buildtime returned {len(samples)} samples; expected exactly 1"
                )
            buildtime_sec = samples[0]
        else:
            LOGGER.info("Skipping buildtime for non-buildable benchmark %s", benchmark_uri)

    runtime_median = (
        statistics.median(runtime_samples) if runtime_samples else None
    )
    runtime_mean = statistics.fmean(runtime_samples) if runtime_samples else None
    runtime_std = (
        statistics.stdev(runtime_samples) if len(runtime_samples) >= 2 else None
    )

    return ProgramFeatures(
        schema_version=SCHEMA_VERSION,
        benchmark_uri=benchmark_uri,
        state_id=state_id,
        autophase=autophase,
        ir_instruction_count=ir_instruction_count,
        object_text_size_bytes=object_text_size,
        total_basic_blocks=autophase["TotalBlocks"],
        total_functions=autophase["TotalFuncs"],
        total_instructions=autophase["TotalInsts"],
        total_memory_instructions=autophase["TotalMemInst"],
        is_runnable=is_runnable,
        is_buildable=is_buildable,
        runtime_samples_sec=runtime_samples,
        runtime_median_sec=runtime_median,
        runtime_mean_sec=runtime_mean,
        runtime_std_sec=runtime_std,
        buildtime_sec=buildtime_sec,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one NeuroCompiler feature snapshot for diagnostics."
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark://cbench-v1/qsort",
        help="CompilerGym benchmark URI.",
    )
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument("--runtime-count", type=int, default=5)
    parser.add_argument("--runtime-warmup-count", type=int, default=1)
    parser.add_argument("--measure-buildtime", action="store_true")
    parser.add_argument(
        "--skip-object-text-size",
        action="store_true",
        help="Skip the platform-dependent object .TEXT size observation.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run a one-benchmark Stage 1 diagnostic against CompilerGym."""

    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        import compiler_gym
    except ImportError as error:
        raise SystemExit(
            "CompilerGym is unavailable. Activate the 'neurocompiler' Conda "
            "environment before running this command."
        ) from error

    config = MeasurementConfig(
        measure_runtime=args.measure_runtime,
        runtime_count=args.runtime_count,
        runtime_warmup_count=args.runtime_warmup_count,
        measure_buildtime=args.measure_buildtime,
        collect_object_text_size=not args.skip_object_text_size,
    )

    env = compiler_gym.make("llvm-v0")
    try:
        env.reset(benchmark=args.benchmark)
        snapshot = extract_features(env, config)
        print(json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
