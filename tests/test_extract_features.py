"""Unit tests for Stage 1 feature extraction without a CompilerGym service."""

from __future__ import annotations

import json

import pytest

from scripts.extract_features import (
    AUTOPHASE_FEATURE_NAMES,
    FeatureExtractionError,
    MeasurementConfig,
    extract_features,
)


class FakeObservationView:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, name):
        value = self.values[name]
        if isinstance(value, Exception):
            raise value
        return value


class FakeBenchmark:
    uri = "benchmark://cbench-v1/qsort"


class FakeEnv:
    def __init__(self):
        autophase = {name: index for index, name in enumerate(AUTOPHASE_FEATURE_NAMES)}
        self.in_episode = True
        self.benchmark = FakeBenchmark()
        self.runtime_observation_count = 30
        self.runtime_warmup_runs_count = 0
        self.observation = FakeObservationView(
            {
                "AutophaseDict": autophase,
                "IrSha1": "abc123",
                "IrInstructionCount": [500],
                "ObjectTextSizeBytes": [4096],
                "IsRunnable": 1,
                "IsBuildable": [1],
                "Runtime": [0.9, 1.0, 1.1],
                "Buildtime": [0.25],
            }
        )


def test_extracts_complete_snapshot_and_flattens_all_features():
    env = FakeEnv()
    result = extract_features(env)

    assert result.benchmark_uri == "benchmark://cbench-v1/qsort"
    assert result.state_id == "abc123"
    assert result.ir_instruction_count == 500
    assert result.object_text_size_bytes == 4096
    assert len(result.autophase) == 56
    assert result.total_basic_blocks == AUTOPHASE_FEATURE_NAMES.index("TotalBlocks")

    flat = result.flattened("pre_")
    feature_columns = [key for key in flat if key.startswith("pre_autophase_")]
    assert len(feature_columns) == 56
    assert flat["pre_state_id"] == "abc123"
    assert json.loads(flat["pre_runtime_samples_json"]) == []


def test_runtime_and_buildtime_are_opt_in_and_restore_environment_settings():
    env = FakeEnv()
    result = extract_features(
        env,
        MeasurementConfig(
            measure_runtime=True,
            runtime_count=3,
            runtime_warmup_count=2,
            measure_buildtime=True,
        ),
    )

    assert result.runtime_samples_sec == (0.9, 1.0, 1.1)
    assert result.runtime_median_sec == pytest.approx(1.0)
    assert result.runtime_mean_sec == pytest.approx(1.0)
    assert result.runtime_std_sec == pytest.approx(0.1)
    assert result.buildtime_sec == pytest.approx(0.25)
    assert env.runtime_observation_count == 30
    assert env.runtime_warmup_runs_count == 0


def test_non_runnable_benchmark_skips_runtime():
    env = FakeEnv()
    env.observation.values["IsRunnable"] = 0
    env.observation.values["Runtime"] = AssertionError("must not be queried")

    result = extract_features(
        env, MeasurementConfig(measure_runtime=True, collect_object_text_size=False)
    )
    assert result.runtime_samples_sec == ()
    assert result.runtime_median_sec is None


def test_rejects_autophase_schema_mismatch():
    env = FakeEnv()
    del env.observation.values["AutophaseDict"]["TotalFuncs"]

    with pytest.raises(FeatureExtractionError, match="Autophase schema mismatch"):
        extract_features(env)


def test_requires_active_episode():
    env = FakeEnv()
    env.in_episode = False

    with pytest.raises(FeatureExtractionError, match="call env.reset"):
        extract_features(env)
