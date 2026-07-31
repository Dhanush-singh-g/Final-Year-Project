"""Integration-style tests for transition recording and CSV processing."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.extract_features import AUTOPHASE_FEATURE_NAMES, MeasurementConfig
from scripts.generate_dataset import CSV_FIELDS, PROVENANCE_FIELDS
from scripts.process_dataset import process_dataset
from scripts.run_passes import ActionMetadata, run_pass_sequence, transition_fieldnames


class ObservationView:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, name):
        values = {
            "AutophaseDict": {
                feature: index + self.env.offset
                for index, feature in enumerate(AUTOPHASE_FEATURE_NAMES)
            },
            "IrSha1": f"state-{self.env.offset}",
            "IrInstructionCount": [500 - self.env.offset],
            "ObjectTextSizeBytes": [4096 - self.env.offset],
            "IsRunnable": 0,
            "IsBuildable": 0,
        }
        return values[name]


class FakeEnv:
    in_episode = True
    benchmark = "benchmark://cbench-v1/qsort"
    episode_reward = 0.0

    def __init__(self):
        self.offset = 0
        self.observation = ObservationView(self)

    def step(self, action, timeout=300):
        self.offset += 1
        self.episode_reward += 0.25
        return None, 0.25, False, {"action_had_no_effect": False}


def test_transition_row_has_complete_schema():
    env = FakeEnv()
    action = ActionMetadata(7, "adce", "-adce", "dead code elimination")
    transitions = run_pass_sequence(
        env,
        [action],
        reward_space="IrInstructionCountO3",
        measurement=MeasurementConfig(collect_object_text_size=True),
    )
    row = transitions[0].to_row()
    assert set(row) == set(transition_fieldnames())
    assert row["pass_success"] is True
    assert row["pre_state_id"] == "state-0"
    assert row["post_state_id"] == "state-1"
    assert row["delta_ir_instruction_count"] == -1
    assert row["delta_autophase_TotalBlocks"] == 1


def test_process_dataset_produces_training_csv(tmp_path: Path):
    env = FakeEnv()
    action = ActionMetadata(7, "adce", "-adce", "dead code elimination")
    transition = run_pass_sequence(
        env, [action], reward_space="IrInstructionCountO3"
    )[0]
    raw_row = {field: "" for field in CSV_FIELDS}
    raw_row.update(
        {
            "transition_key": "key-1",
            "run_id": "run",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "benchmark_suite": "benchmark://cbench-v1",
            "compiler_gym_version": "0.2.5",
            "compiler_version": "LLVM 10",
            "host_name": "test",
            "python_version": "3.10",
        }
    )
    raw_row.update(transition.to_row())

    raw_path = tmp_path / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(raw_row)

    output = tmp_path / "hybrid.csv"
    process_dataset(raw_path, output)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["dataset_split"] == "train"
    assert "norm_pre_autophase_TotalBlocks" in rows[0]
    assert (tmp_path / "normalization.json").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
