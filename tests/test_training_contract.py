"""Tests for runtime SL training/inference encoding contract."""
from training.train_sl import build_action_vocab, encode_row, target_value


def test_action_encoding_is_one_hot_and_not_ordinal_pass_id():
    rows = [
        {"pass_flag": "-gvn"},
        {"pass_flag": "-licm"},
    ]
    vocab = build_action_vocab(rows)
    row = {"pre_ir_instruction_count": "100", "pass_flag": "-gvn", "pass_id": "99"}
    encoded = encode_row(row, ["pre_ir_instruction_count"], vocab)
    assert encoded[0] == 100.0
    assert encoded[1:] == [1.0 if i == vocab["-gvn"] else 0.0 for i in range(2)]
    assert sum(encoded[1:]) == 1.0


def test_runtime_target_is_read_directly():
    row = {"runtime_improvement_pct": "12.5", "step_reward": "0.1"}
    assert target_value(row, "runtime_improvement_pct") == 12.5
    assert target_value(row, "step_reward") == 0.1
