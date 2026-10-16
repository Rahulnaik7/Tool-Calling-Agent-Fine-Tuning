import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_gen import generate_bfcl_subset, generate_training_set, verify_call
from tools_registry import TOOLS


def test_verify_call_accepts_valid_required_args():
    assert verify_call("math_service", {"operation": "add", "x": 1.0, "y": 2.0})


def test_verify_call_rejects_missing_required_arg():
    assert not verify_call("math_service", {"operation": "add", "x": 1.0})


def test_verify_call_rejects_wrong_type():
    assert not verify_call("math_service", {"operation": "add", "x": "not a number", "y": 2.0})


def test_verify_call_rejects_invalid_enum_value():
    assert not verify_call("weather_api", {"location": "Tokyo", "units": "lightyears"})


def test_verify_call_rejects_unknown_tool():
    assert not verify_call("does_not_exist", {})


def test_verify_call_rejects_unknown_argument_key():
    assert not verify_call("weather_api", {"location": "Tokyo", "bogus": 1})


def test_generate_training_set_every_row_is_execution_verified():
    rows = generate_training_set(target_size=80, seed=1)
    assert len(rows) == 80
    for row in rows:
        assert verify_call(row["tool"], row["arguments"])


def test_generate_training_set_includes_distractor_tools():
    rows = generate_training_set(target_size=50, seed=1)
    for row in rows:
        assert row["tool"] in row["candidate_tools"]
        assert len(row["candidate_tools"]) >= 2  # correct tool + at least one distractor
        assert len(set(row["candidate_tools"])) == len(row["candidate_tools"])  # no duplicate tools


def test_generate_training_set_has_no_duplicate_instructions():
    rows = generate_training_set(target_size=200, seed=2)
    instructions = [row["instruction"] for row in rows]
    assert len(instructions) == len(set(instructions))


def test_generate_bfcl_subset_every_row_is_execution_verified():
    rows = generate_bfcl_subset(size=40, seed=3)
    for row in rows:
        assert verify_call(row["tool"], row["arguments"])


def test_generate_bfcl_subset_has_distractors():
    rows = generate_bfcl_subset(size=40, seed=3, n_distractors=2)
    for row in rows:
        assert row["tool"] in row["candidate_tools"]
        assert len(row["candidate_tools"]) == 3  # 1 correct + 2 distractors


def test_all_registered_tools_have_verifiable_schema():
    # Sanity check on the registry itself: every tool's required/optional
    # param dicts use real Python types the verifier understands.
    for name, spec in TOOLS.items():
        for params in (spec["required"], spec["optional"]):
            for key, ptype in params.items():
                assert ptype in (str, int, float, bool), f"{name}.{key} has unsupported type {ptype}"
