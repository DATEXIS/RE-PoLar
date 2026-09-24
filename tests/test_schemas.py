"""merged_mcts_samples.json emitter: structure the router training code needs
(see re_polar/datasets/schemas.py docstring)."""

import json

from re_polar.datasets.schemas import load_samples, sample_record, write_merged_samples


def test_emitted_structure(tmp_path):
    inp = {
        "query_id": "MATH/train/algebra/1.json",
        "question": "1+1?",
        "gt_ans": "2",
        "difficulty": 3,
        "fail_rate": 0.4,
    }
    rec = sample_record(
        inp, valid_paths=[[0, 1], [0, 1, 2]], invalid_paths=[[2]], initial_metric=1.0
    )
    out = write_merged_samples(tmp_path, "Qwen/Qwen3-8B", "dart-math-diff-3", [rec])

    assert out == tmp_path / "Qwen/Qwen3-8B/dart-math-diff-3/merged_mcts_samples.json"
    data = json.loads(out.read_text())
    s = data["samples"][0]
    # exactly what the router training code consumes:
    assert s["question"] == "1+1?" and s["gt_ans"] == "2"
    assert s["final_valid_transitions"] == [[0, 1], [0, 1, 2]]
    assert s["initial_transition_metric"] == 1.0
    # our extras, tolerated by that reader:
    assert s["final_invalid_transitions"] == [[2]]
    assert s["sample_info"]["difficulty"] == 3


def test_load_samples_round_trips(tmp_path):
    inp = {
        "query_id": "MATH/train/algebra/1.json",
        "question": "1+1?",
        "gt_ans": "2",
        "difficulty": 3,
        "fail_rate": 0.4,
    }
    rec = sample_record(
        inp, valid_paths=[[0, 1], [0, 1, 2]], invalid_paths=[[2]], initial_metric=1.0
    )
    out = write_merged_samples(tmp_path, "Qwen/Qwen3-8B", "dart-math-diff-3", [rec])

    loaded = load_samples(out)
    assert loaded == [rec]


def test_trajectory_omitted_when_not_passed():
    """Default (no trajectory arg) -> no "search_trajectory" key at all (not
    even null), so a reader that doesn't know about it sees exactly the
    older, smaller schema."""
    inp = {"query_id": "q1", "question": "1+1?", "gt_ans": "2"}
    rec = sample_record(inp, valid_paths=[[0, 1]], invalid_paths=[], initial_metric=1.0)
    assert "search_trajectory" not in rec


def test_trajectory_included_when_passed(tmp_path):
    inp = {"query_id": "q1", "question": "1+1?", "gt_ans": "2"}
    traj = [{"path": [0, 1], "parent_path": None, "reward": 1.0}]
    rec = sample_record(
        inp, valid_paths=[[0, 1]], invalid_paths=[], initial_metric=1.0, trajectory=traj
    )
    assert rec["search_trajectory"] == traj

    out = write_merged_samples(tmp_path, "Qwen/Qwen3-8B", "dart-math-diff-3", [rec])
    loaded = load_samples(out)
    assert loaded[0]["search_trajectory"] == traj
