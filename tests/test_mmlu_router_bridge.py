"""re_polar/router/mmlu_bridge.py: mmlu supervision -> re_polar.router.train-ready shape.

CPU-only, no model download: `format_mmlu_question` only needs
`re_polar.core.mmlu_pro_scoring._build_prompt` (pure string formatting), and
`load_mmlu_supervision` only needs a JSON file on disk.
"""

import json

from re_polar.router.mmlu_bridge import format_mmlu_question, load_mmlu_supervision


def test_format_mmlu_question_is_lettered_and_hashable():
    q = {"question": "2+2=?", "options": ["3", "4", "5"], "category": "math"}
    text = format_mmlu_question(q)
    assert isinstance(text, str)
    assert "2+2=?" in text
    assert "A. 3" in text and "B. 4" in text and "C. 5" in text
    # deterministic -- same input, same output (needed as an Example dict key)
    assert format_mmlu_question(q) == text


def test_format_mmlu_question_no_think_prefix_toggle():
    q = {"question": "2+2=?", "options": ["3", "4"], "category": "math"}
    assert format_mmlu_question(q, no_think=True).startswith("/no_think")
    assert not format_mmlu_question(q, no_think=False).startswith("/no_think")


def _mmlu_sample_record(question_text, options, category, gt_ans, valid, invalid):
    # mirrors re_polar/datasets/schemas.py::sample_record's output shape for an mmlu
    # input (question is a NESTED DICT, unlike DART-Math's plain string).
    return {
        "question": {"question": question_text, "options": options, "category": category},
        "gt_ans": gt_ans,
        "final_valid_transitions": valid,
        "final_invalid_transitions": invalid,
        "initial_transition_metric": 1.0,
        "sample_info": {"category": category, "query_id": "mmlu-0"},
    }


def test_load_mmlu_supervision_flattens_question_and_keeps_struct(tmp_path):
    rec = _mmlu_sample_record(
        "2+2=?",
        ["3", "4"],
        "math",
        gt_ans=1,
        valid=[[0, 1, 2]],
        invalid=[],
    )
    path = tmp_path / "merged_mcts_samples.json"
    path.write_text(json.dumps({"samples": [rec]}))

    out = load_mmlu_supervision(path)
    assert len(out) == 1
    s = out[0]
    # question is now a router-encodable STRING, matching DART-Math's shape
    assert isinstance(s["question"], str)
    assert s["question"] == format_mmlu_question(rec["question"])
    # original structured dict preserved for reward-fn grading
    assert s["question_struct"] == rec["question"]
    # everything else passes through untouched (gt_ans stays an INT, unlike DART-Math)
    assert s["gt_ans"] == 1
    assert isinstance(s["gt_ans"], int)
    assert s["final_valid_transitions"] == [[0, 1, 2]]


def test_load_mmlu_supervision_rejects_dart_math_shaped_samples(tmp_path):
    # DART-Math's question is a plain string -- load_mmlu_supervision must
    # crash loudly rather than silently mis-flattening it (str has no
    # "question"/"options" keys, so a dict-typed check must reject it upfront).
    rec = {
        "question": "solve x",
        "gt_ans": "42",
        "final_valid_transitions": [],
        "final_invalid_transitions": [],
        "initial_transition_metric": 1.0,
    }
    path = tmp_path / "merged_mcts_samples.json"
    path.write_text(json.dumps({"samples": [rec]}))

    import pytest

    with pytest.raises(ValueError, match="mmlu question dict"):
        load_mmlu_supervision(path)


def test_load_mmlu_supervision_concatenates_multiple_files(tmp_path):
    rec_a = _mmlu_sample_record("qa?", ["x", "y"], "math", 0, [[0]], [])
    rec_b = _mmlu_sample_record("qb?", ["x", "y"], "physics", 1, [[1]], [])
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps({"samples": [rec_a]}))
    path_b.write_text(json.dumps({"samples": [rec_b]}))

    out = load_mmlu_supervision([path_a, path_b])
    assert len(out) == 2
    assert {s["question_struct"]["category"] for s in out} == {"math", "physics"}
