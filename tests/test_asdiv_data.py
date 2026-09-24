"""re_polar/datasets/asdiv.py: answer-parsing logic (no network).

The real build_eval_split() needs the HF dataset; here `datasets.load_dataset`
is faked so the test exercises only our parsing/flagging logic at CPU speed.
"""

import json

from re_polar.datasets.asdiv import build_eval_split, parse_answer


def test_parse_answer_strips_single_unit():
    assert parse_answer("9 (apples)") == ["9"]


def test_parse_answer_passes_through_no_unit():
    assert parse_answer("Purple") == ["Purple"]
    assert parse_answer("383") == ["383"]


def test_parse_answer_splits_multi_value_before_stripping_units():
    # a naive single regex over the whole string grabs only the LAST "(...)"
    # and mis-parses everything before it as one blob -- must split on ";" first.
    assert parse_answer("5 (years old); 15 (years old); 20 (years old)") == ["5", "15", "20"]


def _row(body, question, solution_type, answer, formula="1+1=2"):
    return {
        "body": body,
        "question": question,
        "solution_type": solution_type,
        "answer": answer,
        "formula": formula,
    }


def test_build_eval_split_flags_numeric_other_and_multi(tmp_path, monkeypatch):
    rows = [
        _row("b1", "q1", "Addition", "9 (apples)"),
        _row("b2", "q2", "Comparison", "Purple"),
        _row("b3", "q3", "TVQ-Change", "5 (years old); 15 (years old)"),
    ]

    import datasets

    class FakeDS(list):
        pass

    fake = FakeDS(rows)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: fake)
    monkeypatch.setattr("re_polar.datasets.asdiv.EXPECTED_N", 3)

    out_dir = tmp_path / "asdiv"
    manifest = build_eval_split(out_dir)

    records = json.loads((out_dir / "test.json").read_text())
    assert len(records) == 3

    r0, r1, r2 = records
    assert r0["question"] == "b1 q1"
    assert r0["gt_ans"] == "9" and r0["is_pure_numeric"] is True and r0["is_multi_answer"] is False

    assert (
        r1["gt_ans"] == "Purple"
        and r1["is_pure_numeric"] is False
        and r1["is_multi_answer"] is False
    )

    assert r2["gt_ans"] == "5, 15" and r2["gt_ans_parts"] == ["5", "15"]
    assert r2["is_multi_answer"] is True and r2["is_pure_numeric"] is False

    assert manifest["answer_shape"] == {
        "single_part_pure_numeric": 1,
        "single_part_non_numeric": 1,
        "multi_part": 1,
    }
    assert manifest["n_total"] == 3
    # ids are a flat 0..n-1 sequence
    assert [r["id"] for r in records] == [0, 1, 2]


def test_build_eval_split_counts_duplicate_content_keys(tmp_path, monkeypatch):
    rows = [
        _row("b1", "q1", "Addition", "9 (apples)"),
        _row("b1", "q1", "Addition", "9 (apples)"),  # exact duplicate row
    ]
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr("re_polar.datasets.asdiv.EXPECTED_N", 2)

    manifest = build_eval_split(tmp_path / "asdiv")
    assert manifest["n_duplicate_content_keys"] == 1
    assert manifest["n_total"] == 2  # duplicates are counted, not dropped
