"""re_polar/datasets/mawps.py: record-building + dupe/fraction accounting (no network)."""

import json

from re_polar.datasets.mawps import build_eval_split


def _row(id_, question, result, result_float, equation="x=1", expression="1"):
    return {
        "id": id_,
        "question": question,
        "chain": "",
        "result": result,
        "result_float": result_float,
        "equation": equation,
        "expression": expression,
    }


def test_build_eval_split_basic_fields_and_fraction_flag(tmp_path, monkeypatch):
    rows = [
        _row("mawps__a", "How many apples?", "9", 9.0),
        _row("mawps__b", "What fraction?", "56/9", 6.222),
    ]
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr("re_polar.datasets.mawps.EXPECTED_N", 2)

    out_dir = tmp_path / "mawps"
    manifest = build_eval_split(out_dir)

    records = json.loads((out_dir / "test.json").read_text())
    assert len(records) == 2
    assert records[0]["gt_ans"] == "9"
    assert records[1]["gt_ans"] == "56/9"
    assert manifest["n_fractional_gt_ans"] == 1
    assert manifest["n_total"] == 2
    assert [r["id"] for r in records] == [0, 1]


def test_build_eval_split_counts_duplicate_questions(tmp_path, monkeypatch):
    rows = [
        _row("mawps__a", "same question", "1", 1.0),
        _row("mawps__b", "same question", "1", 1.0),
    ]
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr("re_polar.datasets.mawps.EXPECTED_N", 2)

    manifest = build_eval_split(tmp_path / "mawps")
    assert manifest["n_duplicate_questions_within_test"] == 1
    assert manifest["n_total"] == 2  # duplicates are counted, not dropped
