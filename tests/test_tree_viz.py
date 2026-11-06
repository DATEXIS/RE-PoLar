"""Viewer trees from the released data: question text only via --questions-dir, a search's
unevaluated parents stay in the tree, and runs on the same questions merge into a run
picker."""

import gzip
import json
import sys

import pytest

from analysis.tree_viz import build, extract

DEPTH = 28


def _released_run(root, run):
    path = list(range(DEPTH))
    skip = path[:3] + path[4:]
    two_skips = path[:3] + path[4:8] + path[9:]
    sample = {
        "final_valid_transitions": [path],
        "final_invalid_transitions": [skip, two_skips],
        "initial_transition_metric": 1.0,
        "sample_info": {"query_id": "q0", "domain": "Algebra", "math_level": 1},
        "search_trajectory": [
            {"path": skip, "parent_path": path, "reward": 0.0},
            {"path": two_skips, "parent_path": path[:8] + path[9:], "reward": 0.0},
        ],
    }
    d = root / "mcts_results" / run / "seed42" / "dart-math-diff-1"
    d.mkdir(parents=True)
    with gzip.open(d / "merged_mcts_samples.json.gz", "wt") as f:
        json.dump({"samples": [sample]}, f)
    a = root / "generated_answers" / run / "seed42"
    a.mkdir(parents=True)
    with gzip.open(a / "diff1.jsonl.gz", "wt") as f:
        f.write(json.dumps({"query_id": "q0", "path": path, "text": "t", "reward": 1.0}) + "\n")


def _extract(monkeypatch, root, out, *extra):
    argv = ["extract", "--released", "--data-root", str(root), "--model-tag", "qwen3_8b"]
    argv += ["--depth", str(DEPTH), "--pick", "1:0", "--output", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    extract.main()


def test_released_trees_questions_unevaluated_parents_and_run_picker(tmp_path, monkeypatch):
    _released_run(tmp_path / "ours", "qwen3_8b")
    _released_run(tmp_path / "theirs", "qwen3_8b/oneshot")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _extract(monkeypatch, tmp_path / "ours", a, "--label", "Ours")
    _extract(monkeypatch, tmp_path / "theirs", b, "--label", "Theirs", "--prompt", "oneshot")
    tree = json.loads(a.read_text())["trees"][0]
    assert tree["question"] is None and tree["gt_ans"] is None
    assert tree["n_valid"] == 1 and tree["n_invalid"] == 2 and tree["n_programs"] == 3
    assert {
        n["edits_label"]: n["reward"] for n in tree["nodes"].values() if n["is_valid"] is not None
    }[
        "skip 3"
    ] == 0.0  # no answer logged: the reward comes from the search's verdict
    unevaluated = [n for n in tree["nodes"].values() if n["is_valid"] is None]
    assert [n["edits_label"] for n in unevaluated] == ["skip 8"] and unevaluated[0]["children"]

    qdir = tmp_path / "questions" / "diff1"
    qdir.mkdir(parents=True)
    rec = {"query_id": "q0", "question": "Q", "gt_ans": "A"}
    (qdir / "train.json").write_text(json.dumps([rec]))
    c = tmp_path / "c.json"
    extra = ("--questions-dir", str(qdir.parent), "--prompt-style", "oneshot")
    _extract(monkeypatch, tmp_path / "ours", c, *extra)
    run = json.loads(c.read_text())
    assert run["prompt_style"] == "oneshot"
    tree = run["trees"][0]
    assert tree["question"] == "Q" and tree["gt_ans"] == "A"

    out = tmp_path / "viewer.html"
    monkeypatch.setattr(sys, "argv", ["build", str(a), str(b), str(out)])
    build.main()
    embedded = out.read_text().split('id="tree-data">')[1].split("</script>")[0]
    models = json.loads(embedded)["models"]
    assert [m["model_label"] for m in models] == ["qwen3_8b", "qwen3_8b"]
    assert [m["label"] for m in models] == ["Ours", "Theirs"]
    assert [m["seed"] for m in models] == [42, 42]
    assert all(t["fit_depth"] == t["max_edit_count"] for m in models for t in m["trees"])
    assert [m["trees"][0]["label"] for m in models] == ["Question 1", "Question 1"]

    lazy = tmp_path / "lazy" / "viewer.html"
    lazy.parent.mkdir()
    monkeypatch.setattr(sys, "argv", ["build", str(a), str(b), str(lazy), "--lazy", "runs"])
    build.main()
    embedded = lazy.read_text().split('id="tree-data">')[1].split("</script>")[0]
    run = json.loads(embedded)["models"][1]
    assert "trees" not in run and run["src"] == "runs/qwen3_8b_theirs_seed42.json"
    assert json.loads((lazy.parent / run["src"]).read_text())["trees"][0]["n_valid"] == 1

    runs = [json.loads(a.read_text()), json.loads(a.read_text())]
    runs[1]["seed"] = 1
    runs[1]["trees"][0]["max_edit_count"] = 7
    build.add_fit(runs)
    assert [r["trees"][0]["fit_depth"] for r in runs] == [7, 7]

    monkeypatch.setattr(sys, "argv", ["build", str(a), str(b), str(out)])
    other = json.loads(b.read_text())
    other["trees"][0]["query_id"] = "q1"
    b.write_text(json.dumps(other))
    with pytest.raises(AssertionError, match="same questions"):
        build.main()
