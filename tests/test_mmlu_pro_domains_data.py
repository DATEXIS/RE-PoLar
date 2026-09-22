"""re_polar/datasets/mmlu_pro_domains.py: dedup-key + disjoint-capping logic (no
network). build_train_split() reads a local mmlu_pro_official-shaped JSON
file (written to a tmp path here), no HF download, no frozen-fixture mock:
it sources from the local mmlu_pro_official pool directly.
"""

import json

from re_polar.datasets.mmlu_pro_domains import _dedup_key, build_train_split


def _row(category, question, options, answer_index):
    return {"category": category, "question": question, "options": options,
            "answer_index": answer_index}


def _write_official(tmp_path, rows):
    path = tmp_path / "mmlu_pro_official_test.json"
    path.write_text(json.dumps(rows))
    return path


def test_dedup_key_distinguishes_same_question_different_options():
    # real-world case (MMLU-Pro `law`): same lead-in question text, different
    # options/answer -> must NOT collide.
    a = _row("law", "Maine's famous aphorism...", ["opt A", "opt B"], 0)
    b = _row("law", "Maine's famous aphorism...", ["opt C", "opt D"], 1)
    assert _dedup_key(a) != _dedup_key(b)


def test_dedup_key_stable_regardless_of_extra_fields():
    # HF rows carry extra columns (question_id, cot_content, src, answer);
    # the key must only depend on the 4 fields that define row IDENTITY here.
    a = _row("math", "2+2=?", ["3", "4"], 1)
    b = dict(a, question_id=999, cot_content="", src="ori_mmlu-x", answer="B")
    assert _dedup_key(a) == _dedup_key(b)


def test_build_train_split_caps_thin_domains(tmp_path):
    domains = ["math", "history"]

    # math has plenty; history has only 3 rows total -> after carving
    # test_n_per_domain=1, only 2 remain, too thin for the target n_per_domain=3
    # -> must be CAPPED, not raise/oversample.
    official_rows = (
        [_row("math", f"m-{i}", ["a", "b"], i % 2) for i in range(8)]
        + [_row("history", f"h-{i}", ["a", "b"], i % 2) for i in range(3)]
    )
    official_path = _write_official(tmp_path, official_rows)

    out_dir = tmp_path / "mmlu_pro_domains_14"
    manifest = build_train_split(out_dir, official_path=official_path, seed=0,
                                 n_per_domain=3, domains=domains, test_n_per_domain=1)

    train = json.loads((out_dir / "train.json").read_text())
    test = json.loads((out_dir / "test.json").read_text())
    train_keys = {_dedup_key(r) for r in train}
    test_keys = {_dedup_key(r) for r in test}
    assert not (train_keys & test_keys)  # carved test never overlaps train

    by_domain = {}
    for r in train:
        by_domain.setdefault(r["category"], []).append(r)
    assert len(by_domain["math"]) == 3          # plenty of pool -> hits the target
    assert len(by_domain["history"]) == 2       # 3 total - 1 carved test = 2 available -> capped

    assert manifest["domains"]["math"]["capped"] is False
    assert manifest["domains"]["history"]["capped"] is True
    assert manifest["domains"]["history"]["available_pool"] == 2
    assert manifest["n_total"] == 5
    # ids are a flat 0..n-1 sequence across the combined output
    assert [r["id"] for r in train] == list(range(len(train)))
    # val_frac=0 (default) -> no val.json
    assert not (out_dir / "val.json").exists()


def test_build_train_split_val_frac_is_disjoint_from_train_and_test(tmp_path):
    # 3 target domains, larger pools so a val_frac=0.2 carve-out has room to
    # produce a non-trivial per-domain split.
    domains = ["math", "physics", "history"]

    official_rows = (
        [_row("math", f"m-{i}", ["a", "b"], i % 2) for i in range(22)]
        + [_row("physics", f"p-{i}", ["a", "b"], i % 2) for i in range(22)]
        # history stays THIN (5 total) so val_frac's round() must behave
        # sanely (n_val=round(3*0.2)=1) even under scarcity after a 2-row carve.
        + [_row("history", f"h-{i}", ["a", "b"], i % 2) for i in range(5)]
    )
    official_path = _write_official(tmp_path, official_rows)

    out_dir = tmp_path / "mmlu_pro_domains_14"
    manifest = build_train_split(out_dir, official_path=official_path, seed=0,
                                 n_per_domain=10, val_frac=0.2, domains=domains,
                                 test_n_per_domain=2)

    train = json.loads((out_dir / "train.json").read_text())
    val = json.loads((out_dir / "val.json").read_text())
    test = json.loads((out_dir / "test.json").read_text())
    train_keys = {_dedup_key(r) for r in train}
    val_keys = {_dedup_key(r) for r in val}
    test_keys = {_dedup_key(r) for r in test}

    # the 3-way split (TRAIN/VAL/TEST) is pairwise disjoint by content key
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)

    assert manifest["val_frac"] == 0.2
    assert manifest["n_val_total"] == len(val)
    # math/physics: 22 total - 2 carved test = 20 available, target=10,
    # val=round(10*0.2)=2, train=8
    assert manifest["domains"]["math"] == {**manifest["domains"]["math"],
                                           "train": 8, "val": 2}
    assert manifest["domains"]["physics"]["train"] == 8
    assert manifest["domains"]["physics"]["val"] == 2
    # history: 5 total - 2 carved test = 3 available, capped at 3,
    # val=round(3*0.2)=1, train=2
    assert manifest["domains"]["history"]["capped"] is True
    assert manifest["domains"]["history"]["train"] == 2
    assert manifest["domains"]["history"]["val"] == 1
    # ids are flat 0..n-1 within each output file independently
    assert [r["id"] for r in train] == list(range(len(train)))
    assert [r["id"] for r in val] == list(range(len(val)))


def test_every_domain_gets_its_own_carved_test_split(tmp_path):
    """There is no frozen fixture to fall back on for any domain,
    build_train_split must carve TEST itself for ALL domains, BEFORE drawing
    train/val, so the three splits are disjoint by construction and a router
    can never be evaluated on rows it trained on."""
    domains = ["math", "biology"]

    official_rows = (
        [_row("math", f"m-{i}", ["a", "b"], i % 2) for i in range(12)]
        + [_row("biology", f"b-{i}", ["a", "b"], i % 2) for i in range(10)]
    )
    official_path = _write_official(tmp_path, official_rows)

    out_dir = tmp_path / "mmlu14"
    manifest = build_train_split(out_dir, official_path=official_path, seed=0,
                                 n_per_domain=5, val_frac=0.2, domains=domains,
                                 test_n_per_domain=3)

    train = json.loads((out_dir / "train.json").read_text())
    val = json.loads((out_dir / "val.json").read_text())
    test = json.loads((out_dir / "test.json").read_text())

    # both domains contribute carved test rows now
    assert {r["category"] for r in test} == {"math", "biology"}
    assert len(test) == 6
    assert manifest["domains"]["biology"]["carved_test_rows"] == 3
    assert manifest["domains"]["math"]["carved_test_rows"] == 3

    # three-way disjoint
    tr, va, te = ({_dedup_key(r) for r in x} for x in (train, val, test))
    assert not (tr & va) and not (tr & te) and not (va & te)

    # biology's train/val pool must be what's LEFT after the carve (10 - 3 = 7,
    # capped at n_per_domain=5), i.e. the carve happens first
    assert manifest["domains"]["biology"]["available_pool"] == 7


def test_uncapped_n_per_domain_makes_train_val_test_cover_the_whole_pool(tmp_path):
    """A full MCTS run uses --split all (train+val+test unioned, see
    re_polar/mcts/run_search_mmlu_pro_domains.py) and that's only truly the WHOLE
    dataset if no domain's train pool got truncated. A large enough
    --n-per-domain (uncapped relative to the pool) must make train+val+test
    partition every single row, nothing dropped."""
    domains = ["math", "history"]
    official_rows = (
        [_row("math", f"m-{i}", ["a", "b"], i % 2) for i in range(1351)]
        + [_row("history", f"h-{i}", ["a", "b"], i % 2) for i in range(381)]
    )
    official_path = _write_official(tmp_path, official_rows)

    out_dir = tmp_path / "mmlu_pro_domains_14"
    manifest = build_train_split(out_dir, official_path=official_path, seed=0,
                                 n_per_domain=100000, val_frac=0.15, domains=domains,
                                 test_n_per_domain=200)

    train = json.loads((out_dir / "train.json").read_text())
    val = json.loads((out_dir / "val.json").read_text())
    test = json.loads((out_dir / "test.json").read_text())

    # NOTE: `capped` is True here for both domains -- it means "pool < the
    # n_per_domain argument" (100000, a deliberately-huge sentinel), NOT
    # "rows were dropped". The real no-rows-dropped guarantee is the row-count
    # check below (every row lands in train, val, or test -- none discarded).
    assert manifest["domains"]["math"]["total_rows"] == 1351
    assert manifest["domains"]["history"]["total_rows"] == 381
    by_domain = {"math": 1351, "history": 381}
    for d, total in by_domain.items():
        n_in_splits = (sum(1 for r in train if r["category"] == d)
                      + sum(1 for r in val if r["category"] == d)
                      + sum(1 for r in test if r["category"] == d))
        assert n_in_splits == total, f"{d}: {n_in_splits} != {total} -- rows dropped"
    assert len(train) + len(val) + len(test) == 1351 + 381


def test_paper_domain_list_is_the_thirteen_reported_ones():
    """Guards the constant against silent edits: PoLar's ICML Tables 3/8/9/10
    report exactly these 13 MMLU-Pro subjects, and notably NOT computer science
    (which IS in ALL_MMLU_PRO_DOMAINS, the current default -- an easy confusion)."""
    from re_polar.datasets.mmlu_pro_domains import PAPER_MMLU_PRO_DOMAINS

    assert len(PAPER_MMLU_PRO_DOMAINS) == 13
    assert "computer science" not in PAPER_MMLU_PRO_DOMAINS
    assert PAPER_MMLU_PRO_DOMAINS[:4] == ["math", "physics", "chemistry", "law"]


def test_all_domain_list_has_all_fourteen_including_computer_science():
    from re_polar.datasets.mmlu_pro_domains import ALL_MMLU_PRO_DOMAINS, PAPER_MMLU_PRO_DOMAINS

    assert len(ALL_MMLU_PRO_DOMAINS) == 14
    assert "computer science" in ALL_MMLU_PRO_DOMAINS
    assert set(PAPER_MMLU_PRO_DOMAINS) < set(ALL_MMLU_PRO_DOMAINS)
