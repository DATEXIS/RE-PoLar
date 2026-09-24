"""Tests for re_polar/mcts/analysis/ -- the MCTS-supervision-data analysis pipeline.
Torch-free, local CPU, synthetic fixtures only (no released data needed).
"""

import json

import pytest

from re_polar.mcts.analysis import segments
from re_polar.mcts.analysis.config import load_config
from re_polar.mcts.analysis.loader import build_groups, load_menu_crosschecks
from re_polar.mcts.analysis.registry import get_insight, insight, list_insights

D = 36  # matches qwen3_8b, used throughout as the test model


# --------------------------------------------------------------------------- #
# Synthetic sample fixtures -- shaped exactly like merged_mcts_samples.json,
# small enough to reason about by hand.
# --------------------------------------------------------------------------- #
def identity_path():
    return list(range(D))


def skip_path(layers):
    return [i for i in range(D) if i not in layers]


def repeat_path(start, times=2):
    path = []
    for i in range(D):
        path.extend([i] * times if i == start else [i])
    return path


def make_sample(qid, difficulty, base_solved, extra_valid=None, extra_invalid=None):
    valid = [identity_path()] if base_solved else list(extra_valid or [])
    invalid = list(extra_invalid or [])
    return {
        "question": f"q {qid}",
        "gt_ans": "1",
        "final_valid_transitions": valid,
        "final_invalid_transitions": invalid,
        "initial_transition_metric": 1.0 if base_solved else 0.0,
        "sample_info": {
            "query_id": qid,
            "difficulty": difficulty,
            "domain": "Algebra" if int(qid[-1]) % 2 == 0 else "Geometry",
        },
    }


@pytest.fixture
def synthetic_samples():
    """5 questions: 2 base-solved, 2 rescued (skip + repeat programs), 1 unsolved."""
    return [
        make_sample("q0", 1, True),
        make_sample("q1", 1, True),
        make_sample(
            "q2", 1, False, extra_valid=[skip_path([5, 6])], extra_invalid=[skip_path([1, 2, 3])]
        ),
        make_sample("q3", 1, False, extra_valid=[skip_path([5, 6]), repeat_path(10)]),
        make_sample("q4", 1, False),  # unsolved: no valid program at all
    ]


@pytest.fixture
def groups(synthetic_samples):
    return {"diff1": synthetic_samples}


# --------------------------------------------------------------------------- #
# segments.py
# --------------------------------------------------------------------------- #
def test_decode_round_trips_identity_and_edits():
    for path in (identity_path(), skip_path([5, 6]), repeat_path(10)):
        prog = segments.decode(path, D)
        assert prog.to_layer_path() == path


def test_edit_signature_identity_is_empty():
    prog = segments.decode(identity_path(), D)
    assert segments.edit_signature(prog) == ()
    assert segments.sig_str(()) == "identity"


def test_edit_signature_nonempty_for_skip():
    prog = segments.decode(skip_path([5, 6]), D)
    sig = segments.edit_signature(prog)
    assert sig != ()
    assert "identity" not in segments.sig_str(sig)


def test_entropy_bits_bounds():
    assert segments.entropy_bits([]) == 0.0
    assert segments.entropy_bits([5]) == 0.0  # single outcome, no uncertainty
    assert segments.entropy_bits([1, 1, 1, 1]) == 2.0  # 4 equally likely outcomes


def test_gini_bounds():
    assert segments.gini([]) == 0.0
    assert segments.gini([5, 5, 5, 5]) == pytest.approx(0.0)  # perfectly even
    g = segments.gini([100, 1, 1, 1])
    assert 0.0 < g <= 1.0


def test_topk_coverage_share_monotonic_nondecreasing():
    counts = [50, 30, 10, 5, 5]
    out = segments.topk_coverage_share(counts, sum(counts), ks=(1, 2, 3, 5))
    vals = [out[str(k)] for k in (1, 2, 3, 5)]
    assert vals == sorted(vals)
    assert vals[-1] == pytest.approx(1.0)


def test_op_class_all_four():
    assert segments.op_class(identity_path(), D) == "identity"
    assert segments.op_class(skip_path([5]), D) == "skip-only"
    assert segments.op_class(repeat_path(10), D) == "repeat-only"
    both = skip_path([5])
    both[9] = both[8]  # crude way to introduce a repeat alongside the skip
    assert segments.op_class(both, D) == "both"


def test_shortest_valid_prefers_shorter_then_fewer_edits():
    paths = [tuple(repeat_path(10)), tuple(skip_path([5, 6]))]
    chosen = segments.shortest_valid(paths, D)
    assert chosen == tuple(skip_path([5, 6]))  # shorter executed length wins


def test_layer_freq_counts_correct_layers():
    freq = segments.layer_freq([skip_path([5, 6])], "skip", D)
    assert freq[5] == 1 and freq[6] == 1
    assert sum(freq) == 2


# --------------------------------------------------------------------------- #
# registry.py
# --------------------------------------------------------------------------- #
def test_registry_has_all_nine_insights():
    import re_polar.mcts.analysis.insights  # noqa: F401 -- populates the registry

    ids = {spec.id for spec in list_insights()}
    assert ids == {
        "menu_concentration",
        "op_mix",
        "segmentation_structure",
        "rescue_breakdown",
        "op_class_coverage",
        "menu_size_distribution",
        "dismemberment",
        "layer_ops_heatmap",
        "shorter_than_identity",
    }


def test_registry_unknown_id_raises_with_known_list():
    import re_polar.mcts.analysis.insights  # noqa: F401

    with pytest.raises(KeyError, match="unknown insight id"):
        get_insight("does_not_exist")


def test_registry_duplicate_id_raises():
    @insight("test_dup_zzz", "t", "table")
    def _f(groups, num_layers):
        return {}

    with pytest.raises(ValueError, match="duplicate insight id"):

        @insight("test_dup_zzz", "t2", "table")
        def _g(groups, num_layers):
            return {}


def test_registry_rejects_unknown_chart_kind():
    with pytest.raises(ValueError, match="unknown chart_kind"):

        @insight("test_bad_chart_zzz", "t", "not_a_real_kind")
        def _f(groups, num_layers):
            return {}


# --------------------------------------------------------------------------- #
# insights.py -- sanity on each registered insight's output shape/invariants
# --------------------------------------------------------------------------- #
def test_menu_concentration_shape_and_invariants(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    result = get_insight("menu_concentration").fn(groups, D)
    row = result["table"]["diff1"]
    assert row["solved"] == 4  # q0,q1,q2,q3 solved; q4 unsolved
    assert 0.0 <= row["menu_share"] <= 1.0
    curve = result["coverage_curve"]["diff1"]
    assert curve == sorted(curve)  # cumulative coverage is non-decreasing
    if curve:
        assert curve[-1] <= 1.0 + 1e-9


def test_rescue_breakdown_counts_sum_to_n(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    result = get_insight("rescue_breakdown").fn(groups, D)
    row = result["table"]["diff1"]
    assert row["baseline_solved"] + row["rescued"] + row["unsolved"] == row["n"] == 5
    assert row["baseline_solved"] == 2
    assert row["rescued"] == 2
    assert row["unsolved"] == 1


def test_op_class_coverage_denominators(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    result = get_insight("op_class_coverage").fn(groups, D)
    row = result["table"]["diff1"]
    assert row["n_solved"] == 4
    assert row["n_all"] == 5
    for lens in ("solved", "all"):
        for v in row[lens].values():
            assert 0.0 <= v <= 1.0 + 1e-9


def test_menu_size_distribution_shape(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    result = get_insight("menu_size_distribution").fn(groups, D)
    row = result["boxplot"]["diff1"]
    assert row["min"] <= row["q1"] <= row["median"] <= row["q3"] <= row["max"]
    assert row["n"] == 5


def test_dismemberment_matrix_dims(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    result = get_insight("dismemberment").fn(groups, D)
    M = result["diff1"]["shortest_valid"]["skip"]
    assert len(M) == 4
    assert all(len(row) == D for row in M)
    # "edit" = skip UNION repeat -- a segment is always exactly one of the
    # two, so edit is their element-wise sum.
    edit = result["diff1"]["shortest_valid"]["edit"]
    repeat = result["diff1"]["shortest_valid"]["repeat"]
    for r in range(4):
        for c in range(D):
            assert edit[r][c] == M[r][c] + repeat[r][c]


def test_op_mix_and_segmentation_structure_run_clean(groups):
    import re_polar.mcts.analysis.insights  # noqa: F401

    op_result = get_insight("op_mix").fn(groups, D)
    assert "diff1" in op_result["op_distribution"]
    seg_result = get_insight("segmentation_structure").fn(groups, D)
    assert seg_result["table"]["diff1"]["n_distinct_edit_signatures"] >= 1


# --------------------------------------------------------------------------- #
# loader.py + config.py
# --------------------------------------------------------------------------- #
def test_build_groups_by_key(tmp_path, synthetic_samples):
    path = tmp_path / "diff1.json"
    path.write_text(json.dumps({"samples": synthetic_samples}))
    groups = build_groups([{"label": "diff1", "path": str(path)}], group_by="difficulty")
    assert set(groups) == {"1"}
    assert len(groups["1"]) == 5


def test_build_groups_by_source_label_when_no_group_by(tmp_path, synthetic_samples):
    path = tmp_path / "diff1.json"
    path.write_text(json.dumps({"samples": synthetic_samples}))
    groups = build_groups([{"label": "my_label", "path": str(path)}], group_by=None)
    assert set(groups) == {"my_label"}


# --------------------------------------------------------------------------- #
# loader.load_menu_crosschecks -- auto-detection of menu-crossexec-run output
# --------------------------------------------------------------------------- #
def _write_crossexec_file(path, model="qwen3_8b", difficulties=(1,)):
    data = {
        "model": model,
        "k": 50,
        "per_difficulty": [
            {
                "difficulty": d,
                "n_train": 1250,
                "k_requested": 50,
                "k_actual": 50,
                "n_reused_cells": 0,
                "n_new_cells": 62500,
                "real_topk_coverage_curve": [
                    {"k": 1, "real_coverage": 0.3},
                    {"k": 50, "real_coverage": 0.8},
                ],
                "final_real_coverage": 0.8,
                "per_program": [],
            }
            for d in difficulties
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_load_menu_crosschecks_missing_dir_returns_empty(tmp_path):
    assert load_menu_crosschecks(tmp_path / "does_not_exist", "qwen3_8b") == {}


def test_load_menu_crosschecks_finds_per_model_diff_files(tmp_path):
    _write_crossexec_file(tmp_path / "qwen3_8b" / "diff1.json", difficulties=(1,))
    _write_crossexec_file(tmp_path / "qwen3_8b" / "diff2.json", difficulties=(2,))
    out = load_menu_crosschecks(tmp_path, "qwen3_8b")
    assert set(out) == {"1", "2"}
    assert out["1"]["final_real_coverage"] == 0.8


def test_load_menu_crosschecks_skips_wrong_model(tmp_path):
    _write_crossexec_file(tmp_path / "other_model" / "diff1.json", model="other_model")
    assert load_menu_crosschecks(tmp_path, "qwen3_8b") == {}


def test_load_menu_crosschecks_ignores_unparseable_file(tmp_path):
    (tmp_path / "qwen3_8b").mkdir(parents=True)
    (tmp_path / "qwen3_8b" / "diff_broken.json").write_text("{not json")
    _write_crossexec_file(tmp_path / "qwen3_8b" / "diff1.json", difficulties=(1,))
    out = load_menu_crosschecks(tmp_path, "qwen3_8b")
    assert set(out) == {"1"}


# --------------------------------------------------------------------------- #
# loader._with_excl_identity_curve, via load_menu_crosschecks: the "excluding
# identity" real-coverage curve computed by pure post-processing of
# `per_program` (dropping rank-0 identity, re-accumulating OR-coverage over
# the rest). `_write_crossexec_file` above writes an empty `per_program: []`
# (matching an older file shape), so those existing tests exercise the
# "nothing to compute from, leave fields absent" path already.
# --------------------------------------------------------------------------- #
def _write_crossexec_file_with_per_program(path, model="qwen3_8b", difficulty=1):
    identity_path = [
        0,
        1,
        2,
    ]  # D=3, matches synthetic_samples' D (see make_sample/segments helpers)
    programs = [
        {
            "path": identity_path,
            "rewards": [1, 0, 0, 0],
            "provenance": "identity",
        },  # rank 0 (dropped)
        {"path": [0, 2], "rewards": [0, 1, 0, 0], "provenance": "p1"},  # rank 1
        {"path": [1, 2], "rewards": [0, 0, 1, 1], "provenance": "p2"},  # rank 2
    ]
    data = {
        "model": model,
        "k": 3,
        "per_difficulty": [
            {
                "difficulty": difficulty,
                "n_train": 4,
                "n_pool": 4,
                "k_requested": 3,
                "k_actual": 3,
                "real_topk_coverage_curve": [
                    {"k": 1, "real_coverage": 0.25},
                    {"k": 2, "real_coverage": 0.5},
                    {"k": 3, "real_coverage": 1.0},
                ],
                "final_real_coverage": 1.0,
                "per_program": programs,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_load_menu_crosschecks_computes_excl_identity_curve(tmp_path):
    _write_crossexec_file_with_per_program(tmp_path / "qwen3_8b" / "diff1.json")
    out = load_menu_crosschecks(tmp_path, "qwen3_8b")
    # identity (rank 0) dropped; k=1 -> just p1 (covers qid1, 1/4); k=2 -> p1|p2 (qid1,2,3, 3/4)
    assert out["1"]["real_topk_coverage_curve_excl_identity"] == [
        {"k": 1, "real_coverage": 0.25},
        {"k": 2, "real_coverage": 0.75},
    ]
    assert out["1"]["final_real_coverage_excl_identity"] == 0.75
    # identity-included fields untouched by the augmentation
    assert out["1"]["final_real_coverage"] == 1.0


def test_load_menu_crosschecks_raises_if_rank0_program_is_not_identity(tmp_path):
    entry = {
        "difficulty": 1,
        "n_pool": 2,
        "per_program": [
            {"path": [0, 2], "rewards": [1, 0], "provenance": "not_identity_but_ranked_first"},
            {"path": [0, 1, 2], "rewards": [0, 1], "provenance": "identity_but_not_ranked_first"},
        ],
    }
    data = {"model": "qwen3_8b", "k": 2, "per_difficulty": [entry]}
    (tmp_path / "qwen3_8b").mkdir(parents=True)
    (tmp_path / "qwen3_8b" / "diff_bad.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity program"):
        load_menu_crosschecks(tmp_path, "qwen3_8b")


def test_load_config_valid(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": "qwen3_8b",
                "sources": [{"label": "a", "path": "x.json"}],
                "insights": ["rescue_breakdown"],
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.model == "qwen3_8b"
    assert cfg.title == "Analysis report"  # default


@pytest.mark.parametrize("missing_key", ["model", "sources", "insights"])
def test_load_config_missing_required_key_raises(tmp_path, missing_key):
    raw = {
        "model": "qwen3_8b",
        "sources": [{"label": "a", "path": "x.json"}],
        "insights": ["rescue_breakdown"],
    }
    del raw[missing_key]
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing required key"):
        load_config(cfg_path)


def test_load_config_crosscheck_dir_defaults_to_none(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": "qwen3_8b",
                "sources": [{"label": "a", "path": "x.json"}],
                "insights": ["rescue_breakdown"],
            }
        )
    )
    assert load_config(cfg_path).crosscheck_dir is None


def test_load_config_resolves_crosscheck_dir_relative_to_config_file(tmp_path):
    cfg_path = tmp_path / "sub" / "cfg.json"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        json.dumps(
            {
                "model": "qwen3_8b",
                "sources": [{"label": "a", "path": "x.json"}],
                "insights": ["rescue_breakdown"],
                "crosscheck_dir": "../crossexec",
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.crosscheck_dir == str((tmp_path / "crossexec").resolve())


def test_load_config_empty_insights_raises(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": "qwen3_8b",
                "sources": [{"label": "a", "path": "x.json"}],
                "insights": [],
            }
        )
    )
    with pytest.raises(ValueError, match="insights"):
        load_config(cfg_path)
