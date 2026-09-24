import json

from analysis.structural_analysis.reversion_robustness import (
    aggregate,
    classify_reversions,
    load_by_query,
)

D = 36


def _identity_path():
    return list(range(D))


def _skip_path(start, length):
    return [l for l in range(D) if not (start <= l < start + length)]


def _repeat_path(start, length, times=2):
    return (
        list(range(start))
        + list(range(start, start + length)) * times
        + list(range(start + length, D))
    )


def test_not_eligible_when_identity_already_correct():
    records = [
        {"query_id": "q1", "path": _identity_path(), "reward": 1.0},
        {"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0},
    ]
    res = classify_reversions(records, D)
    assert res["eligible"] is False
    assert res["reversions"] == []


def test_not_eligible_when_no_identity_record_logged():
    records = [{"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0}]
    res = classify_reversions(records, D)
    assert res["eligible"] is False


def test_reversion_pair_marks_break_when_keep_side_fails():
    # identity wrong; skip(8,4) is a RESCUE program (reward 1); reverting that
    # one segment back to keep -- same 9x4-aligned boundary grid, only the op
    # at that one segment differs -- fails (reward 0) => "broke".
    records = [
        {"query_id": "q1", "path": _identity_path(), "reward": 0.0},
        {"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0},
    ]
    res = classify_reversions(records, D)
    assert res["eligible"] is True
    assert len(res["reversions"]) == 1
    assert res["reversions"][0]["broke"] is True
    assert res["reversions"][0]["edit_side_path"] == tuple(_skip_path(8, 4))


def test_reversion_pair_no_break_when_keep_side_also_correct():
    # Two non-identity RESCUE programs sharing a boundary grid, differing at
    # exactly one segment: skip(4,4) alone (keep at [8,12)) vs skip(4,4)+
    # skip(8,4) (skip at [8,12)). The keep side (skip(4,4) alone) is also
    # reward 1 -- reverting [8,12) back to keep doesn't break the rescue.
    records = [
        {"query_id": "q1", "path": _identity_path(), "reward": 0.0},
        {"query_id": "q1", "path": _skip_path(4, 4), "reward": 1.0},  # RESCUE, keep at [8,12)
        {
            "query_id": "q1",
            "path": [l for l in range(D) if not (4 <= l < 8) and not (8 <= l < 12)],
            "reward": 1.0,
        },
        # ^ skips BOTH [4,8) and [8,12) -- shares skip(4,4)'s boundary grid,
        # differs only at segment [8,12): keep vs skip. Both reward 1 => no break.
    ]
    res = classify_reversions(records, D)
    assert res["eligible"] is True
    assert any(rev["broke"] is False for rev in res["reversions"])


def test_repeat_skip_pair_is_not_a_reversion_neither_side_keep():
    records = [
        {"query_id": "q1", "path": _identity_path(), "reward": 0.0},
        {"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0},
        {"query_id": "q1", "path": _repeat_path(8, 4), "reward": 1.0},
    ]
    res = classify_reversions(records, D)
    # only the identity<->skip(8,4) and identity<->repeat(8,4) pairs qualify
    # (each has a keep side); skip(8,4)<->repeat(8,4) has no keep side and
    # does not contribute a reversion.
    assert len(res["reversions"]) == 2


def test_edit_side_reward_zero_is_excluded():
    # skip(8,4) does NOT solve the question (reward 0) -- not a confirmed
    # RESCUE program, so its keep-reversion is not counted at all.
    records = [
        {"query_id": "q1", "path": _identity_path(), "reward": 0.0},
        {"query_id": "q1", "path": _skip_path(8, 4), "reward": 0.0},
    ]
    res = classify_reversions(records, D)
    assert res["reversions"] == []


def test_aggregate_computes_break_rate_and_distinct_program_count():
    all_results = [
        {
            "eligible": True,
            "reversions": [
                {"broke": True, "edit_side_path": (1, 2, 3)},
                {"broke": True, "edit_side_path": (1, 2, 3)},  # same program, 2nd segment reverted
                {"broke": False, "edit_side_path": (4, 5, 6)},
            ],
        },
        {"eligible": True, "reversions": []},
        {"eligible": False, "reversions": []},
    ]
    agg = aggregate(all_results)
    assert agg["n_queries_eligible"] == 2
    assert agg["n_reversions_tested"] == 3
    assert agg["n_broken"] == 2
    assert abs(agg["break_rate"] - 2 / 3) < 1e-9
    assert agg["n_distinct_rescue_programs_covered"] == 2  # (1,2,3) and (4,5,6), per-query-indexed


def test_aggregate_handles_zero_reversions():
    agg = aggregate([{"eligible": True, "reversions": []}])
    assert agg["n_reversions_tested"] == 0
    assert agg["break_rate"] is None


def test_load_by_query_raw_cache_log(tmp_path):
    f = tmp_path / "raw.jsonl"
    f.write_text(
        json.dumps({"query_id": "q1", "path": _identity_path(), "reward": 0.0})
        + "\n"
        + json.dumps({"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0})
        + "\n"
        + json.dumps({"query_id": "q2", "path": _identity_path(), "reward": 1.0})
        + "\n"
    )
    by_query = load_by_query(str(f))
    assert set(by_query) == {"q1", "q2"}
    assert len(by_query["q1"]) == 2
    assert len(by_query["q2"]) == 1


def test_load_by_query_released_merged_mcts_samples(tmp_path):
    # Same underlying data as the raw-cache-log test above, but shaped as the
    # RELEASED merged_mcts_samples.json format (search_trajectory nested per
    # sample, tagged by sample_info.query_id) -- must produce an equivalent
    # by_query mapping ({query_id: [{"path","reward"}, ...]}). Note:
    # search_trajectory here deliberately does NOT itself contain an
    # identity-path entry (matches real data) -- the identity record must
    # come from initial_transition_metric instead.
    f = tmp_path / "merged_mcts_samples.json"
    data = {
        "samples": [
            {
                "question": "q1 text",
                "gt_ans": "1",
                "final_valid_transitions": [_skip_path(8, 4)],
                "final_invalid_transitions": [],
                "initial_transition_metric": 0.0,
                "sample_info": {"query_id": "q1"},
                "search_trajectory": [
                    {"path": _skip_path(8, 4), "parent_path": _identity_path(), "reward": 1.0},
                ],
            },
            {
                "question": "q2 text",
                "gt_ans": "2",
                "final_valid_transitions": [_identity_path()],
                "final_invalid_transitions": [],
                "initial_transition_metric": 1.0,
                "sample_info": {"query_id": "q2"},
                "search_trajectory": [
                    {"path": _skip_path(4, 4), "parent_path": _identity_path(), "reward": 0.0},
                ],
            },
            {
                # no search_trajectory attached -- identity record (from
                # initial_transition_metric) must still be added, not crash.
                "question": "q3 text",
                "gt_ans": "3",
                "final_valid_transitions": [],
                "final_invalid_transitions": [],
                "initial_transition_metric": 0.0,
                "sample_info": {"query_id": "q3"},
            },
        ]
    }
    f.write_text(json.dumps(data))
    by_query = load_by_query(str(f), num_layers=D)
    assert set(by_query) == {"q1", "q2", "q3"}
    assert len(by_query["q1"]) == 2  # synthetic identity record + 1 trajectory entry
    assert len(by_query["q2"]) == 2
    assert len(by_query["q3"]) == 1  # identity record only
    assert by_query["q1"][0]["path"] == _identity_path()
    assert by_query["q1"][0]["reward"] == 0.0
    assert "parent_path" not in by_query["q1"][0]  # trimmed to just path/reward
    assert by_query["q3"][0]["path"] == _identity_path()
    assert by_query["q3"][0]["reward"] == 0.0


def test_load_by_query_both_formats_give_identical_classification(tmp_path):
    # End-to-end equivalence: same underlying search data through both file
    # formats must produce the exact same classify_reversions() result. The
    # merged format's search_trajectory omits the identity-path entry (as
    # real data does) -- initial_transition_metric supplies it instead.
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps({"query_id": "q1", "path": _identity_path(), "reward": 0.0})
        + "\n"
        + json.dumps({"query_id": "q1", "path": _skip_path(8, 4), "reward": 1.0})
        + "\n"
    )
    merged = tmp_path / "merged_mcts_samples.json"
    merged.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "question": "q",
                        "gt_ans": "1",
                        "final_valid_transitions": [_skip_path(8, 4)],
                        "final_invalid_transitions": [],
                        "initial_transition_metric": 0.0,
                        "sample_info": {"query_id": "q1"},
                        "search_trajectory": [
                            {
                                "path": _skip_path(8, 4),
                                "parent_path": _identity_path(),
                                "reward": 1.0,
                            },
                        ],
                    }
                ]
            }
        )
    )
    by_query_raw = load_by_query(str(raw))
    by_query_merged = load_by_query(str(merged), num_layers=D)
    res_raw = classify_reversions(by_query_raw["q1"], D)
    res_merged = classify_reversions(by_query_merged["q1"], D)
    assert res_raw == res_merged
