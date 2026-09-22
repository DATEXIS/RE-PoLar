"""Tests for re_polar/mcts/analysis/polar_comparison.py -- checks against PoLar's
own diagnostic claims, computed as plain dicts (no rendering)."""
import pytest

D = 36


def identity_path():
    return list(range(D))


def skip_path(skip_layers):
    return [l for l in range(D) if l not in skip_layers]


def repeat_path(layer, times=2):
    path = []
    for l in range(D):
        path.append(l)
        if l == layer:
            path.extend([layer] * (times - 1))
    return path


def skip_and_repeat_path(skip_layers, repeat_layer):
    """A single path that's genuinely "both" op-class: skip some layers AND
    repeat one, inline (not just concatenated -- a repeat must interleave
    ascending runs to parse, [10,10] appended after a full pass doesn't)."""
    path = []
    for l in range(D):
        if l in skip_layers:
            continue
        path.append(l)
        if l == repeat_layer:
            path.append(l)
    return path


def make_sample(qid, difficulty, base_solved, extra_valid=None, extra_invalid=None):
    valid = list(extra_valid or [])
    invalid = list(extra_invalid or [])
    return {
        "question": f"q {qid}", "gt_ans": "1",
        "final_valid_transitions": valid, "final_invalid_transitions": invalid,
        "initial_transition_metric": 1.0 if base_solved else 0.0,
        "sample_info": {"query_id": qid, "difficulty": difficulty, "domain": "Algebra"},
    }


@pytest.fixture
def samples():
    """5 questions covering every branch: base-solved w/ no extra (q0),
    base-solved w/ a shorter skip alt (q1), rescued by skip-only (q2),
    rescued by repeat-only (q3), rescued needing "both" (q4), unsolved (q5)."""
    return [
        make_sample("q0", 1, True),
        make_sample("q1", 1, True, extra_valid=[skip_path([5, 6])]),
        make_sample("q2", 1, False, extra_valid=[skip_path([5, 6])],
                    extra_invalid=[skip_path([1, 2, 3])]),
        make_sample("q3", 1, False, extra_valid=[repeat_path(10)]),
        make_sample("q4", 1, False, extra_valid=[skip_and_repeat_path([5, 6], 10)]),
        make_sample("q5", 1, False),
    ]


@pytest.fixture
def groups(samples):
    return {"1": samples}


def test_compute_skip_loop_accuracy_matches_rescue_breakdown_semantics(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_skip_loop_accuracy
    t1 = compute_skip_loop_accuracy(groups, D)["1"]
    assert t1["n"] == 6
    # base: q0, q1 -> 2/6
    assert t1["base"] == pytest.approx(2 / 6 * 100)
    # skiploop (any valid or base): q0,q1,q2,q3,q4 -> 5/6 (q5 unsolved)
    assert t1["skiploop"] == pytest.approx(5 / 6 * 100)
    assert t1["gain"] == pytest.approx(t1["skiploop"] - t1["base"])
    # skip column >= base (skip-only-or-identity is a superset of identity-only)
    assert t1["skip"] >= t1["base"]
    assert t1["loop"] >= t1["base"]


def test_compute_accuracy_by_depth_budget_is_monotonic_and_bounded(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_accuracy_by_depth_budget, DEPTH_BUDGETS_PCT
    f3 = compute_accuracy_by_depth_budget(groups, D)["1"]
    assert f3["budgets"] == DEPTH_BUDGETS_PCT
    accs = f3["accuracy"]
    # accuracy(budget) must be non-decreasing as budget widens (inequality, not exact)
    for a, b in zip(accs, accs[1:]):
        assert b >= a - 1e-9
    assert all(0.0 <= a <= 1.0 for a in accs)


def test_compute_mean_executed_depth_splits_cc_and_wc(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_mean_executed_depth
    f4 = compute_mean_executed_depth(groups, D)["1"]
    assert f4["n_cc"] == 2  # q0, q1
    assert f4["n_wc"] == 3  # q2, q3, q4 (q5 unsolved, excluded from both)
    assert f4["cc_depth_pct"] <= 100.0
    assert f4["wc_unique_pct"] <= f4["wc_depth_pct"] + 1e-9  # unique layers <= total executed length


def test_compute_valid_coverage_by_recurrence_budget_is_monotonic(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_valid_coverage_by_recurrence_budget
    f5a = compute_valid_coverage_by_recurrence_budget(groups, D, max_r=8)
    curve = f5a["p_valid"]
    for a, b in zip(curve, curve[1:]):
        assert b >= a - 1e-9
    assert curve[-1] <= 1.0


def test_compute_recurrence_and_skip_requirement(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_recurrence_and_skip_requirement
    f5b = compute_recurrence_and_skip_requirement(groups, D)["1"]
    # q3 is repeat-only-rescued with no skip-only/identity alt -> requires recurrence
    # q2 is skip-only-rescued with no repeat-only/identity alt -> requires skip
    assert f5b["n_solved"] == 5  # q0-q4
    assert 0.0 <= f5b["p_require_recurrence"] <= 1.0
    assert 0.0 <= f5b["p_require_skip"] <= 1.0


def test_compute_accuracy_by_executed_depth_within_bucket_is_a_rate(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_accuracy_by_executed_depth
    f6 = compute_accuracy_by_executed_depth(groups, D)["1"]
    for pt in f6:
        assert 0.0 <= pt["accuracy"] <= 1.0
        assert pt["n"] > 0


def test_compute_segment_recurrence_distribution_fractions_sum_to_one(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_segment_recurrence_distribution
    f7b = compute_segment_recurrence_distribution(groups, D)
    assert f7b["n"] == 4  # q1, q2, q3, q4 have >=1 valid program (q0 has none recorded, q5 unsolved)
    total = f7b["0"] + f7b["1"] + f7b["2"]
    assert total == pytest.approx(1.0)


def test_compute_segment_length_distribution_sums_to_one(groups):
    from re_polar.mcts.analysis.polar_comparison import compute_segment_length_distribution
    f7a = compute_segment_length_distribution(groups, D)
    total = sum(f7a[k] for k in ("1", "2", "3", "4"))
    assert total == pytest.approx(1.0)
    assert f7a["n"] > 0
    assert 0.0 <= f7a["le2_pct"] <= 100.0
