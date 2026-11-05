"""PoLar split (re_polar.datasets.polar_split) and its router folds (re_polar.router.fold_data)."""

from re_polar.datasets.polar_split import EXPECTED, SPLIT_TOTALS, split_sizes
from re_polar.router.fold_data import fold_qids


def test_split_sizes_reproduce_their_stated_totals_and_test_sizes():
    sizes = {lvl: split_sizes(n) for lvl, n in EXPECTED.items()}
    assert tuple(sum(s[i] for s in sizes.values()) for i in range(3)) == SPLIT_TOTALS
    assert [sizes[lvl][2] for lvl in sorted(sizes)] == [141, 337, 395, 384, 394]


def test_every_question_is_test_exactly_once_and_splits_disjoint():
    qids = [f"q{i}" for i in range(565)]
    seen = []
    for k in range(4):
        tr, va, te = fold_qids(qids, k)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == 565 and len(va) == round(565 / 8)
        seen += te
    assert sorted(seen) == sorted(qids)


def test_folds_depend_only_on_the_question_set_and_seed():
    qids = [f"q{i}" for i in range(100)]
    assert fold_qids(qids, 1) == fold_qids(list(reversed(qids)), 1)
    assert fold_qids(qids, 1, seed=0) != fold_qids(qids, 1, seed=1)
