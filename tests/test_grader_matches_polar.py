"""Regression tests for real, deliberately-kept-not-fixed behaviours of the
vendored dart-math grader (`re_polar/vendor/dart_math/eval.py`, see NOTICE.md),
each pinned so a change to the grader is noticed. The first:
`EvaluatorMath.extract_ans` always speculates via `EvaluatorBase`'s own
default (`strict_extract=False`), never actually consulting
`self.strict_extract`, so passing `True` changes nothing.

Kept as a locked-in assertion, not a bug we're fixing here, since PoLar's
own upstream grader has the identical behavior and matching it exactly is
what makes this a faithful reproduction. If this test fails, the vendored
grader differs from PoLar's. Grading in this repo does not depend on the bug:
every grading path goes through `re_polar/core/grader.py::_safe_grade`, which
scores text without a `\\boxed{}` as 0 before extraction, so the last-number
fallback is never reached.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_strict_extract_true_and_false_give_identical_extraction():
    sys.path.insert(0, str(REPO_ROOT))
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    # no \boxed{}, no explicit marker phrase -- only reachable via the
    # "speculate from the last number" fallback, which is exactly the branch
    # strict_extract is supposed to disable.
    text = "Let me work through this. First, 3 times 4 is 12. Then 12 plus 5 is 17."

    strict = EvaluatorMath(strict_extract=True)
    lenient = EvaluatorMath(strict_extract=False)

    assert strict.extract_ans(text) == lenient.extract_ans(text) == "17", (
        "EvaluatorMath.extract_ans honours strict_extract: the vendored grader differs "
        "from PoLar's copy, re-check re_polar/vendor/dart_math/eval.py against it"
    )


def test_caret_is_xor_known_bug():
    """Documented grader bug kept for fidelity with PoLar's identical grader: `^` between integers
    is Python XOR in EvaluatorMath.eq step 3.1 (see re_polar/core/grader.py). If this fails, the
    vendored grader no longer grades like PoLar's, and grades from before and after the change are
    not comparable."""
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    ev = EvaluatorMath(strict_extract=True)
    assert ev.eq("21", "16^5")
    assert ev.eq("1048576", "16^5")


def test_marker_after_box_overrides_box_known_quirk():
    """Documented grader quirk kept for fidelity with PoLar's identical grader: a single box
    is not cut, so a "Therefore"/"####"/"# Answer" after it makes the grader take the text
    after the marker instead of the box (see re_polar/mcts/rewards.py::
    truncate_after_first_boxed). Runs the real reward path: cut, then _safe_grade."""
    from re_polar.core.grader import _safe_grade
    from re_polar.mcts.rewards import truncate_after_first_boxed
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    ev = EvaluatorMath(strict_extract=True)

    def grade(text):
        return _safe_grade(ev, "17", truncate_after_first_boxed(text))

    assert grade("\\boxed{17}. Then 99")
    for marker in ("Therefore", "####", "# Answer"):
        assert not grade(f"\\boxed{{17}}. {marker} the answer is 99"), marker
    # two spans are cut after the first, so the marker never reaches the grader
    assert grade("\\boxed{17}. Therefore \\boxed{ANSWER}")
