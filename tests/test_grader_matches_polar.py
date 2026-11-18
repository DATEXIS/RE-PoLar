"""Regression test for a real, deliberately-kept-not-fixed bug in the
vendored dart-math grader (`re_polar/vendor/dart_math/eval.py`, see NOTICE.md):
`EvaluatorMath.extract_ans` always speculates via `EvaluatorBase`'s own
default (`strict_extract=False`), never actually consulting
`self.strict_extract`, so passing `True` changes nothing.

Kept as a locked-in assertion, not a bug we're fixing here, since PoLar's
own upstream grader has the identical behavior and matching it exactly is
what makes this a faithful reproduction. If this test ever fails, the bug
has been fixed (upstream or locally) and `re_polar/core/grader.py` (which
grades with `strict_extract=True`, the actual MCTS reward path) needs to be
re-checked against anything that grades with `strict_extract=False`.
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
        "EvaluatorMath.extract_ans's strict_extract bug appears to have been "
        "fixed (upstream or locally); re-check whether anything relying on "
        "strict_extract=False behaving like True (e.g. analysis/error_analysis/"
        "prompt_variant_pilot.py's grader='ours', which never passes strict_extract) "
        "needs to change to match re_polar/core/grader.py's strict_extract=True path, "
        "now that the two could behave differently"
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
