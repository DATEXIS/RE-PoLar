"""Torch-free grading worker for GenerationReward's process pool.

Isolated in its own module (no torch / engine imports) so Pebble's SPAWN
workers stay lightweight, they import only this plus the vendored sympy
evaluator, not the whole model stack.

Why a subprocess at all: a layer-mutilated model occasionally emits an
answer that is tiny to write but
astronomically expensive to evaluate symbolically, e.g. `10^10^10`, an integer
with ~10^10 digits. sympy then tries to materialize it and allocates tens of GB
in well under the 5 s time guard, so the kernel OOM-kills the whole job before
any timeout can fire. Grading inside a memory-capped worker turns that into
'incorrect' + a logged sample instead of a dead job.
"""

import resource

_WORKER_EV = None


def _safe_grade(evaluator, ref: str, resp_text: str) -> bool:
    """Grade a generation against gt under PoLar's eval protocol (eval.py:185).

    BOXED GATE: a generation with no ``\\boxed{}`` is wrong (return False) and,
    crucially, is never handed to sympy, so a collapsed program that echoes a
    giant expression as free text can't explode the grader. This is both the
    paper's protocol AND the primary defense against the grading blow-up
    (a real incident found ~55% of the observed explosion answers were
    unboxed babble; the gate turns every one of them into an instant 0 with
    no symbolic work).

    A boxed-but-still-pathological answer (giant expression *inside* the box)
    can still reach sympy; that residual case is contained by the surrounding
    process pool's memory + time caps, not here.

    ``evaluator`` must be strict-extract (see _grade_worker_init) so only the
    boxed span is parsed, never a last-number fallback over the whole text."""
    if "oxed{" not in resp_text:  # matches PoLar eval.py: `if "oxed{" not in answer_part`
        return False
    try:
        ans = evaluator.extract_ans(resp_text)
        return bool(ans) and bool(evaluator.eq(ref, ans))
    except Exception:
        return False


def _grade_worker_init(mem_bytes: int) -> None:
    """Pebble worker initializer: build the evaluator once, then cap this
    worker's address space so a runaway symbolic allocation dies here (as
    MemoryError, or a kernel kill of just this worker) instead of taking the
    parent job down. Cap AFTER imports so the sympy import itself isn't starved."""
    global _WORKER_EV
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    # strict_extract=True: only the \boxed{} span is parsed (PoLar eval.py uses
    # EvaluatorMathBatch(strict_extract=True)). No last-number fallback over the
    # whole generation -> collapsed babble can't feed a giant expression to sympy.
    _WORKER_EV = EvaluatorMath(strict_extract=True)
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except (ValueError, OSError):
        pass  # best-effort; Pebble still isolates a worker that OOMs the cgroup


def _grade_worker(ref: str, resp_text: str) -> bool:
    return _safe_grade(_WORKER_EV, ref, resp_text)
