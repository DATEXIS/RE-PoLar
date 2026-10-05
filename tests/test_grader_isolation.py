"""Grading workers must stay torch-free, and a memory-cap hit must be logged.

GenerationReward grades in spawned workers capped with RLIMIT_AS (Linux
enforces it, macOS doesn't). Anything a worker imports counts against that
cap: with torch+transformers loaded a worker sat at ~7.2 GiB virtual size
before grading anything, leaving <1 GiB of the 8 GiB production cap, and caps
<=4 GiB graded `\\boxed{1/2}` vs `0.5` as 0. Two import paths put torch there:
the package `__init__`s (`re_polar.core`, `re_polar.mcts`), and spawn
re-importing the parent's main module (`python -m re_polar.mcts.run_search_*`)
in every worker. The MemoryError the cap raises was also swallowed as a plain
wrong answer, with no fail-log entry.
"""

import json
import subprocess
import sys

import pytest

# modules a spawned grading worker imports: the grader itself, plus the main
# module of every entry point that builds a GenerationReward pool
WORKER_IMPORTS = [
    "re_polar.core.grader",
    "re_polar.mcts.run_search_dart_math",
    "re_polar.mcts.run_search_mmlu_pro_domains",
    "re_polar.router.infer",
]


@pytest.mark.parametrize("module", WORKER_IMPORTS)
def test_worker_imports_do_not_pull_torch(module):
    # fresh interpreter: this pytest process has torch loaded already
    code = (
        f"import sys, importlib; importlib.import_module({module!r}); "
        "print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"{module} imports {out.stdout.strip()}"


def _fake_reward(**kw):
    from re_polar.mcts.rewards import GenerationReward

    class _Tok:
        padding_side = "left"

    class _Eng:
        tokenizer = _Tok()
        device = "cpu"

    class _Exe:
        engine = _Eng()

    return GenerationReward(_Exe(), grade_workers=1, **kw)


def test_spawned_grading_worker_is_torch_free():
    r = _fake_reward()
    try:
        # `eval` is a picklable builtin, so the probe itself imports nothing
        fut = r._ensure_pool().schedule(
            eval,
            args=(
                "sorted(m for m in ('torch', 'transformers') if m in __import__('sys').modules)",
            ),
        )
        assert fut.result(timeout=120) == []
    finally:
        r._reset_pool()


def test_safe_grade_reraises_memory_error():
    from re_polar.core.grader import _safe_grade

    class _OOMEvaluator:
        def extract_ans(self, text):
            return "1/2"

        def eq(self, ref, ans):
            raise MemoryError

    with pytest.raises(MemoryError):
        _safe_grade(_OOMEvaluator(), "0.5", "\\boxed{1/2}")


def test_grade_batch_logs_memory_cap_hit(tmp_path, monkeypatch):
    """A worker MemoryError is scored 0 AND written to the fail log."""
    import re_polar.mcts.rewards as R

    class _InlinePool:  # run the task in-process; the pool plumbing is tested elsewhere
        def schedule(self, fn, args, timeout):
            class _Fut:
                def result(self_inner):
                    raise MemoryError

            return _Fut()

    log = tmp_path / "fails.jsonl"
    r = _fake_reward(fail_log_path=str(log), difficulty=1)
    monkeypatch.setattr(r, "_ensure_pool", lambda: _InlinePool())
    from re_polar.core import Program

    out = r._grade_batch(["0.5"], ["\\boxed{1/2}"], ["q"], Program.identity(36))
    assert out == [0.0]
    assert [json.loads(l)["failure"] for l in log.read_text().splitlines()] == ["MemoryError"]
