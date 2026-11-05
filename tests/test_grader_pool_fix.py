"""Grading pool: a per-answer timeout is contained (score 0, logged, no pool reset), resets reap
the old workers, and an answer is never scored 0 without being graded."""

import json
import multiprocessing as mp

import pytest

from re_polar.core import Program
from re_polar.mcts.rewards import GenerationReward, _is_pool_break

BOMB = " \\boxed{9^{9^{9^{9}}}}"  # times out (or hits the memory cap) in grading


class _Tok:
    padding_side = "left"


class _Eng:
    tokenizer = _Tok()
    device = "cpu"


class _Exe:
    engine = _Eng()


def _children():
    return {c.pid for c in mp.active_children()}  # other tests' pools may still be alive


def _reward(tmp_path, **kw):
    return GenerationReward(
        _Exe(),
        fail_log_path=str(tmp_path / "fails.jsonl"),
        difficulty=1,
        grade_timeout_s=1,
        grade_mem_bytes=512 * 1024**2,
        grade_workers=2,
        **kw,
    )


def test_is_pool_break_excludes_contained_failures():
    from concurrent.futures.process import BrokenProcessPool

    from pebble import ProcessExpired

    assert _is_pool_break(BrokenProcessPool("x"))
    assert not _is_pool_break(TimeoutError())  # an OSError subclass
    assert not _is_pool_break(ProcessExpired("Abnormal termination", code=-9, pid=1))
    assert not _is_pool_break(ValueError())


def test_timeouts_never_reset_or_leak_and_never_go_unresolved(tmp_path):
    before = _children()
    r = _reward(tmp_path)
    resets = []
    orig = r._reset_pool
    r._reset_pool = lambda: (resets.append(1), orig())
    prog = Program.identity(28)
    children = []
    n_batches = 6
    for _ in range(n_batches):
        out = r._grade_batch(
            ["2"] * 4, [BOMB, " \\boxed{2}", " \\boxed{3}", " \\boxed{2}"], ["q"] * 4, prog
        )
        assert out == [0.0, 1.0, 0.0, 1.0]
        children.append(len(_children() - before))
    orig()
    fails = [json.loads(x)["failure"] for x in (tmp_path / "fails.jsonl").read_text().splitlines()]
    assert resets == []  # a timeout is contained, never a pool reset
    assert max(children) <= 2  # grade_workers=2, constant
    assert len(fails) == n_batches and set(fails) <= {"TimeoutError", "ProcessExpired"}
    assert _children() <= before  # _reset_pool joins the workers


def test_reset_pool_joins_workers(tmp_path):
    before = _children()
    r = _reward(tmp_path)
    for _ in range(3):
        assert r._grade_batch(["2"], [" \\boxed{2}"], ["q"], Program.identity(28)) == [1.0]
        r._reset_pool()
        assert _children() <= before


def test_fresh_pool_that_cannot_schedule_crashes(tmp_path):
    r = _reward(tmp_path)
    new = r._new_pool

    def dead_pool(workers):
        p = new(workers)
        p.stop()
        p.join()
        return p

    r._new_pool = dead_pool  # every pool, incl. the rebuild, refuses schedule()
    with pytest.raises(RuntimeError):
        r._grade_batch(["2"], [" \\boxed{2}"], ["q"], Program.identity(28))
    assert not (tmp_path / "fails.jsonl").exists()  # nothing scored 0 ungraded


def test_prompt_variant_pilot_pool_contains_timeouts_and_joins(monkeypatch):
    import re_polar.mcts.rewards as rewards
    from analysis.error_analysis.prompt_variant_pilot import grade_all_pooled

    monkeypatch.setattr(rewards, "GRADE_TIMEOUT_S", 1)
    monkeypatch.setattr(rewards, "GRADE_MEM_BYTES", 512 * 1024**2)
    before = _children()
    texts = [BOMB, " \\boxed{2}", " \\boxed{3}", " \\boxed{2}"]
    assert grade_all_pooled(["2"] * 4, texts, workers=2) == [False, True, False, True]
    assert _children() <= before
