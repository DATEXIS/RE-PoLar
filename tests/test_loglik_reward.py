"""LogLikReward: one log-likelihood forward pass, no generation/grading.

Uses a tiny in-process fake model + tokenizer (real torch tensors, CPU-only,
no downloaded weights) so the test exercises the REAL
re_polar.core.mmlu_pro_domain_eval.run_mmlu_pro_domains scoring path end-to-end.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from re_polar.mcts.rewards import LogLikReward
from re_polar.core import Program

D = 36
VOCAB = 20


class _FakeTokenizer:
    """encode(): letter -> fixed single-token id (A=1, B=2, ... matches
    re_polar.core.mmlu_pro_scoring._choice_token_ids's ascending letter order)."""

    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[100, 101, 102]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text, add_special_tokens=False):
        return [ord(text.strip()) - ord("A") + 1]


class _TinyModel:
    """Deterministic fake LM: the n-th row EVER SEEN (across however many
    batches run_mmlu_pro_domains's internal chunking issues) gets argmax
    choice logit `target_ids[n]` -- a cursor, not a per-call index, so this
    stays correct however batch_size splits the item list into calls."""

    def __init__(self, target_ids):
        self.target_ids = target_ids
        self._cursor = 0

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, attention_mask, use_cache=False, logits_to_keep=1):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, logits_to_keep, VOCAB)
        for row in range(batch):
            logits[row, -1, self.target_ids[self._cursor]] = 10.0
            self._cursor += 1
        return SimpleNamespace(logits=logits)


class _FakeExecutor:
    def __init__(self, model, tokenizer):
        self.engine = SimpleNamespace(tokenizer=tokenizer, device="cpu")
        self._model = model
        self.applied = []

    @contextmanager
    def apply(self, program):
        self.applied.append(program)
        yield self._model


def test_loglik_reward_scores_correct_and_incorrect_in_order():
    # sample0: predicted letter index 1 ('B', token id 2); gt=1 -> correct
    # sample1: predicted letter index 2 ('C', token id 3); gt=0 -> incorrect
    model = _TinyModel(target_ids=[2, 3])
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8)

    questions = [
        {"question": "2+2=?", "options": ["1", "4", "3", "5"], "category": "math"},
        {
            "question": "capital of France?",
            "options": ["Berlin", "Madrid", "Paris", "Rome"],
            "category": "history",
        },
    ]
    gt_answers = [1, 0]

    program = Program.identity(D)
    out = reward_fn(program, questions, gt_answers)

    assert out == [1.0, 0.0]
    assert exe.applied == [program]  # program was actually applied via executor.apply


def test_loglik_reward_defaults_missing_category_to_empty_string():
    model = _TinyModel(target_ids=[1])  # predicts index 0
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8)

    questions = [{"question": "q", "options": ["a", "b"]}]  # no "category" key
    out = reward_fn(Program.identity(D), questions, [0])

    assert out == [1.0]


def test_loglik_reward_empty_input_returns_empty_without_applying_program():
    exe = _FakeExecutor(_TinyModel(target_ids=[]), _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8)

    out = reward_fn(Program.identity(D), [], [])

    assert out == []
    assert exe.applied == []  # no wasted executor.apply for an empty batch


def test_loglik_reward_batches_smaller_than_batch_size_still_order_correctly():
    # 3 samples, batch_size=1 forces 3 separate forward-pass batches -> order
    # must still line up 1:1 with the input list (regression guard for any
    # off-by-one in how per_prompt scores get zipped back to `questions`).
    model = _TinyModel(target_ids=[1, 2, 1])  # predicts idx0, idx1, idx0
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=1)

    questions = [
        {"question": "q0", "options": ["a", "b"], "category": "math"},
        {"question": "q1", "options": ["a", "b"], "category": "math"},
        {"question": "q2", "options": ["a", "b"], "category": "math"},
    ]
    gt_answers = [0, 0, 1]  # correct, incorrect, incorrect

    out = reward_fn(Program.identity(D), questions, gt_answers)

    assert out == [1.0, 0.0, 0.0]


def test_loglik_reward_default_writes_no_probs_log(tmp_path):
    """probs_log_path=None (the default): behavior identical to before this
    feature existed -- same scores, and nothing written anywhere."""
    model = _TinyModel(target_ids=[2, 3])
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8)  # no probs_log_path

    questions = [
        {
            "question": "2+2=?",
            "options": ["1", "4", "3", "5"],
            "category": "math",
            "query_id": "q0",
        },
        {
            "question": "capital of France?",
            "options": ["Berlin", "Madrid", "Paris", "Rome"],
            "category": "history",
            "query_id": "q1",
        },
    ]
    out = reward_fn(Program.identity(D), questions, [1, 0])

    assert out == [1.0, 0.0]  # unchanged from the equivalent test above
    assert reward_fn.probs_log_path is None
    assert not (tmp_path / "probs.jsonl").exists()


def test_loglik_reward_logs_full_probability_distribution_when_requested(tmp_path):
    log_path = tmp_path / "probs.jsonl"
    model = _TinyModel(target_ids=[2, 3])
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8, probs_log_path=str(log_path))

    questions = [
        {
            "question": "2+2=?",
            "options": ["1", "4", "3", "5"],
            "category": "math",
            "query_id": "q0",
        },
        {
            "question": "capital of France?",
            "options": ["Berlin", "Madrid", "Paris", "Rome"],
            "category": "history",
            "query_id": "q1",
        },
    ]
    program = Program.identity(D)
    out = reward_fn(program, questions, [1, 0])

    # scoring is UNCHANGED by turning probs logging on
    assert out == [1.0, 0.0]

    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert len(lines) == 2
    assert [l["query_id"] for l in lines] == ["q0", "q1"]
    assert [l["path"] for l in lines] == [program.to_layer_path()] * 2
    assert [l["predicted"] for l in lines] == [1, 2]  # target_ids=[2,3] -> letter idx 1, 2
    assert [l["score"] for l in lines] == [1.0, 0.0]
    for l, n_options in zip(lines, [4, 4]):
        probs = l["probs"]
        assert len(probs) == n_options
        assert abs(sum(probs) - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)


def test_loglik_reward_probs_log_appends_across_calls(tmp_path):
    log_path = tmp_path / "probs.jsonl"
    model = _TinyModel(target_ids=[1, 1])
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8, probs_log_path=str(log_path))

    q = [{"question": "q", "options": ["a", "b"], "query_id": "only"}]
    reward_fn(Program.identity(D), q, [0])
    reward_fn(Program.identity(D), q, [0])

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2  # second call APPENDS, does not overwrite


def test_loglik_reward_probs_log_falls_back_to_question_text_without_query_id(tmp_path):
    log_path = tmp_path / "probs.jsonl"
    model = _TinyModel(target_ids=[1])
    exe = _FakeExecutor(model, _FakeTokenizer())
    reward_fn = LogLikReward(exe, batch_size=8, probs_log_path=str(log_path))

    q = [{"question": "no query_id here", "options": ["a", "b"]}]  # no "query_id" key
    reward_fn(Program.identity(D), q, [0])

    line = json.loads(log_path.read_text().splitlines()[0])
    assert line["query_id"] == "no query_id here"
