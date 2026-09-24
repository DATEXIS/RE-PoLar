"""Router inference + evaluation -- pure logic on CPU, no model/GPU/download.

The graded pipeline is exercised with a STUB reward_fn (so no target model runs),
and the decode path is driven by a PolarRouter built with ``embed_dim=`` and fed
random token embeddings (so Qwen3-Embedding is never downloaded). Importing
``re_polar.router.infer`` must stay torch-free at its top level; these tests only
touch torch through the router itself.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from re_polar.core import Op, Program, Segment, is_valid
from re_polar.router import PolarRouter
from re_polar.router.infer import (
    assert_roundtrips,
    decode_programs,
    grade_router,
    grade_router_topk,
    group_by_program,
    load_test_split,
    resolve_difficulties,
    summarize,
)
from re_polar.router.train import program_from_layer_path

D = 8  # small target-model layer count


# --------------------------------------------------------------------------- #
# small program fixtures (distinct executed layer-paths)
# --------------------------------------------------------------------------- #
def _p_keep() -> Program:  # path [0..7], len 8, no skip/repeat
    return Program.identity(D)


def _p_skip() -> Program:  # path [0,1,2,3], len 4, has SKIP
    return Program(D, [Segment(0, 4, Op.KEEP), Segment(4, 8, Op.SKIP)])


def _p_repeat() -> Program:  # path [0..3,4,5,6,7,4,5,6,7], len 12, has REPEAT
    return Program(D, [Segment(0, 4, Op.KEEP), Segment(4, 8, Op.REPEAT, {"times": 2})])


# --------------------------------------------------------------------------- #
# resolve_difficulties / load_test_split
# --------------------------------------------------------------------------- #
def test_resolve_difficulties_defaults_and_all():
    assert resolve_difficulties(None) == [1, 2, 3, 4, 5]
    assert resolve_difficulties([]) == [1, 2, 3, 4, 5]
    assert resolve_difficulties(["all"]) == [1, 2, 3, 4, 5]
    assert resolve_difficulties(["2", "all"]) == [1, 2, 3, 4, 5]  # 'all' anywhere wins


def test_resolve_difficulties_dedup_and_sort():
    assert resolve_difficulties(["3", "1", "1"]) == [1, 3]
    assert resolve_difficulties([5, 2]) == [2, 5]


def test_load_test_split_and_limit(tmp_path):
    (tmp_path / "diff3").mkdir()
    data = [{"query_id": "a", "question": "q?", "gt_ans": "1"}]
    (tmp_path / "diff3" / "test.json").write_text(json.dumps(data))
    assert load_test_split(tmp_path, 3) == data

    (tmp_path / "diff1").mkdir()
    many = [{"query_id": str(i), "question": f"q{i}", "gt_ans": str(i)} for i in range(5)]
    (tmp_path / "diff1" / "test.json").write_text(json.dumps(many))
    assert len(load_test_split(tmp_path, 1, limit=2)) == 2


# --------------------------------------------------------------------------- #
# grouping + grading (stub reward_fn, no target model)
# --------------------------------------------------------------------------- #
class _RecordingReward:
    """Stub reward_fn: looks each question's reward up in a dict; records calls."""

    def __init__(self, per_question):
        self.per_question = per_question
        self.calls = []  # (program_path, questions, gts) per invocation

    def __call__(self, program, questions, gt_answers):
        self.calls.append((tuple(program.to_layer_path()), list(questions), list(gt_answers)))
        return [self.per_question[q] for q in questions]


def test_group_by_program_merges_equal_paths_across_distinct_objects():
    # Same executed path, different segmentation objects -> one group.
    a1 = Program.identity(D)
    a2 = Program(D, [Segment(0, 2, Op.KEEP), Segment(2, 4, Op.KEEP), Segment(4, 8, Op.KEEP)])
    assert a1.to_layer_path() == a2.to_layer_path()
    groups = group_by_program([a1, _p_skip(), a2])
    assert len(groups) == 2  # {identity-path, skip-path}
    assert groups[tuple(a1.to_layer_path())] == [0, 2]


def test_grade_router_one_call_per_distinct_program_and_scatters():
    programs = [_p_keep(), _p_skip(), _p_keep(), _p_skip(), _p_keep()]
    questions = [f"q{i}" for i in range(5)]
    gts = [f"g{i}" for i in range(5)]
    per_q = {"q0": 1.0, "q1": 0.0, "q2": 0.0, "q3": 1.0, "q4": 1.0}
    reward = _RecordingReward(per_q)

    out = grade_router(programs, questions, gts, reward)

    assert out == [1.0, 0.0, 0.0, 1.0, 1.0]  # scattered to original positions
    assert len(reward.calls) == 2  # two DISTINCT programs -> two calls
    keep_call = next(c for c in reward.calls if c[0] == tuple(_p_keep().to_layer_path()))
    assert keep_call[1] == ["q0", "q2", "q4"]  # only the keep-group questions...
    assert keep_call[2] == ["g0", "g2", "g4"]  # ...and their gts, in order
    skip_call = next(c for c in reward.calls if c[0] == tuple(_p_skip().to_layer_path()))
    assert skip_call[1] == ["q1", "q3"]


def test_grade_router_rejects_bad_reward_length():
    programs = [_p_keep(), _p_keep()]
    bad = lambda program, qs, gs: [1.0]  # returns fewer than the group size
    with pytest.raises(ValueError):
        grade_router(programs, ["q0", "q1"], ["g0", "g1"], bad)


class _PathReward:
    """Stub reward_fn keyed on (executed path, question) -> 0/1; records calls."""

    def __init__(self, table):
        self.table = table  # {(path_tuple, question): reward}
        self.calls = []

    def __call__(self, program, questions, gt_answers):
        key = tuple(program.to_layer_path())
        self.calls.append((key, list(questions)))
        return [self.table.get((key, q), 0.0) for q in questions]


def test_grade_router_topk_any_solve_and_program_major_batching():
    keep = tuple(_p_keep().to_layer_path())
    skip = tuple(_p_skip().to_layer_path())
    rep = tuple(_p_repeat().to_layer_path())
    topk = [
        [_p_keep(), _p_skip()],  # q0: top-1 keep FAILS, skip SOLVES -> pass@1 0, pass@k 1
        [_p_keep()],  # q1: keep SOLVES                   -> pass@1 1, pass@k 1
        [_p_repeat(), _p_skip()],  # q2: neither solves                -> pass@1 0, pass@k 0
    ]
    table = {
        (keep, "q0"): 0.0,
        (skip, "q0"): 1.0,
        (keep, "q1"): 1.0,
        (rep, "q2"): 0.0,
        (skip, "q2"): 0.0,
    }
    reward = _PathReward(table)
    rewards_at_k, rewards_at_1, chosen = grade_router_topk(
        topk, ["q0", "q1", "q2"], ["g0", "g1", "g2"], reward
    )

    assert rewards_at_k == [1.0, 1.0, 0.0]
    assert rewards_at_1 == [0.0, 1.0, 0.0]
    assert chosen[0].to_layer_path() == list(skip)  # the solving candidate reported
    # program-major: each DISTINCT path graded exactly once (skip shared by q0,q2)
    assert len(reward.calls) == 3
    skip_call = next(c for c in reward.calls if c[0] == skip)
    assert skip_call[1] == ["q0", "q2"]


# --------------------------------------------------------------------------- #
# summarize metrics
# --------------------------------------------------------------------------- #
def test_summarize_metrics():
    programs = [_p_keep(), _p_skip(), _p_repeat()]  # exec lens 8, 4, 12
    rewards = [1.0, 1.0, 0.0]  # router_acc = 2/3
    identity_rewards = [1.0, 0.0, 0.0]  # identity_acc = 1/3

    s = summarize(rewards, programs, identity_rewards, D)

    assert s["n"] == 3
    assert s["router_acc"] == pytest.approx(2 / 3)
    assert s["identity_acc"] == pytest.approx(1 / 3)
    assert s["delta"] == pytest.approx(1 / 3)
    assert s["mean_executed_len"] == pytest.approx((8 + 4 + 12) / 3)
    assert s["mean_identity_len"] == float(D)
    assert s["frac_programs_with_skip"] == pytest.approx(1 / 3)  # only _p_skip
    assert s["frac_programs_with_repeat"] == pytest.approx(1 / 3)  # only _p_repeat
    assert s["recurrence_rate"] == pytest.approx(1 / 3)


def test_summarize_handles_empty():
    s = summarize([], [], [], D)
    assert s["n"] == 0
    assert s["router_acc"] == 0.0
    assert s["identity_acc"] == 0.0
    assert s["mean_executed_len"] == 0.0
    assert s["recurrence_rate"] == 0.0


# --------------------------------------------------------------------------- #
# decode path end-to-end (embed_dim router, no encoder download)
# --------------------------------------------------------------------------- #
EMBED = 16
DM = 32
T = 7


def _build_router(num_layers=12):
    torch.manual_seed(0)
    return PolarRouter(
        num_layers=num_layers, embed_dim=EMBED, d_model=DM, nheads=4, n_layer_blocks=2
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_decode_programs_valid_and_roundtrip(seed):
    Dr = 12
    router = _build_router(Dr)
    router.eval()
    assert router.encoder is None  # embed_dim path: nothing downloaded

    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(4, T, EMBED, generator=g)
    mask = torch.zeros(4, T, dtype=torch.bool)
    mask[:, -2:] = True  # pad the last two token positions

    seg_logits, op_logits = router(token_hidden_states=hidden, key_padding_mask=mask)
    programs = decode_programs(router, seg_logits, op_logits)

    assert len(programs) == 4
    for p in programs:
        assert isinstance(p, Program)
        assert p.num_layers == Dr
        assert is_valid(p)  # contiguous full cover, no all-skip
        reparsed = program_from_layer_path(p.to_layer_path(), Dr)
        assert reparsed.to_layer_path() == p.to_layer_path()

    # the standing-invariant helper agrees (does not raise)
    assert_roundtrips(programs, Dr)
