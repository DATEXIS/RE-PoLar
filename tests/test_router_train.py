"""Router training loop: path parsing, multi-path targets, one train step.

All fast/CPU: the router is built with ``embed_dim=`` (no encoder download) and
examples carry their own synthetic ``token_hidden``, so the whole
supervision -> multi-path targets -> train -> checkpoint path is exercised
without touching Qwen3-Embedding.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, is_valid, validate_program
from re_polar.router import DEFAULT_OPS, PolarRouter
from re_polar.router.model import decode_topk
from re_polar.router.train import (
    Example,
    _make_lr_scheduler,
    build_examples,
    collate,
    compute_loss,
    encode_examples,
    evaluate_val_programs,
    load_checkpoint,
    load_supervision,
    load_supervision_many,
    path_to_polar_targets,
    program_from_layer_path,
    program_to_targets,
    save_checkpoint,
    split_samples_train_val,
    targets_from_path,
    train,
)

D = 8  # small target-model layer count
EMBED = 12  # synthetic encoder hidden size
T = 6  # question token count
DM = 32  # router d_model


def build_router(num_layers=D, n_ops=3, **kw):
    torch.manual_seed(0)
    return PolarRouter(
        num_layers=num_layers,
        n_ops=n_ops,
        embed_dim=EMBED,
        d_model=DM,
        nheads=4,
        n_layer_blocks=2,
        **kw,
    )


def synthetic_hidden(seed=0, tokens=T, embed=EMBED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, embed, generator=g)


# --------------------------------------------------------------------------- #
# path -> program parsing (round-trip)
# --------------------------------------------------------------------------- #
def _programs_for_roundtrip():
    return [
        Program.identity(D),
        Program(
            D,
            [
                Segment(0, 2, Op.KEEP),
                Segment(2, 4, Op.SKIP),
                Segment(4, 6, Op.REPEAT, {"times": 2}),
                Segment(6, 8, Op.KEEP),
            ],
        ),
        # skip in the middle
        Program(D, [Segment(0, 4, Op.KEEP), Segment(4, 6, Op.SKIP), Segment(6, 8, Op.KEEP)]),
        # leading skip
        Program(D, [Segment(0, 2, Op.SKIP), Segment(2, 6, Op.KEEP), Segment(6, 8, Op.KEEP)]),
        # size-4 repeat block
        Program(D, [Segment(0, 4, Op.KEEP), Segment(4, 8, Op.REPEAT, {"times": 2})]),
    ]


@pytest.mark.parametrize("program", _programs_for_roundtrip())
def test_program_from_layer_path_roundtrips(program):
    path = program.to_layer_path()
    parsed = program_from_layer_path(path, D)
    validate_program(parsed)
    assert parsed.to_layer_path() == path


def test_parse_distinguishes_repeat_interleaving():
    # [4,5,4,5] is one REPEAT[4,6); [4,4,5,5] is two size-1 REPEATs -- order matters.
    interleaved = program_from_layer_path([0, 1, 2, 3, 4, 5, 4, 5, 6, 7], D)
    assert any(s.op is Op.REPEAT and s.start == 4 and s.end == 6 for s in interleaved.segments)

    stacked = program_from_layer_path([0, 1, 2, 3, 4, 4, 5, 5, 6, 7], D)
    repeats = [s for s in stacked.segments if s.op is Op.REPEAT]
    assert all(len(s) == 1 for s in repeats)
    assert stacked.to_layer_path() == [0, 1, 2, 3, 4, 4, 5, 5, 6, 7]


def test_parse_keep_then_repeat_adjacency():
    # counts: 6,7 -> 2; 4,5 -> 1. KEEP[4,6) then REPEAT[6,8) -- greedy-ascending must not over-reach.
    parsed = program_from_layer_path([0, 1, 2, 3, 4, 5, 6, 7, 6, 7], D)
    assert parsed.to_layer_path() == [0, 1, 2, 3, 4, 5, 6, 7, 6, 7]
    assert any(s.op is Op.REPEAT and s.start == 6 and s.end == 8 for s in parsed.segments)


def test_parse_rejects_out_of_range_layer():
    with pytest.raises(ValueError):
        program_from_layer_path([0, 1, 99], D)


def test_parse_splits_long_keep_at_max_segment_len():
    parsed = program_from_layer_path(list(range(D)), D)  # identity
    assert all(len(s) <= MAX_SEGMENT_LEN for s in parsed.segments)
    assert all(s.op is Op.KEEP for s in parsed.segments)
    assert parsed.is_identity()


# --------------------------------------------------------------------------- #
# cap_keep=False ablation: only SKIP/REPEAT capped, KEEP merges
# --------------------------------------------------------------------------- #
def test_parse_cap_keep_false_merges_full_identity_into_one_segment():
    parsed = program_from_layer_path(list(range(D)), D, cap_keep=False)
    assert len(parsed.segments) == 1
    seg = parsed.segments[0]
    assert seg.op is Op.KEEP and seg.start == 0 and seg.end == D  # NOT capped
    assert parsed.is_identity()
    # This program is intentionally outside the strict grammar (KEEP > MAX_SEGMENT_LEN).
    assert not is_valid(parsed)
    with pytest.raises(ValueError):
        validate_program(parsed)


def test_parse_cap_keep_false_still_caps_skip_and_repeat():
    # layers 0-9 KEEP (10, > MAX_SEGMENT_LEN), 10-15 SKIP (6, > MAX_SEGMENT_LEN),
    # 16-19 KEEP (4). Only the KEEP run should escape the cap.
    num_layers = 20
    path = list(range(10)) + [16, 17, 18, 19]
    parsed = program_from_layer_path(path, num_layers, cap_keep=False)
    assert parsed.to_layer_path() == path

    keep_segs = [s for s in parsed.segments if s.op is Op.KEEP]
    skip_segs = [s for s in parsed.segments if s.op is Op.SKIP]
    assert len(keep_segs) == 2
    assert (keep_segs[0].start, keep_segs[0].end) == (0, 10)  # uncapped, len 10
    assert (keep_segs[1].start, keep_segs[1].end) == (16, 20)  # len 4

    # the 6-layer skip run must still be chunked into consecutive <= MAX_SEGMENT_LEN
    # pieces (two adjacent SKIP segments are fine -- this is the "allowed" case).
    assert all(len(s) <= MAX_SEGMENT_LEN for s in skip_segs)
    assert [s.start for s in skip_segs] == [10, 14]
    assert sum(len(s) for s in skip_segs) == 6


def test_parse_cap_keep_false_repeat_block_still_capped_at_four():
    # layer 0 KEEP; layers 1-2 SKIP; layers 3-6 REPEAT(x2, block len 4, at the cap);
    # layers 7-9 KEEP (merged, uncapped).
    num_layers = 10
    path = [0] + [3, 4, 5, 6, 3, 4, 5, 6] + [7, 8, 9]
    parsed = program_from_layer_path(path, num_layers, cap_keep=False)
    assert parsed.to_layer_path() == path

    repeat_segs = [s for s in parsed.segments if s.op is Op.REPEAT]
    assert len(repeat_segs) == 1
    assert (repeat_segs[0].start, repeat_segs[0].end) == (3, 7)
    assert len(repeat_segs[0]) == MAX_SEGMENT_LEN
    assert repeat_segs[0].times == 2

    keep_segs = [s for s in parsed.segments if s.op is Op.KEEP]
    assert (0, 1) in [(s.start, s.end) for s in keep_segs]
    assert (7, 10) in [(s.start, s.end) for s in keep_segs]  # uncapped merge, len 3 here anyway


# --------------------------------------------------------------------------- #
# strict_repeat_2x=True: reject any REPEAT with times != 2, matching PoLar's
# own parser exactly (their DP only ever tries chunk+chunk).
# --------------------------------------------------------------------------- #
def test_strict_repeat_2x_accepts_exactly_2x():
    path = [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]  # REPEAT[4,6) x2
    parsed = program_from_layer_path(path, D, strict_repeat_2x=True)
    assert parsed.to_layer_path() == path


def test_strict_repeat_2x_rejects_3x_and_up():
    # REPEAT[0,2) x3 -- valid under the lenient parser, rejected under
    # strict_repeat_2x (the current default; explicit False here to exercise
    # the lenient path).
    path = [0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    lenient = program_from_layer_path(path, D, strict_repeat_2x=False)
    rep = next(s for s in lenient.segments if s.op is Op.REPEAT)
    assert rep.times == 3
    with pytest.raises(ValueError):
        program_from_layer_path(path, D, strict_repeat_2x=True)


def test_build_examples_strict_repeat_2x_drops_whole_path_not_just_segment():
    keep = list(range(D))  # identity, no repeat -> unaffected
    repeat_3x = [0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 6, 7]  # REPEAT[4,6) x3 -- must be dropped whole
    sample = {
        "question": "q",
        "gt_ans": "0",
        "final_valid_transitions": [keep, repeat_3x],
        "final_invalid_transitions": [],
        "initial_transition_metric": 1.0,
    }
    lenient = build_examples([sample], D, strict_repeat_2x=False)
    strict = build_examples([sample], D, strict_repeat_2x=True)
    assert len(lenient) == 2  # both paths parse
    assert len(strict) == 1  # only identity survives; the 3x-repeat path is dropped entirely
    assert _reconstruct_program(strict[0]).to_layer_path() == keep


# --------------------------------------------------------------------------- #
# program -> targets
# --------------------------------------------------------------------------- #
def test_targets_match_program():
    program = Program(
        D,
        [
            Segment(0, 2, Op.KEEP),
            Segment(2, 4, Op.SKIP),
            Segment(4, 6, Op.REPEAT, {"times": 2}),
            Segment(6, 8, Op.KEEP),
        ],
    )
    seg_target, op_target, op_mask = program_to_targets(program)

    # boundaries at 0,2,4,6 only
    assert seg_target.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    assert op_mask.tolist() == [True, False, True, False, True, False, True, False]
    # ops at the starts: KEEP, SKIP, REPEAT, KEEP
    i_keep, i_skip, i_rep = (DEFAULT_OPS.index(o) for o in (Op.KEEP, Op.SKIP, Op.REPEAT))
    assert op_target[0].item() == i_keep
    assert op_target[2].item() == i_skip
    assert op_target[4].item() == i_rep
    assert op_target[6].item() == i_keep


def test_targets_from_path_matches_program_boundaries():
    program = _programs_for_roundtrip()[1]
    seg_a, op_a, mask_a = program_to_targets(program)
    seg_b, op_b, mask_b, parsed = targets_from_path(program.to_layer_path(), D)
    assert torch.equal(seg_a, seg_b)
    assert torch.equal(op_a, op_b)
    assert torch.equal(mask_a, mask_b)
    assert parsed.to_layer_path() == program.to_layer_path()


def test_seg_target_marks_every_segment_start():
    program = program_from_layer_path(list(range(D)), D)
    seg_target, _, op_mask = program_to_targets(program)
    starts = {s.start for s in program.segments}
    for i in range(D):
        assert (seg_target[i].item() == 1.0) == (i in starts)
        assert op_mask[i].item() == (i in starts)


def test_path_to_polar_targets_flip_and_ignore_index():
    # PoLar per-path target form: seg_flip[0]==0, op_labels==-100 off segment starts.
    program = Program(
        D,
        [
            Segment(0, 2, Op.KEEP),
            Segment(2, 4, Op.SKIP),
            Segment(4, 6, Op.REPEAT, {"times": 2}),
            Segment(6, 8, Op.KEEP),
        ],
    )
    seg_flip, op_labels, parsed = path_to_polar_targets(program.to_layer_path(), D)

    # boundaries at 0,2,4,6 -> flip drops layer 0 (always a start, carries no info)
    assert seg_flip.tolist() == [0, 0, 1, 0, 1, 0, 1, 0]
    # op_labels: op index at starts (incl. layer 0), -100 elsewhere
    i_keep, i_skip, i_rep = (DEFAULT_OPS.index(o) for o in (Op.KEEP, Op.SKIP, Op.REPEAT))
    assert op_labels.tolist() == [i_keep, -100, i_skip, -100, i_rep, -100, i_keep, -100]
    assert parsed.to_layer_path() == program.to_layer_path()


# --------------------------------------------------------------------------- #
# supervision loading / multi-path example building
# --------------------------------------------------------------------------- #
def _synthetic_samples():
    keep = list(range(D))  # identity path (D executed layers)
    skip_mid = [0, 1, 2, 3, 6, 7]  # SKIP[4,6) -- shorter
    repeat = [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]  # REPEAT[4,6) -- longer
    return [
        {
            "question": "q0",
            "gt_ans": "0",
            "final_valid_transitions": [keep, skip_mid, repeat],  # 3 valid paths
            "final_invalid_transitions": [],
            "initial_transition_metric": 1.0,
            "sample_info": {"difficulty": 1},
        },
        {
            "question": "q1",
            "gt_ans": "1",
            "final_valid_transitions": [keep],
            "final_invalid_transitions": [],
            "initial_transition_metric": 1.0,
            "sample_info": {"difficulty": 1},
        },
        {
            "question": "q2_no_valid",
            "gt_ans": "2",
            "final_valid_transitions": [],  # no valid program -> dropped
            "final_invalid_transitions": [[0, 1]],
            "initial_transition_metric": 0.0,
            "sample_info": {"difficulty": 1},
        },
    ]


def test_load_supervision_reads_samples(tmp_path):
    path = tmp_path / "merged_mcts_samples.json"
    path.write_text(json.dumps({"samples": _synthetic_samples()}))
    samples = load_supervision(path)
    assert len(samples) == 3


def _reconstruct_program(example) -> Program:
    """Reconstruct a program from an Example's seg_flip/op_labels (test helper)."""
    starts = [0] + [i for i in range(1, D) if example.seg_flip[i].item() == 1.0]
    segs = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else D
        op = DEFAULT_OPS[int(example.op_labels[start].item())]
        params = {"times": 2} if op is Op.REPEAT else {}
        segs.append(Segment(start, end, op, params))
    return Program(D, segs)


def test_build_examples_multi_path_one_example_per_valid_path(tmp_path):
    samples = _synthetic_samples()
    examples = build_examples(samples, D)
    # q0 -> 3 examples (one per valid path), q1 -> 1, q2 dropped (no valid). Total 4.
    assert len(examples) == 4
    q0 = [e for e in examples if e.question == "q0"]
    assert len(q0) == 3
    assert [e.question for e in examples] == ["q0", "q0", "q0", "q1"]

    # the 3 q0 examples cover exactly the 3 valid paths (via their targets).
    paths = {tuple(_reconstruct_program(e).to_layer_path()) for e in q0}
    assert paths == {
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 6, 7),
        (0, 1, 2, 3, 4, 5, 4, 5, 6, 7),
    }
    # path_len tracks the executed length of each path
    assert sorted(e.path_len for e in q0) == [6, 8, 10]


def test_build_examples_shortest_only_picks_min_length_path_per_sample():
    # q0 has 3 valid paths (lens 6, 8, 10) -> exactly 1 example, the len-6 one.
    # q1 has only the identity path (len D=8) -> that lone path, unchanged.
    # q2 has no valid path -> still dropped.
    examples = build_examples(_synthetic_samples(), D, shortest_only=True)
    assert len(examples) == 2
    by_q = {e.question: e for e in examples}
    assert set(by_q) == {"q0", "q1"}
    assert by_q["q0"].path_len == 6
    assert by_q["q1"].path_len == D


def test_build_examples_cap_keep_false_merges_keep_runs_only():
    # q1's only valid path is full identity (D=8, all KEEP). cap_keep=True (default)
    # chunks it into two 4-layer segments (one boundary, at layer 4); cap_keep=False
    # must merge it into a single D-layer KEEP segment (zero boundaries).
    capped = build_examples(_synthetic_samples(), D, cap_keep=True)
    uncapped = build_examples(_synthetic_samples(), D, cap_keep=False)
    q1_capped = next(e for e in capped if e.question == "q1")
    q1_uncapped = next(e for e in uncapped if e.question == "q1")

    assert q1_capped.seg_flip.sum().item() == 1  # boundary at layer 4
    assert q1_uncapped.seg_flip.sum().item() == 0  # no internal boundary at all

    prog_uncapped = _reconstruct_program(q1_uncapped)
    assert len(prog_uncapped.segments) == 1
    assert prog_uncapped.segments[0].op is Op.KEEP
    assert len(prog_uncapped.segments[0]) == D
    assert prog_uncapped.to_layer_path() == list(range(D))


def test_build_examples_shortest_only_breaks_ties_deterministically():
    tie_a = [0, 1, 2, 3, 6, 7]  # len 6
    tie_b = [0, 1, 4, 5, 6, 7]  # len 6, distinct path, same length
    longer = list(range(D))  # len D=8
    sample = {
        "question": "q",
        "gt_ans": "0",
        "final_valid_transitions": [tie_a, tie_b, longer],
        "final_invalid_transitions": [],
        "initial_transition_metric": 1.0,
    }
    a = build_examples([sample], D, shortest_only=True, seed=3)
    b = build_examples([sample], D, shortest_only=True, seed=3)
    assert len(a) == len(b) == 1
    assert a[0].path_len == b[0].path_len == 6
    assert _reconstruct_program(a[0]).to_layer_path() == _reconstruct_program(b[0]).to_layer_path()


def test_build_examples_caps_paths_per_sample_deterministically():
    keep = list(range(D))
    skip_mid = [0, 1, 2, 3, 6, 7]
    repeat = [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]
    sample = {
        "question": "q",
        "gt_ans": "0",
        "final_valid_transitions": [keep, skip_mid, repeat],
        "final_invalid_transitions": [],
        "initial_transition_metric": 1.0,
    }
    a = build_examples([sample], D, max_paths_per_sample=2, seed=7)
    b = build_examples([sample], D, max_paths_per_sample=2, seed=7)
    assert len(a) == len(b) == 2  # capped
    assert [e.path_len for e in a] == [e.path_len for e in b]  # deterministic


def test_build_examples_per_sample_weight_normalize():
    samples = _synthetic_samples()
    examples = build_examples(samples, D, per_sample_weight_normalize=True)
    q0 = [e for e in examples if e.question == "q0"]
    q1 = [e for e in examples if e.question == "q1"]
    assert all(abs(e.weight - 1.0 / 3.0) < 1e-9 for e in q0)  # 3 paths -> 1/3 each
    assert all(abs(e.weight - 1.0) < 1e-9 for e in q1)  # 1 path -> 1


def test_build_examples_reweights_original_path_only_when_shorter_exists():
    # q0 has identity (len D) + a strictly-shorter valid -> trigger; q1 has only
    # the identity path -> no shorter, no downweight.
    examples = build_examples(
        _synthetic_samples(), D, reweight_original_path=True, original_path_weight=0.3
    )
    q0 = {e.path_len: e.weight for e in examples if e.question == "q0"}
    q1 = {e.path_len: e.weight for e in examples if e.question == "q1"}
    assert abs(q0[D] - 0.3) < 1e-9  # identity path downweighted
    assert all(abs(w - 1.0) < 1e-9 for pl, w in q0.items() if pl != D)  # others full
    assert abs(q1[D] - 1.0) < 1e-9  # no shorter valid -> untouched
    # OFF by default: identity path keeps weight 1.0
    default = build_examples(_synthetic_samples(), D)
    assert all(abs(e.weight - 1.0) < 1e-9 for e in default if e.question == "q0")


def test_reweight_original_path_composes_with_per_sample_normalize():
    # PoLar order: assign base weight (0.3 for downweighted identity) THEN scale by 1/n.
    examples = build_examples(
        _synthetic_samples(),
        D,
        reweight_original_path=True,
        original_path_weight=0.3,
        per_sample_weight_normalize=True,
    )
    q0 = {e.path_len: e.weight for e in examples if e.question == "q0"}
    assert abs(q0[D] - 0.3 / 3.0) < 1e-9  # identity: 0.3 * (1/3)
    assert all(abs(w - 1.0 / 3.0) < 1e-9 for pl, w in q0.items() if pl != D)


def test_drop_original_path_hard_removes_identity_when_shorter_exists():
    # q0 has identity + a strictly-shorter valid -> identity DROPPED from targets;
    # q1 has only identity (no shorter) -> kept (drop only fires on the trigger).
    examples = build_examples(_synthetic_samples(), D, drop_original_path=True)
    q0_lens = [e.path_len for e in examples if e.question == "q0"]
    q1_lens = [e.path_len for e in examples if e.question == "q1"]
    assert D not in q0_lens  # identity (len D) removed for q0
    assert len(q0_lens) == 2  # skip_mid + repeat remain
    assert q1_lens == [D]  # q1's lone identity survives


def test_drop_original_path_ignored_when_reweight_on():
    # PoLar mutual exclusion: drop is skipped while reweight is set.
    examples = build_examples(
        _synthetic_samples(),
        D,
        reweight_original_path=True,
        original_path_weight=0.3,
        drop_original_path=True,
    )
    q0 = {e.path_len: e.weight for e in examples if e.question == "q0"}
    assert D in q0 and abs(q0[D] - 0.3) < 1e-9  # identity kept, just downweighted


def test_keep_original_prob_one_keeps_identity():
    # soft-drop with keep prob 1.0 == keep the identity path.
    examples = build_examples(_synthetic_samples(), D, keep_original_prob=1.0)
    q0_lens = [e.path_len for e in examples if e.question == "q0"]
    assert D in q0_lens


def test_lr_scheduler_warmup_then_cosine_decay():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = _make_lr_scheduler(opt, "cosine", warmup_steps=2, total_steps=10)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == 0.0 and lrs[1] < lrs[2]  # linear warmup ramps 0 -> peak
    assert abs(lrs[2] - 1.0) < 1e-9  # peak LR right after warmup
    assert lrs[2] > lrs[-1] and lrs[-1] < 1e-9  # cosine decays to ~0 at the end
    assert _make_lr_scheduler(opt, "none", 0, 10) is None  # flat -> no scheduler


class _FakeEncodeRouter:
    """Duck-typed stand-in: encode_examples only calls ``.encode_questions``."""

    def __init__(self):
        self.calls = []

    def encode_questions(self, questions):
        self.calls.append(list(questions))
        n = len(questions)
        return torch.randn(n, T, EMBED), torch.zeros(n, T, dtype=torch.bool)


def test_build_examples_shares_token_hidden_across_paths():
    examples = build_examples(_synthetic_samples(), D)  # q0 -> 3 paths, q1 -> 1
    fake = _FakeEncodeRouter()
    encode_examples(fake, examples, batch_size=16)

    # each UNIQUE question encoded once (2 questions here), not once per path
    encoded_questions = [q for call in fake.calls for q in call]
    assert sorted(encoded_questions) == ["q0", "q1"]

    q0 = [e for e in examples if e.question == "q0"]
    # all q0 path-examples share ONE token_hidden tensor object (encoded once)
    assert q0[0].token_hidden is q0[1].token_hidden is q0[2].token_hidden


def test_anti_original_flag_only_when_identity_invalid():
    identity = list(range(D))
    shorter = [0, 1, 2, 3, 6, 7]
    with_identity = {
        "question": "qa",
        "gt_ans": "0",
        "final_valid_transitions": [identity, shorter],
        "final_invalid_transitions": [],
    }
    without_identity = {
        "question": "qb",
        "gt_ans": "1",
        "final_valid_transitions": [shorter],
        "final_invalid_transitions": [],
    }
    examples = build_examples([with_identity, without_identity], D, anti_original=True)
    qa = [e for e in examples if e.question == "qa"]
    qb = [e for e in examples if e.question == "qb"]
    assert all(not e.anti_original_active for e in qa)  # identity IS valid -> off
    assert all(e.anti_original_active for e in qb)  # identity NOT valid -> on
    # OFF by default (anti_original not requested)
    default = build_examples([without_identity], D)
    assert all(not e.anti_original_active for e in default)


def test_load_supervision_many_concatenates_difficulty_files(tmp_path):
    # the router trains on all difficulties combined: 5 per-difficulty files -> one set
    paths = []
    for diff in range(1, 6):
        p = tmp_path / f"diff{diff}.json"
        s = dict(_synthetic_samples()[1])  # one valid sample, tag its question by diff
        s["question"] = f"d{diff}_q"
        p.write_text(json.dumps({"samples": [s]}))
        paths.append(p)

    combined = load_supervision_many(paths)
    assert len(combined) == 5
    assert [s["question"] for s in combined] == [f"d{d}_q" for d in range(1, 6)]  # order preserved
    # a lone path (str/Path) is accepted too and matches load_supervision
    assert load_supervision_many(paths[0]) == load_supervision(paths[0])


def test_split_samples_train_val_holds_out_last_fraction():
    samples = [
        {"question": f"q{i}", "final_valid_transitions": [list(range(D))]} for i in range(10)
    ]
    tr, va = split_samples_train_val(samples, val_frac=0.1)
    assert len(tr) == 9 and len(va) == 1
    assert va[0]["question"] == "q9"  # last fraction held out
    # val_frac 0 -> everything trains
    tr0, va0 = split_samples_train_val(samples, val_frac=0.0)
    assert len(tr0) == 10 and va0 == []


# --------------------------------------------------------------------------- #
# collate / one training step / multi-path loss
# --------------------------------------------------------------------------- #
def _encoded_examples(seed0=0, n=6):
    """n path-examples of ONE program, each with its own synthetic token_hidden."""
    program = Program(
        D,
        [
            Segment(0, 2, Op.KEEP),
            Segment(2, 4, Op.SKIP),
            Segment(4, 6, Op.REPEAT, {"times": 2}),
            Segment(6, 8, Op.KEEP),
        ],
    )
    seg_flip, op_labels, _ = path_to_polar_targets(program.to_layer_path(), D)
    examples = []
    for i in range(n):
        examples.append(
            Example(
                question=f"q{i}",
                seg_flip=seg_flip.clone(),
                op_labels=op_labels.clone(),
                path_len=len(program.to_layer_path()),
                token_hidden=synthetic_hidden(seed=seed0 + i),
            )
        )
    return examples


def _multi_path_batch():
    """A batch mixing several valid paths (varying #segments), sharing one question."""
    paths = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # identity
        [0, 1, 2, 3, 6, 7],  # SKIP[4,6)
        [0, 1, 2, 3, 4, 5, 4, 5, 6, 7],  # REPEAT[4,6)
    ]
    examples = []
    for i, path in enumerate(paths):
        seg_flip, op_labels, _ = path_to_polar_targets(path, D)
        examples.append(
            Example(
                question="q",
                seg_flip=seg_flip,
                op_labels=op_labels,
                path_len=len(path),
                token_hidden=synthetic_hidden(seed=100 + i),
            )
        )
    return examples


def test_collate_shapes_and_padding():
    examples = _encoded_examples()
    # give one example a shorter token sequence to force padding
    examples[0].token_hidden = synthetic_hidden(seed=99, tokens=T - 2)
    batch = collate(examples)
    assert batch["token_hidden"].shape == (6, T, EMBED)
    assert batch["key_padding_mask"].shape == (6, T)
    # the short example is padded on its last two positions
    assert batch["key_padding_mask"][0, -2:].all()
    assert not batch["key_padding_mask"][0, : T - 2].any()
    assert batch["seg_flip"].shape == (6, D)
    assert batch["op_labels"].shape == (6, D)
    assert batch["path_len"].shape == (6,)
    assert batch["weight"].shape == (6,)
    assert batch["anti_orig"].shape == (6,)


def test_plain_loss_runs_and_backprops_on_multi_path_batch():
    router = build_router()
    batch = collate(_multi_path_batch())
    loss, seg_loss, op_loss = compute_loss(router, batch)
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(seg_loss) and torch.isfinite(op_loss)


def test_seg_loss_ignores_layer0():
    # layer 0's seg target must not affect the (plain) seg/total loss -- it's excluded.
    router = build_router()
    router.eval()  # deterministic forward (no dropout) so we isolate the seg-flip change
    examples = _multi_path_batch()
    loss_a, seg_a, _ = compute_loss(router, collate(examples))

    for e in examples:  # flip layer-0 seg target for every example
        e.seg_flip = e.seg_flip.clone()
        e.seg_flip[0] = 1.0 - e.seg_flip[0]
    loss_b, seg_b, _ = compute_loss(router, collate(examples))

    assert torch.allclose(seg_a, seg_b)  # seg loss unchanged
    assert torch.allclose(loss_a, loss_b)  # so total unchanged


def test_anti_original_penalty_positive_only_for_invalid_samples():
    router = build_router()
    router.eval()  # deterministic forward so the only delta is the anti term
    examples = _multi_path_batch()
    # none flagged -> anti term contributes nothing beyond base loss
    base = collate(examples)
    loss_off, _, _ = compute_loss(router, base, anti_original_lambda=0.0)

    for e in examples:
        e.anti_original_active = True
    on = collate(examples)
    loss_on, _, _ = compute_loss(router, on, anti_original_lambda=1.0)
    loss_base_for_on, _, _ = compute_loss(router, on, anti_original_lambda=0.0)
    # with lambda>0 AND flagged samples, the penalty strictly raises the loss
    assert float(loss_on.detach()) > float(loss_base_for_on.detach())

    # flagged but lambda 0 -> no penalty (equals the unflagged base loss)
    assert abs(float(loss_off.detach()) - float(loss_base_for_on.detach())) < 1e-6


def test_op_class_weights_defaults_to_unweighted_and_reweights_op_loss():
    # The op-head class-weighting lever: default None must be exactly the
    # unweighted PoLar-faithful loss; a non-uniform weight must change the op
    # loss (up-weighting the rare SKIP/REPEAT classes) and still backprop.
    router = build_router()
    router.eval()  # deterministic forward so the only delta is the class weight
    batch = collate(_multi_path_batch())

    _, _, op_none = compute_loss(router, batch)  # default None
    uniform = torch.ones(router.n_ops)
    _, _, op_uniform = compute_loss(router, batch, op_class_weights=uniform)
    assert torch.allclose(op_none, op_uniform, atol=1e-6)  # None == all-ones

    w = torch.tensor([5.0, 0.2, 1.0])[: router.n_ops]
    loss_w, _, op_w = compute_loss(router, batch, op_class_weights=w)
    assert not torch.allclose(op_none, op_w)  # non-uniform weight changes op loss
    assert torch.isfinite(loss_w)
    loss_w.backward()  # still differentiable


def test_op_focal_gamma_reduces_to_ce_at_zero_and_reshapes_gradient():
    # Focal loss lever (DR.LLM-style, gamma=2.0): gamma=0 must equal plain CE;
    # gamma>0 must change the op loss (down-weighting easy/confident positions) and
    # still backprop finitely.
    router = build_router()
    router.eval()
    batch = collate(_multi_path_batch())

    _, _, op_ce = compute_loss(router, batch)
    _, _, op_g0 = compute_loss(router, batch, op_focal_gamma=0.0)
    assert torch.allclose(op_ce, op_g0, atol=1e-6)  # gamma=0 == plain CE

    loss_f, _, op_f = compute_loss(router, batch, op_focal_gamma=2.0)
    assert not torch.allclose(op_ce, op_f)  # focal reshapes the loss
    assert (op_f <= op_ce + 1e-6).all()  # (1-p_t)^gamma <= 1 -> focal <= CE
    assert torch.isfinite(loss_f)
    loss_f.backward()


def test_lenpref_reweights_shorter_paths():
    # polar_lenpref up-weights shorter paths; the loss differs from plain 'polar'.
    router = build_router()
    router.eval()  # deterministic forward so the delta is the reweighting, not dropout
    batch = collate(_multi_path_batch())
    plain, _, _ = compute_loss(router, batch, policy_mode="polar")
    lenpref, _, _ = compute_loss(router, batch, policy_mode="polar_lenpref", lenpref_beta=0.2)
    assert not torch.allclose(plain, lenpref)


def test_training_reduces_loss():
    router = build_router()
    examples = _encoded_examples()
    result = train(router, examples, epochs=60, lr=5e-3, batch_size=3, seed=1)
    assert len(result.train_losses) == 60
    assert result.val_losses == []  # no validation requested
    assert result.train_losses[-1] < result.train_losses[0]  # loss goes down
    assert result.train_losses[-1] < 0.5 * result.train_losses[0]  # substantially


def test_training_learns_to_decode_the_label():
    # After overfitting a single repeated program, decode should recover it.
    router = build_router()
    examples = _encoded_examples()
    train(router, examples, epochs=160, lr=5e-3, batch_size=6, seed=2)
    router.eval()
    batch = collate(examples[:1])
    seg_logits, op_logits = router(
        token_hidden_states=batch["token_hidden"], key_padding_mask=batch["key_padding_mask"]
    )
    prog = router.decode(seg_logits[0], op_logits[0])
    validate_program(prog)
    # KEEP[0,2] SKIP[2,4] REPEAT[4,6]x2 KEEP[6,8] -> [0,1, ,4,5,4,5, 6,7]
    assert prog.to_layer_path() == [0, 1, 4, 5, 4, 5, 6, 7]


# --------------------------------------------------------------------------- #
# validation + best-checkpoint selection
# --------------------------------------------------------------------------- #
def test_train_with_validation_selects_best_checkpoint():
    router = build_router()
    train_examples = _encoded_examples(seed0=0, n=6)
    val_examples = _encoded_examples(seed0=50, n=4)
    result = train(
        router, train_examples, val_examples=val_examples, epochs=15, lr=5e-3, batch_size=3, seed=5
    )
    assert len(result.val_losses) == 15
    assert result.best_epoch is not None
    assert result.best_state is not None
    # best epoch is the argmin of val losses
    assert result.best_epoch == min(range(15), key=lambda i: result.val_losses[i])
    # router weights equal the restored best-checkpoint head
    head = {k: v for k, v in router.state_dict().items() if not k.startswith("encoder.")}
    for k, v in result.best_state.items():
        assert torch.allclose(head[k].cpu(), v)


def test_train_no_validation_keeps_last_epoch():
    router = build_router()
    examples = _encoded_examples()
    result = train(router, examples, val_examples=None, epochs=8, lr=5e-3, batch_size=3, seed=6)
    assert result.val_losses == []
    assert result.best_epoch is None
    assert result.best_state is None  # last epoch kept, nothing restored


# --------------------------------------------------------------------------- #
# frozen encoder
# --------------------------------------------------------------------------- #
class _StubEncoderEMBED(torch.nn.Module):
    """Frozen stub encoder whose hidden size matches this file's EMBED."""

    class _Cfg:
        hidden_size = EMBED

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.proj = torch.nn.Linear(EMBED, EMBED)


def test_encoder_stays_frozen_through_training():
    router = PolarRouter(num_layers=D, encoder=_StubEncoderEMBED(), d_model=DM)
    assert router.embed_dim == EMBED
    before = {n: p.clone() for n, p in router.encoder.named_parameters()}
    # examples carry EMBED-dim hidden already (stub encoder also EMBED) -> forward works
    examples = _encoded_examples()
    train(router, examples, epochs=5, lr=5e-3, batch_size=3, seed=3)
    assert all(not p.requires_grad for p in router.encoder.parameters())
    assert not router.encoder.training
    for n, p in router.encoder.named_parameters():
        assert torch.equal(before[n], p), f"frozen encoder param {n} changed"


# --------------------------------------------------------------------------- #
# checkpoint save / reload (stays loadable by re_polar.router.infer.load_checkpoint)
# --------------------------------------------------------------------------- #
def test_checkpoint_saves_and_reloads(tmp_path):
    router = build_router()
    examples = _encoded_examples()
    train(router, examples, epochs=10, lr=5e-3, batch_size=3, seed=4)
    router.eval()

    out = tmp_path / "router.pt"
    save_checkpoint(router, out, meta={"note": "test"})
    assert out.exists()

    reloaded = load_checkpoint(out)  # embed_dim path, no download (infer.py's default)
    assert reloaded.num_layers == D
    assert reloaded.embed_dim == EMBED
    assert reloaded.encoder is None

    # identical outputs on the same input (both in eval mode -> deterministic)
    batch = collate(examples[:2])
    with torch.no_grad():
        a = router(
            token_hidden_states=batch["token_hidden"], key_padding_mask=batch["key_padding_mask"]
        )
        b = reloaded(
            token_hidden_states=batch["token_hidden"], key_padding_mask=batch["key_padding_mask"]
        )
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert torch.allclose(a[1], b[1], atol=1e-6)


def test_checkpoint_excludes_encoder_weights(tmp_path):
    router = PolarRouter(num_layers=D, encoder=_StubEncoderEMBED(), d_model=DM)
    out = tmp_path / "router_with_encoder.pt"
    save_checkpoint(router, out)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert all(not k.startswith("encoder.") for k in payload["state_dict"])
    assert payload["meta"]["num_layers"] == D


# --------------------------------------------------------------------------- #
# top-k decode + reward-aligned val metric + opt-in checkpoint selection
# --------------------------------------------------------------------------- #
def _forward_one(router, token_hidden):
    with torch.no_grad():
        return router(token_hidden_states=token_hidden.unsqueeze(0), key_padding_mask=None)


def test_decode_topk_distinct_valid_and_top1_matches_decode():
    router = build_router()
    router.eval()
    seg, op = _forward_one(router, synthetic_hidden(seed=7))
    cands = router.decode_topk(seg[0], op[0], k=5)
    assert 1 <= len(cands) <= 5
    for p in cands:
        assert is_valid(p)  # every candidate is a valid program
    paths = [tuple(p.to_layer_path()) for p in cands]
    assert len(paths) == len(set(paths))  # distinct executed paths
    # top-1 is exactly decode() -> pass@1 unchanged
    assert cands[0].to_layer_path() == router.decode(seg[0], op[0]).to_layer_path()


def test_decode_topk_k1_is_single_program():
    router = build_router()
    router.eval()
    seg, op = _forward_one(router, synthetic_hidden(seed=8))
    cands = router.decode_topk(seg[0], op[0], k=1)
    assert len(cands) == 1
    assert cands[0].to_layer_path() == router.decode(seg[0], op[0]).to_layer_path()


def test_evaluate_val_programs_scores_cache_membership():
    router = build_router()
    router.eval()
    th0, th1 = synthetic_hidden(seed=1), synthetic_hidden(seed=2)
    seg, op = _forward_one(router, th0)
    p0 = tuple(router.decode(seg[0], op[0]).to_layer_path())  # guaranteed hit for q0
    m = evaluate_val_programs(
        router,
        ["q0", "q1"],
        {"q0": th0, "q1": th1},
        {"q0": {p0}, "q1": {(0,)}},  # q0 hits, q1 (implausible path) misses
        k=5,
    )
    assert m["val_program_n"] == 2.0
    assert m["val_cache_acc_at1"] == 0.5  # exactly q0
    assert m["val_cache_acc_atk"] >= 0.5
    for key in ("val_nonidentity_rate", "val_skip_frac", "val_rep_frac"):
        assert 0.0 <= m[key] <= 1.0


def _val_program_data_from(examples):
    """{questions, token_hiddens, valid_sets} where each example's own path is valid."""
    token_hiddens, valid_sets = {}, {}
    for e in examples:
        token_hiddens.setdefault(e.question, e.token_hidden)
        # recover this example's executed path from its targets is overkill; use identity
        valid_sets.setdefault(e.question, set()).add(tuple(range(D)))
    return {
        "questions": list(token_hiddens),
        "token_hiddens": token_hiddens,
        "valid_sets": valid_sets,
    }


def test_train_select_by_val_cache_acc_populates_and_selects():
    router = build_router()
    train_ex = _encoded_examples(seed0=0, n=6)
    val_ex = _encoded_examples(seed0=50, n=4)
    vpd = _val_program_data_from(val_ex)
    res = train(
        router,
        train_ex,
        val_examples=val_ex,
        epochs=3,
        batch_size=4,
        device=torch.device("cpu"),
        val_program_data=vpd,
        select_by="val_cache_acc",
        val_topk=3,
    )
    assert len(res.val_cache_acc_atk) == 3
    assert len(res.val_cache_acc_at1) == 3
    assert res.select_by == "val_cache_acc"
    assert res.best_epoch is not None
    assert res.best_metric is not None


def test_train_select_by_val_cache_acc_requires_program_data():
    router = build_router()
    train_ex = _encoded_examples(n=4)
    with pytest.raises(ValueError):
        train(
            router,
            train_ex,
            epochs=1,
            batch_size=4,
            device=torch.device("cpu"),
            select_by="val_cache_acc",
        )


def test_train_select_by_val_cache_acc_at1_selects_on_top1_only():
    """val_cache_acc_at1 selection is immune to the beam-width confound: it
    must track the AT1 trajectory's argmax, not atk's (atk can peak at an
    under-trained epoch that at1 does not agree with, observed empirically)."""
    router = build_router()
    train_ex = _encoded_examples(seed0=0, n=6)
    val_ex = _encoded_examples(seed0=50, n=4)
    vpd = _val_program_data_from(val_ex)
    res = train(
        router,
        train_ex,
        val_examples=val_ex,
        epochs=3,
        batch_size=4,
        device=torch.device("cpu"),
        val_program_data=vpd,
        select_by="val_cache_acc_at1",
        val_topk=3,
    )
    assert res.select_by == "val_cache_acc_at1"
    assert len(res.val_cache_acc_epochs) == 3
    assert res.best_epoch is not None
    best_idx = res.val_cache_acc_epochs.index(res.best_epoch)
    assert res.best_metric == max(res.val_cache_acc_at1)
    assert res.val_cache_acc_at1[best_idx] == max(res.val_cache_acc_at1)


def test_train_select_by_val_cache_acc_at1_requires_program_data():
    router = build_router()
    train_ex = _encoded_examples(n=4)
    with pytest.raises(ValueError):
        train(
            router,
            train_ex,
            epochs=1,
            batch_size=4,
            device=torch.device("cpu"),
            select_by="val_cache_acc_at1",
        )


def test_train_default_select_by_val_loss_unchanged():
    router = build_router()
    train_ex = _encoded_examples(seed0=0, n=6)
    val_ex = _encoded_examples(seed0=50, n=4)
    res = train(
        router, train_ex, val_examples=val_ex, epochs=2, batch_size=4, device=torch.device("cpu")
    )
    assert res.select_by == "val_loss"
    assert not res.val_cache_acc_atk  # no program metric without val_program_data
    assert res.best_epoch is not None


def test_train_val_program_every_skips_epochs():
    router = build_router()
    train_ex = _encoded_examples(seed0=0, n=6)
    val_ex = _encoded_examples(seed0=50, n=4)
    vpd = _val_program_data_from(val_ex)
    res = train(
        router,
        train_ex,
        val_examples=val_ex,
        epochs=4,
        batch_size=4,
        device=torch.device("cpu"),
        val_program_data=vpd,
        val_program_every=2,
    )
    # epochs 0, 2, and the always-included last epoch 3 -> 3 evaluations (epoch 1 skipped)
    assert len(res.val_cache_acc_atk) == 3


def test_train_move_encodings_to_device_cpu_is_noop():
    # device.type == "cpu" -> the move guard is skipped; training still runs.
    router = build_router()
    train_ex = _encoded_examples(seed0=0, n=6)
    res = train(
        router,
        train_ex,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
        move_encodings_to_device=True,
    )
