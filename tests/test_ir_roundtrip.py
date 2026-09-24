"""IR serde round-trips, path expansion, and grammar validity rules."""

import json

import pytest

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, is_valid, validate_program

D = 36  # qwen3_8b


def seg(start, end, op, **params):
    return Segment(start=start, end=end, op=op, params=params)


def test_identity_program():
    p = Program.identity(D)
    validate_program(p)
    assert p.is_identity()
    assert p.to_layer_path() == list(range(D))


def test_roundtrip_json():
    p = Program(
        num_layers=D,
        segments=[
            seg(0, 4, Op.KEEP),
            seg(4, 6, Op.SKIP),
            seg(6, 9, Op.REPEAT, times=3),
            seg(9, 12, Op.KEEP),
            *Program.identity(D).segments[3:],  # keep segments covering [12, 36)
        ],
    )
    validate_program(p)
    restored = Program.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored == p
    assert restored.to_layer_path() == p.to_layer_path()


def test_path_expansion():
    p = Program(
        num_layers=8,
        segments=[
            seg(0, 2, Op.KEEP),
            seg(2, 4, Op.SKIP),
            seg(4, 6, Op.REPEAT, times=2),
            seg(6, 8, Op.KEEP),
        ],
    )
    validate_program(p)
    assert p.to_layer_path() == [0, 1, 4, 5, 4, 5, 6, 7]
    assert not p.is_identity()


def test_repeat_default_times():
    assert seg(0, 2, Op.REPEAT).times == 2
    assert seg(0, 2, Op.KEEP).times == 1


@pytest.mark.parametrize(
    "segments",
    [
        [],  # no segments
        [seg(1, 4, Op.KEEP)],  # doesn't start at 0
        [seg(0, 4, Op.KEEP), seg(5, 8, Op.KEEP)],  # gap
        [seg(0, MAX_SEGMENT_LEN + 1, Op.KEEP)],  # too long
        [seg(0, 4, Op.KEEP), seg(4, 8, Op.KEEP), seg(8, 8, Op.KEEP)],  # empty segment
        [seg(0, 4, Op.KEEP)],  # incomplete cover (D=8)
        [seg(0, 4, Op.SKIP), seg(4, 8, Op.SKIP)],  # all-skip -> empty path
        [seg(0, 4, Op.REPEAT, times=1), seg(4, 8, Op.KEEP)],  # repeat times < 2
        [seg(0, 4, Op.KEEP, times=2), seg(4, 8, Op.KEEP)],  # params on keep
    ],
)
def test_invalid_programs(segments):
    assert not is_valid(Program(num_layers=8, segments=segments))
