"""Program validity rules (PoLar's constraints).

- segments sorted by start, contiguous, covering exactly [0, num_layers)
- every segment has 1 <= length <= MAX_SEGMENT_LEN
- REPEAT needs params["times"] >= 2; KEEP/SKIP take no params
- the expanded path must be non-empty (not everything skipped)
"""

from .ir import MAX_SEGMENT_LEN, Op, Program


def validate_program(program: Program) -> None:
    """Raise ValueError with the reason; returns None if valid."""
    if program.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {program.num_layers}")
    if not program.segments:
        raise ValueError("Program has no segments.")

    cursor = 0
    for seg in program.segments:
        if seg.start != cursor:
            raise ValueError(f"Segment {seg} not contiguous: expected start {cursor}.")
        if not (1 <= len(seg) <= MAX_SEGMENT_LEN):
            raise ValueError(f"Segment {seg} length {len(seg)} outside [1, {MAX_SEGMENT_LEN}].")
        if seg.op is Op.REPEAT:
            if seg.times < 2:
                raise ValueError(f"REPEAT segment {seg} needs times >= 2.")
        elif seg.params:
            raise ValueError(f"{seg.op.value} segment {seg} takes no params.")
        cursor = seg.end
    if cursor != program.num_layers:
        raise ValueError(f"Segments cover [0, {cursor}), expected [0, {program.num_layers}).")

    program.to_layer_path()  # raises on all-skip


def is_valid(program: Program) -> bool:
    try:
        validate_program(program)
        return True
    except ValueError:
        return False
