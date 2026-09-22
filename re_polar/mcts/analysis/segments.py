"""Program/segment math shared by every insight. Nothing here hardcodes a
layer count, every function takes it as an explicit argument (or a
``Program`` that already carries it), so the same code works for any model
in ``MODEL_REGISTRY``.

Decoding always goes through ``re_polar.router.train.program_from_layer_path``
(the SAME canonical path->Program parser the router training/inference
pipeline uses, not a reimplementation) and every decode is asserted to
round-trip, matches the standing invariant elsewhere in this repo.
"""
import math
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from re_polar.core import Program
from re_polar.router.train import program_from_layer_path


def third(start: int, num_layers: int) -> str:
    """Which coarse third of the stack a layer index falls in."""
    if start < num_layers / 3:
        return "early"
    if start < 2 * num_layers / 3:
        return "mid"
    return "late"


@lru_cache(maxsize=None)
def _decode_cached(path: Tuple[int, ...], num_layers: int) -> Program:
    prog = program_from_layer_path(path, num_layers, strict_repeat_2x=False)
    got = prog.to_layer_path()
    if got != list(path):
        raise AssertionError(f"segment decode did not round-trip: {tuple(path)} -> {got}")
    return prog


def decode(path: Sequence[int], num_layers: int) -> Program:
    """path -> Program, asserted round-trip (standing invariant).

    strict_repeat_2x=False: this is ANALYSIS of real MCTS-found paths (what did
    the search actually find?), not router-training label generation -- we want
    the full true `times`, not PoLar's times==2-only rejection (that default in
    re_polar/router/train.py exists for a router op-head/decode mismatch reason
    that doesn't apply here).

    Memoized by (path, num_layers) -- profiling a full report build found
    millions of calls, with cumulative decode time a large share of total
    build time, because the same programs get decoded repeatedly across
    different insight computations. `path` is tupled here so callers can
    still pass a list."""
    return _decode_cached(tuple(path), num_layers)


def edit_signature(prog: Program) -> Tuple[Tuple[int, int], ...]:
    """Ordered tuple of (start, length) for non-KEEP segments; skip/repeat
    collapsed to "edit", repeat times ignored. Identity => ()."""
    return tuple((s.start, len(s)) for s in prog.segments if s.op.value != "keep")


def full_signature(prog: Program) -> Tuple[Tuple[int, int], ...]:
    """Secondary: ordered tuple of (start, length) for ALL segments (incl.
    keep runs), still op-agnostic (position/length only)."""
    return tuple((s.start, len(s)) for s in prog.segments)


def sig_str(sig: Tuple[Tuple[int, int], ...]) -> str:
    if not sig:
        return "identity"
    return ",".join(f"[{s},{s + l})" for s, l in sig)


def entropy_bits(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def gini(counts: Sequence[int]) -> float:
    """Gini coefficient of a count distribution (0 = perfectly even, ->1 = concentrated)."""
    vals = sorted(c for c in counts if c > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def topk_coverage_share(sorted_counts: Sequence[int], total: int,
                         ks: Sequence[int] = (1, 5, 10, 20, 50)) -> Dict[str, float]:
    out = {}
    for k in ks:
        out[str(k)] = sum(sorted_counts[:k]) / total if total else 0.0
    return out


def program_edits(path: Sequence[int], num_layers: int) -> Tuple[set, set]:
    """Return (skipped_set, repeated_set) of layer indices for a path."""
    cnt = Counter(path)
    present = set(cnt)
    skipped = set(range(num_layers)) - present
    repeated = {i for i, c in cnt.items() if c > 1}
    return skipped, repeated


def op_class(path: Sequence[int], num_layers: int) -> str:
    sk, rp = program_edits(path, num_layers)
    if not sk and not rp:
        return "identity"
    if sk and not rp:
        return "skip-only"
    if rp and not sk:
        return "repeat-only"
    return "both"


def n_edits(path: Sequence[int], num_layers: int) -> int:
    """Total edit magnitude: #skipped + #extra repeated executions."""
    sk, _ = program_edits(path, num_layers)
    extra = len(path) - len(set(path))
    return len(sk) + extra


def shortest_valid(paths: Sequence[Sequence[int]], num_layers: int) -> Tuple[int, ...]:
    """Pick THE representative shortest-valid program. Primary: min executed
    length (shorter = better). Tie-break: fewest edits (simplest), then
    lexicographically smallest path (determinism)."""
    return min((tuple(p) for p in paths),
               key=lambda p: (len(p), n_edits(p, num_layers), p))


def layer_freq(paths: Sequence[Sequence[int]], which: str, num_layers: int) -> List[float]:
    """Count, per layer index, how many of `paths` skip / repeat that layer."""
    freq = [0] * num_layers
    for p in paths:
        sk, rp = program_edits(p, num_layers)
        for i in (sk if which == "skip" else rp):
            freq[i] += 1
    return freq


def identity(num_layers: int) -> Tuple[int, ...]:
    return tuple(range(num_layers))
