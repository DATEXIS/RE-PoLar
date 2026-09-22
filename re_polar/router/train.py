"""Router training loop, distil MCTS supervision into the PolarRouter.

Reimplemented from the paper ("Skip a Layer or Loop It? Learning
Program-of-Layers in LLMs", arXiv:2606.06574), no PoLar code is copied or
imported (see NOTICE.md).

Supervision comes from `merged_mcts_samples.json` (see `re_polar/datasets/schemas.py`):

    {"samples": [{"question": str, "gt_ans": str,
                  "final_valid_transitions":   [[int, ...], ...],   # executed layer paths
                  "final_invalid_transitions": [[int, ...], ...],
                  "initial_transition_metric": float,
                  "sample_info": {...}}, ...]}

**Multi-path labelling (the PoLar recipe).** Every *valid* program of a question
is a supervision target, not just the shortest one. For each sample we take ALL
of ``final_valid_transitions`` (capped at ``max_paths_per_sample`` with a
deterministic RNG sample when there are more), parse EACH flat layer-path back
into a `re_polar.core.Program` (contiguous segments <= MAX_SEGMENT_LEN, ops
skip/keep/repeat) and emit ONE :class:`Example` per path. Two router targets per
example:

  * ``seg_flip`` in {0,1}^D, a per-layer segment-*boundary* target. ``seg_flip[i]=1``
    marks a segment start at layer ``i``; ``seg_flip[0]`` is always 0 (layer 0 is
    trivially a start and is EXCLUDED from the BCE, matching decode which never
    thresholds layer 0);
  * ``op_labels`` in ({SKIP,KEEP,REPEAT} ∪ {-100})^D, the op index at each
    segment start and ``-100`` (ignore_index) everywhere else (masked CE).

Op-index convention matches the router's op head columns
(`DEFAULT_OPS` = SKIP=0, KEEP/execute=1, REPEAT=2), so targets, logits, and
PoLar's own bridge format line up.

Training the single-shortest-valid label collapsed the router to identity;
PoLar's multi-path labelling with a *plain* loss (BCE on
``seg_logits[:,1:]``; masked CE at segment starts; NO class rebalancing) is the
fix. That single-shortest-valid recipe was re-added as an explicit opt-in
ablation (``build_examples(shortest_only=True)`` / sweep scripts'
``--label-mode shortest``) to re-test on later per-tree-seed MCTS
data, see that function's docstring for the caveats. Optional ablation knobs
(``polar_lenpref``, anti-original penalty, per-sample weight normalisation)
are implemented but OFF by default.

**Encoder efficiency.** ~65k path-examples come from far fewer (~3.8k) unique
questions. :func:`encode_examples` encodes each UNIQUE question once with the
frozen encoder and SHARES that ``token_hidden`` across all of its path-examples.

Training freezes the Qwen3-Embedding encoder (the router keeps it in eval +
``requires_grad=False``); only the head is optimised (AdamW). The encoder is
*injectable* so tests drive the whole loop on CPU with precomputed token
embeddings and no download (build the router with ``embed_dim=`` and hand the
examples their own ``token_hidden``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, validate_program
from re_polar.router.model import DEFAULT_OPS, PolarRouter

__all__ = [
    "program_from_layer_path",
    "program_to_targets",
    "targets_from_path",
    "path_to_polar_targets",
    "load_supervision",
    "load_supervision_many",
    "split_samples_train_val",
    "Example",
    "build_examples",
    "collate",
    "encode_examples",
    "compute_loss",
    "TrainResult",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "main",
]

# Op-head column of KEEP (== OP_EXECUTE == 1); used by the anti-original penalty.
_KEEP_INDEX = DEFAULT_OPS.index(Op.KEEP)


# --------------------------------------------------------------------------- #
# layer-path -> Program (canonical parse)
# --------------------------------------------------------------------------- #
def program_from_layer_path(
    path: Sequence[int], num_layers: int, *, cap_keep: bool = True, strict_repeat_2x: bool = True
) -> Program:
    """Parse a flat executed layer-path back into a canonical ``Program``.

    A path is `Program.to_layer_path()`: each layer in [0, D) is emitted 0 times
    (SKIP), once (KEEP) or ``t`` times (REPEAT, MCTS uses t=2). The op of a layer
    is therefore fixed by how often it appears (order-independent); the *ordering*
    of the path only disambiguates where REPEAT blocks begin and end
    (``[4,5,4,5]`` = REPEAT[4,6) vs ``[4,4,5,5]`` = two size-1 REPEATs).

    So we (1) count occurrences to fix each layer's op, then (2) walk left to
    right grouping consecutive same-op layers into segments, capped at
    MAX_SEGMENT_LEN; REPEAT segment spans are taken from the ascending run in the
    path so interleaving is honoured. The reconstruction is asserted to
    re-execute to exactly ``path`` (crash-loudly on anything unparseable).

    Segment boundaries inside a long KEEP/SKIP run (or between two adjacent same-
    op segments that produced the same path) are not recoverable and are chosen
    canonically (maximal <= MAX_SEGMENT_LEN chunks), this matches ``decode``'s
    own splitting, so router targets and decode are consistent.

    ``cap_keep`` (default True, unchanged behaviour): whether KEEP runs are also
    chunked at MAX_SEGMENT_LEN like SKIP/REPEAT. PoLar's own text (Section 3.1:
    "each segment length bounded by ... <= K_max") applies the cap to the whole
    partition with no stated exception, and Appendix B.2 confirms K_max=4 is a
    SEARCH-side bound on skip/repeat block size only (MCTS has no "keep" action
    at all -- keep is just whatever the search never touched, and can be
    arbitrarily long in a raw MCTS path). So capping KEEP is a representational
    choice made when building the router's *target*, not something the search
    itself produces or needs. ``cap_keep=False`` (an ablation testing the
    hypothesis that this cap is unnecessary) merges each maximal KEEP run into
    ONE segment, however long,
    while SKIP/REPEAT stay capped at MAX_SEGMENT_LEN exactly as before. This
    changes the labelling ONLY: a program with a long uncapped KEEP segment
    executes identically to the same run chunked into <=4 pieces (KEEP takes no
    params), so this cannot change the search space or what gets executed --
    only what boundary the router's z_seg head is asked to predict. Because such
    a program can have KEEP segments longer than MAX_SEGMENT_LEN, it is NOT
    validated against the strict grammar (``validate_program`` would reject it);
    a relaxed check specific to this mode is used instead (see below).

    ``strict_repeat_2x`` (default True): PoLar's OWN
    released parser (``polar/data.py::parse_path_to_seg_and_ops``) only ever
    matches a REPEAT block against exactly ``chunk + chunk`` (times==2) -- a path
    containing any REPEAT with times 3/4/5 fails to parse in their DP and the
    WHOLE PATH is silently dropped from their training set. This isn't just a
    parser scope limit to route around: their ``decode``/execution side is
    ALSO hardcoded to times==2 (ours too, via ``_REPEAT_TIMES`` in
    ``re_polar/router/model.py``), and the op-head's training target is a bare
    3-way class index with no "times" dimension at all (``program_to_targets``
    only ever writes ``op_index[Op.REPEAT]``). So training on a times=3/4/5
    path (the ``strict_repeat_2x=False`` behavior) teaches the op-head "REPEAT
    is right here" from a label that decode can NEVER faithfully reproduce --
    decode's own realization is always times=2, a program that was never
    independently verified as correct at that exact position. That is a real
    label/decode mismatch, not just a stricter-than-necessary filter.

    Measured across every MCTS search run we have data for: times=2 is a
    MINORITY of REPEAT segments across runs (32.5%-52.6%, depending on the
    run) -- so this isn't a rare edge case, it's the majority of repeat
    supervision. ``strict_repeat_2x=True`` (matching
    PoLar's own DP-parser rejection exactly: any REPEAT segment with
    ``times != 2`` raises ValueError, caught by ``build_examples``'s per-path
    try/except like any other unparseable path -- the WHOLE path is skipped,
    not just that segment) removes this mismatch at the cost of discarding
    ~53% of valid paths. This is the default as a pure fidelity/correctness
    fix -- both defaults were sweep-tested with no material difference to the
    router's KEEP-collapse behavior, so this is not expected to change
    collapse behavior, only to stop training on unreproducible labels.
    """
    path = list(path)
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if any(not (0 <= l < num_layers) for l in path):
        raise ValueError(f"path references a layer outside [0, {num_layers}): {path}")

    count = Counter(path)  # count[l] == op multiplicity of layer l
    segments: List[Segment] = []
    pos = 0          # index into path (entries consumed so far)
    layer = 0        # next layer index needing a segment
    while layer < num_layers:
        mult = count.get(layer, 0)
        if mult == 0:
            # SKIP run: consecutive count-0 layers, capped at MAX_SEGMENT_LEN.
            k = 1
            while layer + k < num_layers and count.get(layer + k, 0) == 0 and k < MAX_SEGMENT_LEN:
                k += 1
            segments.append(Segment(layer, layer + k, Op.SKIP))
            layer += k
        elif mult == 1:
            # KEEP run: consecutive count-1 layers. Capped at MAX_SEGMENT_LEN
            # unless cap_keep=False, in which case the whole run is one segment.
            k = 1
            while (
                layer + k < num_layers
                and count.get(layer + k, 0) == 1
                and (cap_keep is False or k < MAX_SEGMENT_LEN)
            ):
                k += 1
            expected = list(range(layer, layer + k))
            if path[pos:pos + k] != expected:
                raise ValueError(f"cannot parse KEEP at layer {layer}: {path[pos:pos + k]} != {expected}")
            segments.append(Segment(layer, layer + k, Op.KEEP))
            pos += k
            layer += k
        else:
            # REPEAT segment: block = the ascending run in the path (<= MAX_SEGMENT_LEN),
            # repeated `mult` times. Ordering distinguishes block boundaries.
            # Always capped -- cap_keep only relaxes KEEP, never REPEAT.
            k = 1
            while (
                pos + k < len(path)
                and path[pos + k] == layer + k
                and k < MAX_SEGMENT_LEN
            ):
                k += 1
            block = list(range(layer, layer + k))
            expected = block * mult
            if path[pos:pos + k * mult] != expected:
                raise ValueError(
                    f"cannot parse REPEAT at layer {layer} (times={mult}): "
                    f"{path[pos:pos + k * mult]} != {expected}"
                )
            if strict_repeat_2x and mult != 2:
                raise ValueError(
                    f"REPEAT at layer {layer} has times={mult} != 2 -- rejected under "
                    f"strict_repeat_2x (matches PoLar's own parser, which only accepts exactly 2x)"
                )
            segments.append(Segment(layer, layer + k, Op.REPEAT, {"times": mult}))
            pos += k * mult
            layer += k

    program = Program(num_layers=num_layers, segments=segments)
    if program.to_layer_path() != path:
        raise ValueError(f"parsed program does not round-trip to path: {path}")
    if cap_keep:
        validate_program(program)
    else:
        # Relaxed check: same as validate_program but KEEP is exempt from the
        # MAX_SEGMENT_LEN bound (SKIP/REPEAT are still checked at full strength).
        for seg in segments:
            if seg.op is not Op.KEEP and not (1 <= len(seg) <= MAX_SEGMENT_LEN):
                raise ValueError(f"Segment {seg} length {len(seg)} outside [1, {MAX_SEGMENT_LEN}].")
            if seg.op is Op.REPEAT and seg.times < 2:
                raise ValueError(f"REPEAT segment {seg} needs times >= 2.")
    return program


# --------------------------------------------------------------------------- #
# Program -> router targets
# --------------------------------------------------------------------------- #
def program_to_targets(
    program: Program, ops: Sequence[Op] = DEFAULT_OPS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Program -> (seg_target (D,) float, op_target (D,) long, op_mask (D,) bool).

    ``seg_target[i] = 1`` iff layer ``i`` begins a segment (layer 0 always does);
    ``op_target[i]`` is the op-vocabulary index of the segment starting at ``i``
    (defined only where ``op_mask`` is True, the segment-start layers).
    """
    D = program.num_layers
    op_index = {op: i for i, op in enumerate(ops)}
    seg_target = torch.zeros(D, dtype=torch.float32)
    op_target = torch.zeros(D, dtype=torch.long)
    op_mask = torch.zeros(D, dtype=torch.bool)
    for seg in program.segments:
        if seg.op not in op_index:
            raise ValueError(f"op {seg.op} not in op vocabulary {list(ops)}")
        seg_target[seg.start] = 1.0
        op_target[seg.start] = op_index[seg.op]
        op_mask[seg.start] = True
    return seg_target, op_target, op_mask


def targets_from_path(
    path: Sequence[int], num_layers: int, ops: Sequence[Op] = DEFAULT_OPS, *,
    cap_keep: bool = True, strict_repeat_2x: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Program]:
    """Convenience: layer-path -> (seg_target, op_target, op_mask, program)."""
    program = program_from_layer_path(path, num_layers, cap_keep=cap_keep, strict_repeat_2x=strict_repeat_2x)
    seg_target, op_target, op_mask = program_to_targets(program, ops)
    return seg_target, op_target, op_mask, program


def path_to_polar_targets(
    path: Sequence[int], num_layers: int, ops: Sequence[Op] = DEFAULT_OPS, *,
    cap_keep: bool = True, strict_repeat_2x: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Program]:
    """layer-path -> (seg_flip (D,) float, op_labels (D,) long, program).

    PoLar's per-path label form (one training example per valid path):

    * ``seg_flip`` = ``seg_target`` with layer 0 forced to 0, layer 0 is always a
      segment start, so it carries no boundary information and is excluded from
      the BCE (``seg_logits[:,1:]``), matching decode which never thresholds it.
    * ``op_labels`` = ``op_target`` with non-segment-start positions set to the
      cross-entropy ignore_index ``-100``; segment starts (INCLUDING layer 0)
      keep their op index, so the op head is supervised exactly where decode
      reads it.

    ``cap_keep`` (default True) and ``strict_repeat_2x`` (default True) are
    forwarded to :func:`program_from_layer_path`, see that docstring.
    """
    seg_target, op_target, op_mask, program = targets_from_path(
        path, num_layers, ops, cap_keep=cap_keep, strict_repeat_2x=strict_repeat_2x
    )
    seg_flip = seg_target.clone()
    seg_flip[0] = 0.0
    op_labels = op_target.clone()
    op_labels[~op_mask] = -100
    return seg_flip, op_labels, program


# --------------------------------------------------------------------------- #
# supervision loading
# --------------------------------------------------------------------------- #
def load_supervision(samples_path) -> List[dict]:
    """Load a ``merged_mcts_samples.json`` and return its ``samples`` list."""
    with open(samples_path) as f:
        blob = json.load(f)
    samples = blob["samples"] if isinstance(blob, dict) else blob
    if not isinstance(samples, list):
        raise ValueError(f"{samples_path}: expected a list of samples, got {type(samples)}")
    return samples


def load_supervision_many(samples_paths) -> List[dict]:
    """Load + concatenate one or more ``merged_mcts_samples.json`` files.

    The paper trains a SINGLE router over all difficulties, but our full MCTS run
    shards output one file per difficulty (dart-math-diff-{1..5}). This joins them
    into one supervision set (order preserved). A lone str/Path is also accepted.
    """
    if isinstance(samples_paths, (str, Path)):
        samples_paths = [samples_paths]
    combined: List[dict] = []
    for path in samples_paths:
        combined.extend(load_supervision(path))
    return combined


def split_samples_train_val(
    samples: Sequence[dict], val_frac: float = 0.1
) -> Tuple[List[dict], List[dict]]:
    """Split ONE difficulty's samples into (train, val) by the last ``val_frac``.

    Splitting is on QUESTIONS (samples), not path-examples, so a question's
    valid paths never leak across the train/val boundary. Called per-difficulty
    file (before concatenation) so the holdout is balanced across difficulties:
    our full search run has ~1250 samples/difficulty (PoLar assumed 1500).
    ``val_frac <= 0`` -> no holdout.
    """
    samples = list(samples)
    n = len(samples)
    n_val = int(round(n * val_frac)) if val_frac > 0.0 else 0
    n_val = max(0, min(n_val, n))
    if n_val == 0:
        return samples, []
    return samples[: n - n_val], samples[n - n_val:]


# --------------------------------------------------------------------------- #
# examples / batching
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    """One training example: a question and ONE valid program's PoLar targets.

    A question with ``k`` valid paths yields ``k`` examples that SHARE one
    ``token_hidden`` (filled by :func:`encode_examples`, or supplied directly by
    tests to avoid a download). ``token_hidden`` is (T, embed_dim); a None
    ``key_padding_mask`` means no padding (all real tokens).

    * ``path_len``: executed layers of this path (``polar_lenpref`` weighting).
    * ``weight``: per-example multiplier (``per_sample_weight_normalize`` sets
      it to ``1/n_paths`` so many-path questions don't dominate).
    * ``anti_original_active``: True iff this sample's identity path is NOT in
      its ``final_valid_transitions`` (target of the anti-original penalty).
    """

    question: str
    seg_flip: torch.Tensor              # (D,) float, seg_flip[0] == 0 always
    op_labels: torch.Tensor             # (D,) long, op index at seg starts else -100
    path_len: int
    weight: float = 1.0
    anti_original_active: bool = False
    token_hidden: Optional[torch.Tensor] = None        # (T, embed_dim)
    key_padding_mask: Optional[torch.Tensor] = None     # (T,) bool, True = pad


def build_examples(
    samples: Sequence[dict],
    num_layers: int,
    ops: Sequence[Op] = DEFAULT_OPS,
    *,
    max_paths_per_sample: int = 50,
    per_sample_weight_normalize: bool = False,
    reweight_original_path: bool = False,
    original_path_weight: float = 1.0,
    drop_original_path: bool = False,
    keep_original_prob: float = 0.0,
    anti_original: bool = False,
    shortest_only: bool = False,
    cap_keep: bool = True,
    strict_repeat_2x: bool = True,
    seed: int = 0,
) -> List[Example]:
    """Multi-path examples: ONE example per valid path (PoLar labelling).

    For each sample: take ALL of ``final_valid_transitions`` (drop empty/all-skip),
    cap at ``max_paths_per_sample`` with a deterministic ``random.Random(seed)``
    sample when there are more, parse each path to PoLar targets and emit one
    example. Samples with no valid (parseable) path are dropped. Unparseable
    individual paths are skipped (should not occur for MCTS-emitted paths).

    Original-path (identity) handling: PoLar's two MUTUALLY EXCLUSIVE anti-collapse
    levers (their ``data.py``: drop is applied only when reweight is off):

    * ``reweight_original_path``: DOWN-WEIGHT the identity path to
      ``original_path_weight`` when this sample has both the identity path and a
      strictly-shorter valid path; all other paths weigh 1.0 (the README recipe).
    * ``drop_original_path`` / ``keep_original_prob``: instead DROP the identity
      path from the targets when a strictly-shorter valid path exists. Hard-drop
      (``keep_original_prob=0``) removes it always; soft-drop
      (``0<keep_original_prob<1``) keeps it with that probability (per-sample RNG).
      Faithful port of their ``_maybe_drop_original_path_from_valid_paths``; only
      active when ``reweight_original_path`` is False.
    * ``shortest_only``: instead of multi-path labelling, collapse each sample to
      the SINGLE shortest valid path (ties broken by ``random.Random(seed)``) and
      emit exactly one example. Mutually exclusive in effect with the two levers
      above (they operate on ``valid`` before this reduction and become moot once
      only one path remains). NOTE (see this module's own docstring):
      training on the single-shortest-valid
      label was the ORIGINAL recipe and it collapsed the router to identity;
      multi-path labelling was adopted specifically to fix that. Re-added here
      as an explicit, opt-in ablation (testing the hypothesis that one
      canonical, maximally-distinct-per-input label gives a cleaner signal
      than blending all valid paths) to re-test on later per-tree-seed MCTS
      data, which has a much lower menu-top1 concentration (2-3% vs 30-55%)
      than the data that produced the original collapse, NOT expected to
      behave identically, but go in assuming the prior is a real negative
      result, not untested territory.

    Then, if ``per_sample_weight_normalize``, every example of the sample is scaled
    by ``1/n_paths`` (a no-op under ``shortest_only``, since ``n_paths`` == 1).
    ``anti_original`` flags examples of samples whose identity path is absent from
    the valid set.

    ``cap_keep`` (default True, unchanged) is forwarded to
    ``path_to_polar_targets``/``program_from_layer_path``, see that function's
    docstring for the ``cap_keep=False`` "edit-capped" ablation: only
    SKIP/REPEAT segments stay capped at MAX_SEGMENT_LEN; KEEP
    runs merge into ONE segment however long, so ``seg_flip`` boundaries only
    fire where the operation genuinely changes, not on every 4th layer inside a
    long uninterrupted KEEP run. Orthogonal to ``shortest_only``, both can be
    combined.
    """
    rng = random.Random(seed)
    identity = list(range(num_layers))
    examples: List[Example] = []
    for sample in samples:
        valid = [list(p) for p in (sample.get("final_valid_transitions") or []) if p]
        if not valid:
            continue
        anti_active = bool(anti_original and identity not in valid)
        # original-path downweight trigger (PoLar _detect_original_path_and_shorter_valid):
        # identity present AND some valid path strictly shorter than full depth.
        trigger = (identity in valid) and any(len(p) < num_layers for p in valid)
        # Hard/soft DROP of the identity path (mutually exclusive with reweight, per
        # PoLar data.py). Only when a strictly-shorter valid path exists (== trigger).
        drop_enabled = (drop_original_path or keep_original_prob > 0.0) and not reweight_original_path
        if drop_enabled and trigger:
            kp = min(1.0, max(0.0, keep_original_prob))
            keep_it = (rng.random() < kp) if 0.0 < kp < 1.0 else (kp >= 1.0)
            if not keep_it:
                valid = [p for p in valid if p != identity]
                if not valid:
                    continue
        if shortest_only:
            min_len = min(len(p) for p in valid)
            shortest = [p for p in valid if len(p) == min_len]
            valid = [rng.choice(shortest)] if len(shortest) > 1 else shortest
        elif len(valid) > max_paths_per_sample:
            valid = rng.sample(valid, max_paths_per_sample)

        parsed: List[Tuple[int, bool, torch.Tensor, torch.Tensor]] = []
        for path in valid:
            try:
                seg_flip, op_labels, _program = path_to_polar_targets(
                    path, num_layers, ops, cap_keep=cap_keep, strict_repeat_2x=strict_repeat_2x
                )
            except ValueError:
                continue  # unparseable path, skip (MCTS paths should always parse)
            parsed.append((len(path), path == identity, seg_flip, op_labels))
        if not parsed:
            continue

        norm = 1.0 / len(parsed) if per_sample_weight_normalize else 1.0
        question = sample["question"]
        for path_len, is_original, seg_flip, op_labels in parsed:
            base = (original_path_weight
                    if (reweight_original_path and trigger and is_original) else 1.0)
            examples.append(
                Example(
                    question=question,
                    seg_flip=seg_flip,
                    op_labels=op_labels,
                    path_len=path_len,
                    weight=base * norm,
                    anti_original_active=anti_active,
                )
            )
    return examples


def collate(examples: Sequence[Example], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Pad + stack a list of encoded examples into a forward-ready batch.

    Token sequences are right-padded to the batch max; ``key_padding_mask`` is
    True at padded positions (nn.MultiheadAttention convention).
    """
    if any(e.token_hidden is None for e in examples):
        raise ValueError("collate needs encoded examples (call encode_examples first)")
    B = len(examples)
    embed_dim = examples[0].token_hidden.size(-1)
    Tmax = max(e.token_hidden.size(0) for e in examples)

    # Build on the SOURCE device: when encodings are pre-moved to GPU (the speed
    # path) this keeps the whole collate on-GPU and avoids the per-step CPU->GPU
    # copy of token_hidden; on CPU tensors (tests) it is unchanged.
    src_device = examples[0].token_hidden.device
    token_hidden = torch.zeros(B, Tmax, embed_dim, device=src_device)
    key_padding_mask = torch.ones(B, Tmax, dtype=torch.bool, device=src_device)  # all-pad, unmask real
    for i, e in enumerate(examples):
        t = e.token_hidden.size(0)
        token_hidden[i, :t] = e.token_hidden
        if e.key_padding_mask is not None:
            key_padding_mask[i, :t] = e.key_padding_mask
        else:
            key_padding_mask[i, :t] = False

    batch = {
        "token_hidden": token_hidden,
        "key_padding_mask": key_padding_mask,
        "seg_flip": torch.stack([e.seg_flip for e in examples]),
        "op_labels": torch.stack([e.op_labels for e in examples]),
        "path_len": torch.tensor([float(e.path_len) for e in examples], dtype=torch.float32),
        "weight": torch.tensor([float(e.weight) for e in examples], dtype=torch.float32),
        "anti_orig": torch.tensor(
            [1.0 if e.anti_original_active else 0.0 for e in examples], dtype=torch.float32
        ),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def encode_examples(
    router: PolarRouter, examples: Sequence[Example], batch_size: int = 16
) -> None:
    """Fill each example's ``token_hidden`` via the frozen encoder (in place).

    Encodes each UNIQUE question ONCE and SHARES the resulting ``token_hidden``
    tensor across every path-example of that question (there are ~65k examples
    but only ~3.8k distinct questions, encoding per path would waste ~17x the
    encoder passes). Examples that already carry ``token_hidden`` (tests, cache)
    are left alone, so this is a no-op when everything is precomputed and never
    touches the encoder / triggers a download.
    """
    todo = [e for e in examples if e.token_hidden is None]
    if not todo:
        return
    by_question: Dict[str, List[Example]] = {}
    for e in todo:
        by_question.setdefault(e.question, []).append(e)
    unique_questions = list(by_question.keys())
    for start in range(0, len(unique_questions), batch_size):
        chunk = unique_questions[start:start + batch_size]
        token_hidden, key_padding_mask = router.encode_questions(chunk)
        token_hidden = token_hidden.detach().cpu()
        key_padding_mask = key_padding_mask.detach().cpu()
        for i, question in enumerate(chunk):
            valid = ~key_padding_mask[i]  # keep only real tokens; store unpadded
            shared = token_hidden[i][valid].clone()
            for e in by_question[question]:
                e.token_hidden = shared  # shared reference across this question's paths
                e.key_padding_mask = None


# --------------------------------------------------------------------------- #
# loss / training
# --------------------------------------------------------------------------- #
def compute_loss(
    router: PolarRouter,
    batch: Dict[str, torch.Tensor],
    *,
    policy_mode: str = "polar",
    lenpref_beta: float = 0.05,
    anti_original_lambda: float = 0.0,
    op_class_weights: Optional[torch.Tensor] = None,
    op_focal_gamma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PoLar's plain per-example loss. Returns (total, seg_loss, op_loss).

    Per example: seg = mean BCE-with-logits over layers ``1..D-1`` (layer 0
    excluded), op = mean CE (ignore_index -100) over its segment-start layers.
    The batch loss is the (optionally weighted) mean of ``seg + op``. NO class
    weighting / pos_weight, that is the whole point of the multi-path recipe.

    Optional knobs (default OFF, matching PoLar so we can ablate later):
    ``policy_mode="polar_lenpref"`` reweights each example by ``exp(-beta*path_len)``;
    ``anti_original_lambda>0`` adds a penalty on the mean KEEP-probability of
    samples whose identity path is not valid.
    """
    seg_logits, op_logits = router(
        token_hidden_states=batch["token_hidden"],
        key_padding_mask=batch["key_padding_mask"],
    )
    B, D, n_ops = op_logits.shape

    # segmentation: BCE over layers 1..D-1 (seg_flip[0] is always 0, excluded).
    seg_elem = F.binary_cross_entropy_with_logits(
        seg_logits[:, 1:], batch["seg_flip"][:, 1:], reduction="none"
    )
    seg_loss_per = seg_elem.mean(dim=1)  # (B,)

    # op: masked CE at segment starts, per-example mean over labelled positions.
    # op_class_weights (default None == unweighted, PoLar-faithful) up-weights rare
    # op classes to counter the KEEP-dominant collapse (the op head predicts KEEP
    # everywhere and SKIP never without this).
    op_labels = batch["op_labels"]  # (B,D), -100 off segment starts
    weight = None
    if op_class_weights is not None:
        weight = op_class_weights.to(device=op_logits.device, dtype=op_logits.dtype)
    if op_focal_gamma and op_focal_gamma > 0.0:
        # Focal loss (Lin et al. 2017), the form DR.LLM's router uses
        # (their own modeling_qwen3.py, gamma=2.0):
        # FL = -(1-p_t)^gamma * log p_t, optionally * class weight (alpha). Focuses
        # gradient on HARD/rare positions instead of statically reweighting classes.
        flat_logits = op_logits.reshape(-1, n_ops)
        flat_tgt = op_labels.reshape(-1)
        logp = F.log_softmax(flat_logits, dim=-1)
        safe_tgt = flat_tgt.clamp(min=0)  # -100 -> 0; masked out below
        logpt = logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
        focal = -((1.0 - logpt.exp()) ** op_focal_gamma) * logpt
        if weight is not None:
            focal = focal * weight[safe_tgt]
        ce_tok = focal.view(B, D)
    else:
        ce_tok = F.cross_entropy(
            op_logits.reshape(-1, n_ops), op_labels.reshape(-1),
            weight=weight, ignore_index=-100, reduction="none",
        ).view(B, D)
    mask = (op_labels != -100).float()
    op_loss_per = (ce_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B,)

    loss_per = seg_loss_per + op_loss_per  # (B,)

    # length-preference reweighting (shorter valid paths weigh more), OFF by default.
    if policy_mode == "polar_lenpref":
        w = torch.exp(-lenpref_beta * batch["path_len"])
        w = w / (w.mean() + 1e-8)
    else:
        w = torch.ones_like(loss_per)
    w = w * batch["weight"]                 # per-sample-weight-normalize (1.0 by default)
    w = w / (w.mean() + 1e-8)               # keep loss scale ~invariant to the weights
    loss = (w * loss_per).mean()

    # anti-original penalty (TRAIN-ONLY): discourage collapsing to all-KEEP on
    # samples whose identity path is NOT valid. OFF by default.
    if anti_original_lambda > 0.0:
        keep_prob = torch.softmax(op_logits, dim=-1)[:, :, _KEEP_INDEX].mean(dim=1)  # (B,)
        anti = batch["anti_orig"]
        anti_loss = (anti * keep_prob).sum() / anti.sum().clamp(min=1.0)
        loss = loss + anti_original_lambda * anti_loss

    return loss, seg_loss_per.mean(), op_loss_per.mean()


def evaluate_val_programs(
    router: PolarRouter,
    questions: Sequence[str],
    token_hiddens: Dict[str, torch.Tensor],
    valid_sets: Dict[str, set],
    *,
    k: int = 5,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Reward-aligned, GENERATION-FREE validation metric.

    For each val question, decode the router's top-1 and top-``k`` programs from its
    precomputed ``token_hidden`` (no encoder call, no LLM generation) and check
    membership in ``valid_sets[q]``, the set of MCTS-verified valid executed paths
    for that question. A path in the set is one we KNOW solves the question, so:

    * ``val_cache_acc_at1``: top-1 program is a known-valid path (pass@1 proxy);
    * ``val_cache_acc_atk``: any of the top-k is (pass@k proxy);

    plus program-shape signals (``val_nonidentity_rate``/``val_skip_frac``/
    ``val_rep_frac``). A collapsed (all-identity) router scores ~identity-solve-rate
    here; a router that finds shorter/looped programs on hard questions scores
    higher, so this discriminates collapse from real learning, unlike val loss.
    This is what ``--select-by val_cache_acc`` optimises the checkpoint for.
    """
    D = router.num_layers
    identity = tuple(range(D))
    router.eval()
    hit1 = hitk = nonident = skipf = repf = n = 0
    rescue1 = rescuek = rescue_n = 0  # identity-WRONG subset: can the router rescue?
    qs = [q for q in questions if token_hiddens.get(q) is not None]
    with torch.no_grad():
        # Batched forward (pad token_hidden to the batch max; padded positions are
        # masked in cross-attention), the per-question forward was the val bottleneck.
        for start in range(0, len(qs), batch_size):
            chunk = qs[start:start + batch_size]
            ths = [token_hiddens[q] for q in chunk]
            Tmax = max(t.size(0) for t in ths)
            ed = ths[0].size(-1)
            dev = device or ths[0].device
            th = torch.zeros(len(chunk), Tmax, ed, device=dev)
            kpm = torch.ones(len(chunk), Tmax, dtype=torch.bool, device=dev)
            for i, t in enumerate(ths):
                th[i, :t.size(0)] = t.to(dev)
                kpm[i, :t.size(0)] = False
            seg_logits, op_logits = router(token_hidden_states=th, key_padding_mask=kpm)
            for i, q in enumerate(chunk):
                vset = valid_sets.get(q) or set()
                cands = router.decode_topk(seg_logits[i], op_logits[i], k=k)
                n += 1
                top1 = cands[0]
                p1 = tuple(top1.to_layer_path())
                hit_any_k = any(tuple(c.to_layer_path()) in vset for c in cands)
                if p1 in vset:
                    hit1 += 1
                if hit_any_k:
                    hitk += 1
                if identity not in vset:  # RESCUE-relevant: identity is wrong here
                    rescue_n += 1
                    if p1 in vset:
                        rescue1 += 1
                    if hit_any_k:
                        rescuek += 1
                if p1 != identity:
                    nonident += 1
                if any(s.op is Op.SKIP for s in top1.segments):
                    skipf += 1
                if any(s.op is Op.REPEAT for s in top1.segments):
                    repf += 1
    denom = max(1, n)
    return {
        "val_cache_acc_at1": hit1 / denom,
        "val_cache_acc_atk": hitk / denom,
        "val_nonidentity_rate": nonident / denom,
        "val_skip_frac": skipf / denom,
        "val_rep_frac": repf / denom,
        "val_program_n": float(n),
        # RESCUE on the identity-wrong subset (identity scores 0 here by construction):
        "val_rescue_at1": rescue1 / max(1, rescue_n),
        "val_rescue_atk": rescuek / max(1, rescue_n),
        "val_rescue_n": float(rescue_n),
    }


def _trainable_params(router: PolarRouter) -> List[nn.Parameter]:
    """Optimiser params = the head only (the frozen encoder has requires_grad=False)."""
    return [p for p in router.parameters() if p.requires_grad]


def _make_lr_scheduler(optimizer, name: str, warmup_steps: int, total_steps: int):
    """PoLar's warmup + decay schedule (train.py ``_lr_lambda``), stepped per
    optimiser step. ``name`` in {none, cosine, linear}; warmup ramps 0->1 over
    ``warmup_steps`` then decays over the remaining ``total_steps``. Returns None
    for ``none`` (flat LR)."""
    name = (name or "none").lower()
    if name == "none":
        return None
    total_steps = max(1, int(total_steps))
    warmup = max(0, min(int(warmup_steps), total_steps - 1))

    def _lr_lambda(step: int) -> float:
        step = int(step)
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if total_steps <= warmup:
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _head_state_cpu(router: PolarRouter) -> Dict[str, torch.Tensor]:
    """The trainable head's state_dict on CPU (encoder weights excluded)."""
    return {
        k: v.detach().cpu().clone()
        for k, v in router.state_dict().items()
        if not k.startswith("encoder.")
    }


def _epoch_loss(
    router: PolarRouter,
    examples: Sequence[Example],
    batch_size: int,
    device: torch.device,
    *,
    policy_mode: str,
    lenpref_beta: float,
    anti_original_lambda: float,
    op_class_weights: Optional[torch.Tensor] = None,
    op_focal_gamma: float = 0.0,
) -> float:
    """Mean total loss over ``examples`` in eval mode (validation pass)."""
    router.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], device=device)
            loss, _, _ = compute_loss(
                router, batch, policy_mode=policy_mode,
                lenpref_beta=lenpref_beta, anti_original_lambda=anti_original_lambda,
                op_class_weights=op_class_weights, op_focal_gamma=op_focal_gamma,
            )
            total += float(loss.detach())
            batches += 1
    return total / max(batches, 1)


@dataclass
class TrainResult:
    """Outcome of :func:`train`.

    ``train_losses`` / ``val_losses`` are per-epoch means (``val_losses`` empty
    when validation is disabled). ``best_epoch`` is the argmin val epoch whose
    head weights were restored into the router (None when no validation, the
    router then keeps its last-epoch weights). ``best_state`` is that restored
    head state_dict (None without validation).
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_cache_acc_epochs: List[int] = field(default_factory=list)  # epoch index of each entry below
    val_cache_acc_at1: List[float] = field(default_factory=list)
    val_cache_acc_atk: List[float] = field(default_factory=list)
    val_rescue_at1: List[float] = field(default_factory=list)  # rescue on identity-wrong subset
    val_rescue_atk: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_metric: Optional[float] = None  # value of the selection metric at best_epoch
    select_by: str = "val_loss"
    best_state: Optional[Dict[str, torch.Tensor]] = None


def train(
    router: PolarRouter,
    examples: Sequence[Example],
    *,
    val_examples: Optional[Sequence[Example]] = None,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 16,
    weight_decay: float = 0.0,
    policy_mode: str = "polar",
    lenpref_beta: float = 0.05,
    anti_original_lambda: float = 0.0,
    op_class_weights: Optional[torch.Tensor] = None,
    op_focal_gamma: float = 0.0,
    lr_scheduler: str = "none",
    warmup_steps: int = 0,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    seed: int = 0,
    log_every: int = 0,
    val_program_data: Optional[Dict[str, object]] = None,
    select_by: str = "val_loss",
    val_topk: int = 5,
    val_program_every: int = 1,
    move_encodings_to_device: bool = False,
) -> TrainResult:
    """Train the router head on precomputed/encoded multi-path examples.

    AdamW over the head only (frozen encoder). When ``val_examples`` is given a
    validation pass runs after each epoch and the best head is restored into
    ``router`` at the end; otherwise the router keeps its last-epoch weights.

    Checkpoint selection (``select_by``):

    * ``"val_loss"`` (default, unchanged): restore the MIN val-loss epoch;
    * ``"val_cache_acc"``: restore the MAX ``val_cache_acc_atk`` epoch (needs
      ``val_program_data``). Val loss is minimised by predicting the majority op
      (KEEP) everywhere, i.e. it rewards collapse-to-identity;
    * ``"val_cache_acc_at1"``: restore the MAX ``val_cache_acc_at1`` epoch.

    Both cache-acc variants need ``val_program_data`` and reward decoding a
    KNOWN-valid program rather than the collapsed minimum. But ``atk`` has its
    OWN failure mode (observed on an under-trained router): an under-trained
    head has near-uniform op logits, so the top-k beam spans more DISTINCT
    candidate paths and hits the valid-cache more often BY WIDTH, not by learned
    signal, ``val_cache_acc_atk`` can therefore be HIGHEST at epoch 0 and DECLINE
    as training sharpens the distribution (fewer beams -> less lucky diversity),
    even while the single top-1 guess keeps improving. ``val_cache_acc_at1``
    avoids this: it only rewards a single confident correct commitment (the
    exact pass@1 quantity used as the router's success criterion), so it
    cannot be gamed by beam width and is the recommended default once
    you've verified ``atk`` misbehaves in your own run (compare both
    trajectories in the printed summary).

    ``val_program_data`` (optional) = ``{"questions", "token_hiddens",
    "valid_sets"}`` drives the generation-free :func:`evaluate_val_programs`
    metric logged each epoch. Requires examples to be encoded.
    """
    if not examples:
        raise ValueError("no training examples")
    if any(e.token_hidden is None for e in examples):
        raise ValueError("examples must be encoded before train() (call encode_examples)")
    if val_examples and any(e.token_hidden is None for e in val_examples):
        raise ValueError("val examples must be encoded before train() (call encode_examples)")

    device = device or torch.device("cpu")
    router.to(device)

    # Speed path: move each UNIQUE token_hidden to `device` ONCE (examples of a
    # question share one tensor object, so dedup by id preserves sharing). collate
    # then builds batches on-device, eliminating the per-step CPU->GPU copy that
    # dominates wall-time for this tiny head. ~1.5GB for the full run; opt-in.
    if move_encodings_to_device and device.type != "cpu":
        moved: Dict[int, torch.Tensor] = {}
        for e in list(examples) + list(val_examples or []):
            key = id(e.token_hidden)
            if key not in moved:
                moved[key] = e.token_hidden.to(device)
            e.token_hidden = moved[key]

    router.train()  # keeps the frozen encoder in eval (PolarRouter.train override)
    optimizer = torch.optim.AdamW(_trainable_params(router), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)

    n = len(examples)
    steps_per_epoch = max(1, math.ceil(n / max(1, batch_size)))
    lr_sched = _make_lr_scheduler(optimizer, lr_scheduler, warmup_steps, epochs * steps_per_epoch)
    if select_by in ("val_cache_acc", "val_cache_acc_at1", "val_rescue_at1",
                     "val_rescue_atk") and val_program_data is None:
        raise ValueError(f"select_by={select_by!r} requires val_program_data")
    result = TrainResult()
    result.select_by = select_by
    best_val = float("inf")       # val_loss selection: minimise
    best_acc = float("-inf")      # val_cache_acc selection: maximise
    step = 0
    for epoch in range(epochs):
        router.train()
        order = torch.randperm(n, generator=rng).tolist() if shuffle else list(range(n))
        total, batches = 0.0, 0
        for bstart in range(0, n, batch_size):
            batch_examples = [examples[i] for i in order[bstart:bstart + batch_size]]
            batch = collate(batch_examples, device=device)
            loss, seg_loss, op_loss = compute_loss(
                router, batch, policy_mode=policy_mode,
                lenpref_beta=lenpref_beta, anti_original_lambda=anti_original_lambda,
                op_class_weights=op_class_weights, op_focal_gamma=op_focal_gamma,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_sched is not None:
                lr_sched.step()
            total += float(loss.detach())
            batches += 1
            step += 1
            if log_every and step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {float(loss.detach()):.4f} "
                      f"(seg {float(seg_loss.detach()):.4f} op {float(op_loss.detach()):.4f})")
        mean_loss = total / max(batches, 1)
        result.train_losses.append(mean_loss)

        vloss: Optional[float] = None
        if val_examples:
            vloss = _epoch_loss(
                router, val_examples, batch_size, device,
                policy_mode=policy_mode, lenpref_beta=lenpref_beta,
                anti_original_lambda=anti_original_lambda,
                op_class_weights=op_class_weights, op_focal_gamma=op_focal_gamma,
            )
            result.val_losses.append(vloss)

        prog_metrics: Optional[Dict[str, float]] = None
        run_prog = val_program_data is not None and (
            epoch % max(1, val_program_every) == 0 or epoch == epochs - 1)
        if run_prog:
            prog_metrics = evaluate_val_programs(
                router,
                val_program_data["questions"],       # type: ignore[arg-type]
                val_program_data["token_hiddens"],    # type: ignore[arg-type]
                val_program_data["valid_sets"],       # type: ignore[arg-type]
                k=val_topk, device=device, batch_size=max(64, batch_size),
            )
            result.val_cache_acc_epochs.append(epoch)
            result.val_cache_acc_at1.append(prog_metrics["val_cache_acc_at1"])
            result.val_cache_acc_atk.append(prog_metrics["val_cache_acc_atk"])
            result.val_rescue_at1.append(prog_metrics["val_rescue_at1"])
            result.val_rescue_atk.append(prog_metrics["val_rescue_atk"])

        # Best-checkpoint selection by the chosen metric.
        _cache_metric_key = {"val_cache_acc": "val_cache_acc_atk",
                             "val_cache_acc_at1": "val_cache_acc_at1",
                             "val_rescue_at1": "val_rescue_at1",
                             "val_rescue_atk": "val_rescue_atk"}.get(select_by)
        if _cache_metric_key is not None:
            score = prog_metrics[_cache_metric_key] if prog_metrics else float("-inf")
            if score > best_acc:
                best_acc = score
                result.best_state = _head_state_cpu(router)
                result.best_epoch = epoch
                result.best_metric = score
        elif vloss is not None and vloss < best_val:
            best_val = vloss
            result.best_state = _head_state_cpu(router)
            result.best_epoch = epoch
            result.best_metric = vloss

    # Restore the best (val) checkpoint; without validation keep the last epoch.
    if result.best_state is not None:
        router.load_state_dict(result.best_state, strict=False)
    return result


# --------------------------------------------------------------------------- #
# checkpointing (head only; the frozen encoder reloads itself from HF)
# --------------------------------------------------------------------------- #
def _router_meta(router: PolarRouter) -> dict:
    return {
        "num_layers": router.num_layers,
        "n_ops": router.n_ops,
        "d_model": router.d_model,
        "nheads": router.cross_attn.num_heads,
        "n_layer_blocks": router.layer_encoder.num_layers,
        "embed_dim": router.embed_dim,
        "embedding_model_name": router.embedding_model_name,
        "max_question_tokens": router.max_question_tokens,
        "ops": [op.value for op in DEFAULT_OPS],
    }


def save_checkpoint(router: PolarRouter, out_path, meta: Optional[dict] = None) -> Path:
    """Save the trainable head + architecture meta (encoder weights excluded)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head_state = {
        k: v for k, v in router.state_dict().items() if not k.startswith("encoder.")
    }
    payload = {"state_dict": head_state, "meta": _router_meta(router)}
    if meta:
        payload["train_meta"] = meta
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    ckpt_path,
    *,
    encoder: Optional[nn.Module] = None,
    embed_dim: Optional[int] = None,
    map_location="cpu",
) -> PolarRouter:
    """Rebuild a PolarRouter and load a saved head.

    By default rebuilds with the checkpoint's ``embed_dim`` (no encoder
    download, the test / offline path). Pass ``encoder=`` to attach a real encoder for
    inference; the encoder loads its own weights, so its keys are legitimately
    absent from the saved head state.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    kwargs = dict(
        num_layers=meta["num_layers"],
        n_ops=meta["n_ops"],
        d_model=meta["d_model"],
        nheads=meta["nheads"],
        n_layer_blocks=meta["n_layer_blocks"],
        embedding_model_name=meta["embedding_model_name"],
        max_question_tokens=meta["max_question_tokens"],
    )
    if encoder is not None:
        kwargs["encoder"] = encoder
    else:
        kwargs["embed_dim"] = embed_dim if embed_dim is not None else meta["embed_dim"]

    router = PolarRouter(**kwargs)
    missing, unexpected = router.load_state_dict(ckpt["state_dict"], strict=False)
    head_missing = [k for k in missing if not k.startswith("encoder.")]
    if head_missing:
        raise RuntimeError(f"checkpoint missing head params: {head_missing}")
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected params: {unexpected}")
    router.eval()
    return router


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the PolarRouter on MCTS supervision.")
    p.add_argument("--samples", required=True, nargs="+",
                   help="one or more merged_mcts_samples.json (concatenated; the "
                        "router trains on all difficulties combined, per the paper)")
    p.add_argument("--model", default="qwen3_8b",
                   help="target model in MODEL_REGISTRY (sets router D); default qwen3_8b")
    p.add_argument("--out", required=True, help="checkpoint output path (.pt)")
    # PoLar's released recipe defaults.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-paths-per-sample", type=int, default=50,
                   help="cap on valid paths per question (deterministic RNG sample above it)")
    # validation / best-checkpoint
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="per-difficulty holdout fraction for val-loss best-checkpoint")
    p.add_argument("--no-validation", action="store_true",
                   help="train on all samples and save the LAST epoch (no holdout)")
    p.add_argument("--select-by",
                   choices=["val_loss", "val_cache_acc", "val_cache_acc_at1"],
                   default="val_loss",
                   help="checkpoint selection metric. val_loss (default, unchanged) rewards "
                        "collapse-to-identity (min loss = predict majority op everywhere). "
                        "val_cache_acc (top-k any-hit) and val_cache_acc_at1 (top-1 hit only) "
                        "both reward decoding a KNOWN-MCTS-valid program instead; but val_cache_acc "
                        "can favour an UNDER-TRAINED epoch (near-uniform logits -> wider, more "
                        "diverse top-k beam -> more lucky cache hits) even as its top-1 gets worse "
                        "-- val_cache_acc_at1 is immune to that beam-width "
                        "confound and is the recommended reward-aligned choice.")
    p.add_argument("--val-topk", type=int, default=5,
                   help="k for val_cache_acc_atk (generation-free val pass@k metric)")
    p.add_argument("--val-program-every", type=int, default=1,
                   help="run the (batched) val_cache_acc metric every N epochs (last epoch "
                        "always included); >1 speeds up long runs")
    # speed
    p.add_argument("--encodings-on-device", action="store_true",
                   help="move the frozen encodings to the GPU once so training batches are "
                        "built on-device (removes the per-step CPU->GPU copy; ~1.5GB VRAM)")
    # optional ablation knobs (default recipe = plain 'polar', all off)
    p.add_argument("--policy-mode", choices=["polar", "polar_lenpref"], default="polar",
                   help="polar_lenpref reweights each example by exp(-beta*path_len)")
    p.add_argument("--lenpref-beta", type=float, default=0.05,
                   help="polar_lenpref length-preference beta")
    p.add_argument("--anti-original-lambda", type=float, default=0.0,
                   help="penalty on mean KEEP-prob for samples whose identity path is not valid")
    p.add_argument("--per-sample-weight-normalize", action="store_true",
                   help="scale each sample's examples by 1/n_paths")
    p.add_argument("--reweight-original-path-if-shorter-valid", action="store_true",
                   help="downweight the identity/full-depth path to --original-path-weight "
                        "when a strictly-shorter valid path exists (PoLar anti-collapse lever)")
    p.add_argument("--original-path-weight", type=float, default=1.0,
                   help="weight for the identity path when reweighting is on (PoLar uses 0.30)")
    p.add_argument("--drop-original-path-if-shorter-valid", action="store_true",
                   help="DROP the identity path from targets when a strictly-shorter valid path "
                        "exists (stronger PoLar anti-collapse lever). Mutually exclusive with "
                        "--reweight-original-path-if-shorter-valid (ignored while that is set).")
    p.add_argument("--keep-original-prob", type=float, default=0.0,
                   help="soft-drop: keep the identity path with this probability (0.0 = hard drop)")
    p.add_argument("--label-mode", choices=["multi", "shortest"], default="multi",
                   help="multi (default): one example per valid path (PoLar labelling, "
                        "original_path_weight/drop_original_path apply). shortest: collapse each "
                        "sample to its single SHORTEST valid path -- an ablation that collapsed the "
                        "router to identity before multi-path labelling was adopted; see "
                        "build_examples' own docstring. original-path-weight/"
                        "drop-original-path-if-shorter-valid are ignored in this mode.")
    p.add_argument("--segment-cap", choices=["all", "edit-only"], default="all",
                   help="all (default, unchanged): MAX_SEGMENT_LEN caps every segment, including KEEP "
                        "runs -- a long uninterrupted keep stretch is chunked into consecutive <=4-layer "
                        "pieces purely to fit the router's fixed-length segment representation (PoLar "
                        "Sec 3.1's literal text). edit-only (an ablation testing a specific hypothesis): "
                        "only SKIP/REPEAT segments stay capped; each maximal KEEP run merges into ONE "
                        "segment however long, so seg_flip boundaries only fire where the operation "
                        "actually changes. Execution-equivalent either way (KEEP takes no params) -- "
                        "this changes the router's training TARGET only, not the search space. See "
                        "program_from_layer_path(cap_keep=...)'s own docstring.")
    p.add_argument("--strict-repeat-2x", dest="strict_repeat_2x", action="store_true", default=True,
                   help="Drop: repeat-count != 2x paths are dropped, not parsed (default: on, "
                        "matches PoLar's own parser -- current/main-recipe behavior, unchanged).")
    p.add_argument("--no-strict-repeat-2x", dest="strict_repeat_2x", action="store_false",
                   help="Crop instead of Drop: truncate a repeat run to exactly 2x rather than "
                        "dropping the path (see analysis/router_recipe_sweep's Drop-CE/Crop-CE "
                        "comparison; ported here as a plain option, same underlying parser flag "
                        "program_from_layer_path/build_examples already take).")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="DR.LLM-style focal loss gamma for the op-head CE (0.0 = plain CE, default/"
                        "current behavior, unchanged). When >0, op_class_weights are the DR.LLM-exact "
                        "class-balanced weights (Cui et al. effective-number-of-samples, beta=0.999) "
                        "computed from this run's own train op-label counts -- same computation as "
                        "analysis/router_recipe_sweep/sweep_router.py's --focal-gamma, ported "
                        "verbatim, not reimplemented.")
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "linear"], default="none",
                   help="LR schedule (warmup then decay); PoLar sample recipe uses cosine")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear warmup steps before decay (PoLar sample recipe uses 10)")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="use only the first N samples per file")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    return p


def _resolve_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = _build_arg_parser().parse_args(argv)
    device = _resolve_device(args.device)

    router = PolarRouter.for_model(args.model)
    anti_original = args.anti_original_lambda > 0.0
    if args.select_by in ("val_cache_acc", "val_cache_acc_at1") and args.no_validation:
        raise SystemExit(f"--select-by {args.select_by} needs a validation split (drop --no-validation).")

    # Per-DIFFICULTY split: hold out the last val_frac of EACH file's samples so
    # the val set is balanced across difficulties (build examples per split).
    train_samples: List[dict] = []
    val_samples: List[dict] = []
    for path in args.samples:
        s = load_supervision(path)
        if args.limit is not None:
            s = s[: args.limit]
        if args.no_validation:
            train_samples.extend(s)
        else:
            tr, va = split_samples_train_val(s, args.val_frac)
            train_samples.extend(tr)
            val_samples.extend(va)

    build_kwargs = dict(
        max_paths_per_sample=args.max_paths_per_sample,
        per_sample_weight_normalize=args.per_sample_weight_normalize,
        reweight_original_path=args.reweight_original_path_if_shorter_valid,
        original_path_weight=args.original_path_weight,
        drop_original_path=args.drop_original_path_if_shorter_valid,
        keep_original_prob=args.keep_original_prob,
        anti_original=anti_original,
        strict_repeat_2x=args.strict_repeat_2x,
        shortest_only=(args.label_mode == "shortest"),
        cap_keep=(args.segment_cap == "all"),
        seed=args.seed,
    )
    examples = build_examples(train_samples, router.num_layers, **build_kwargs)
    val_examples = build_examples(val_samples, router.num_layers, **build_kwargs) if val_samples else []

    n_train_q = len({e.question for e in examples})
    n_val_q = len({e.question for e in val_examples})
    print(f"Loaded {len(train_samples)} train + {len(val_samples)} val samples "
          f"from {len(args.samples)} file(s).")
    print(f"Multi-path examples: {len(examples)} train (from {n_train_q} questions) + "
          f"{len(val_examples)} val (from {n_val_q} questions).")
    if not examples:
        raise SystemExit("No trainable examples (every sample lacked a valid program).")

    print(f"Encoding {n_train_q + n_val_q} unique questions with {router.embedding_model_name} "
          f"(frozen; shared across each question's paths)...")
    # Move to `device` BEFORE encoding: the frozen encoder's forward over every
    # question is the heavy step. Left on CPU it maxes the CPU while the GPU sits
    # idle; on GPU it's a fast one-shot. encode_examples stores hidden states back
    # on CPU, so the later train() batches still move to device normally.
    router.to(device)
    encode_examples(router, examples, batch_size=args.encode_batch_size)
    if val_examples:
        encode_examples(router, val_examples, batch_size=args.encode_batch_size)

    # Reward-aligned val metric data (generation-free): per val question, its set of
    # MCTS-valid executed paths + one shared token_hidden. Drives val_cache_acc and
    # (opt-in) checkpoint selection. Built from the raw val_samples, not the examples.
    val_program_data: Optional[Dict[str, object]] = None
    if val_examples:
        valid_sets: Dict[str, set] = {}
        for s in val_samples:
            q = s.get("question")
            vs = {tuple(int(x) for x in p) for p in (s.get("final_valid_transitions") or []) if p}
            if q is not None and vs:
                valid_sets[q] = vs
        token_hiddens: Dict[str, torch.Tensor] = {}
        for e in val_examples:
            if e.token_hidden is not None and e.question not in token_hiddens:
                token_hiddens[e.question] = e.token_hidden
        vq = [q for q in token_hiddens if q in valid_sets]
        if vq:
            val_program_data = {"questions": vq, "token_hiddens": token_hiddens,
                                "valid_sets": valid_sets}
            print(f"Val-program metric on {len(vq)} val questions "
                  f"(select_by={args.select_by}, val_topk={args.val_topk}).")

    # DR.LLM-exact class-balanced focal weights, computed once from this run's own
    # TRAIN op-label counts -- same computation as
    # analysis/router_recipe_sweep/sweep_router.py's --focal-gamma, ported
    # verbatim (see that script for the original). None (default) unless
    # --focal-gamma > 0, exactly matching the pre-existing plain-CE default.
    op_class_weights = None
    if args.focal_gamma > 0:
        from collections import Counter

        op_counts: Counter = Counter()
        for e in examples:
            for v in e.op_labels[e.op_labels != -100].tolist():
                op_counts[v] += 1
        cb_beta = 0.999
        counts = [op_counts.get(i, 1) for i in range(3)]
        eff_num = [(1 - cb_beta**c) / (1 - cb_beta) for c in counts]
        op_class_weights = torch.tensor([1.0 / e for e in eff_num], dtype=torch.float32)
        op_class_weights = (op_class_weights / op_class_weights.mean()).to(device)
        print(f"focal_gamma={args.focal_gamma} op_counts={counts} "
              f"class_weights={op_class_weights.tolist()}")

    result = train(
        router,
        examples,
        val_examples=val_examples or None,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        policy_mode=args.policy_mode,
        lenpref_beta=args.lenpref_beta,
        anti_original_lambda=args.anti_original_lambda,
        op_class_weights=op_class_weights,
        op_focal_gamma=args.focal_gamma,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        device=device,
        seed=args.seed,
        log_every=max(1, len(examples) // max(1, args.batch_size)),
        val_program_data=val_program_data,
        select_by=args.select_by,
        val_topk=args.val_topk,
        val_program_every=args.val_program_every,
        move_encodings_to_device=args.encodings_on_device,
    )
    if result.best_epoch is not None:
        sel = result.select_by
        mval = result.best_metric
        print(f"Selected epoch {result.best_epoch} by {sel}"
              + (f" = {mval:.4f}" if mval is not None else "")
              + f" (final train loss {result.train_losses[-1]:.4f}, first {result.train_losses[0]:.4f}).")
        if result.val_cache_acc_atk:
            print(f"  evaluated epochs                : {result.val_cache_acc_epochs}")
            print(f"  val_cache_acc_at1  per evaluated epoch (top-1 known-valid; "
                  f"immune to beam width): {[round(x, 4) for x in result.val_cache_acc_at1]}")
            print(f"  val_cache_acc@{args.val_topk} per evaluated epoch (any-of-{args.val_topk}; "
                  f"can favour an under-trained/diverse epoch): "
                  f"{[round(x, 4) for x in result.val_cache_acc_atk]}")
    else:
        print(f"Final epoch mean loss: {result.train_losses[-1]:.4f} "
              f"(first: {result.train_losses[0]:.4f}).")

    out = save_checkpoint(router, args.out, meta={
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "val_cache_acc_epochs": result.val_cache_acc_epochs,
        "val_cache_acc_at1": result.val_cache_acc_at1,
        "val_cache_acc_atk": result.val_cache_acc_atk,
        "best_epoch": result.best_epoch,
        "best_metric": result.best_metric,
        "select_by": result.select_by,
        "val_topk": args.val_topk,
        "model": args.model,
        "policy_mode": args.policy_mode,
        "max_paths_per_sample": args.max_paths_per_sample,
        "n_examples": len(examples),
    })
    print(f"Saved router checkpoint -> {out}")
    return out


if __name__ == "__main__":
    main()
