"""Checks specific claims from PoLar's own diagnostic study against our
MCTS data: skip/loop/combined accuracy, accuracy vs. depth budget, mean
executed program depth, valid-program coverage vs. recurrence budget,
per-difficulty recurrence/skip requirement, accuracy vs. executed depth,
and segment-length/recurrence structure. Every computation here is
confirmed against PoLar's own published text -- see each function's
docstring for the exact confirmed definition and, where the paper states a
number to check against, the comparison value.

Skip/Loop/Skip&Loop (`compute_skip_loop_accuracy`) are POST-HOC filters of
the one joint skip+repeat MCTS search we run (PoLar's own paper does the
same, not three separate constrained searches), NOT a re-derivation of
`insights.op_class_coverage`'s existing table (that one's "identity" class
means "the identity path itself showed up in final_valid_transitions",
which is a different, narrower thing than "was this sample originally
correct" -- conflating the two would silently undercount here).

`compute_segment_length_distribution`'s matrix is DIAGONAL-ONLY (see its
own docstring for why): off-diagonal ("fragmented"/non-contiguous edit)
cells can only arise from merging multiple separate edit actions into one
"segment", a merging rule PoLar's paper never states and our data (final
path only, no action log) can't recover. The length-1/2/3/4 diagonal needs
no such rule and remains directly checkable against PoLar's own
unambiguous stated text ("54.5% of segments consist of a single layer, and
over two-thirds contain at most two consecutive layers").

This module is pure computation, no rendering -- given a `merged_mcts_
samples.json`-shaped `groups` dict (see `re_polar.mcts.analysis.loader.build_groups`)
and `num_layers`, every `compute_*` function returns a plain dict of
numbers. `compute_findings` closes the comparison out with a one-line
confirm/refute check per this repo's own Findings F1-F4 (which mirror
PoLar's own diagnostic Findings 1-4), computed from the same underlying
data as the functions above.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .segments import decode, identity, op_class, shortest_valid

# Depth-budget levels (% of D) swept for compute_accuracy_by_depth_budget.
DEPTH_BUDGETS_PCT = [90, 95, 100, 105, 110, 115]


# --------------------------------------------------------------------------- #
# Data computation -- one function per check, operating on a single model's
# {difficulty_label: [sample, ...]} groups.
# --------------------------------------------------------------------------- #
def compute_skip_loop_accuracy(groups: Dict[str, List[dict]], num_layers: int) -> Dict[str, dict]:
    """Base/Skip/Loop/Skip&Loop/Gain, per difficulty. Base = identity
    already correct. Skip = identity-OR-skip-only-program exists (the
    subset of the joint search reachable by a skip-only-restricted space).
    Loop = identity-OR-repeat-only-program exists. Skip&Loop = identity-OR-
    any-valid-program exists (== rescue_breakdown's solved_rate)."""
    out = {}
    for g, samples in groups.items():
        n = len(samples)
        base_n = skip_n = loop_n = full_n = 0
        for s in samples:
            base_ok = s.get("initial_transition_metric") == 1.0
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            classes = {op_class(p, num_layers) for p in valid}
            if base_ok:
                base_n += 1
            if base_ok or "skip-only" in classes:
                skip_n += 1
            if base_ok or "repeat-only" in classes:
                loop_n += 1
            if base_ok or valid:
                full_n += 1
        base_pct = base_n / n * 100 if n else 0.0
        skiploop_pct = full_n / n * 100 if n else 0.0
        out[g] = {
            "n": n,
            "base": base_pct,
            "skip": skip_n / n * 100 if n else 0.0,
            "loop": loop_n / n * 100 if n else 0.0,
            "skiploop": skiploop_pct,
            "gain": skiploop_pct - base_pct,
        }
    return out


def compute_accuracy_by_depth_budget(
    groups: Dict[str, List[dict]], num_layers: int
) -> Dict[str, dict]:
    """Accuracy vs. depth budget (fraction of questions solvable by SOME
    candidate -- identity or a found program -- with length <= budget% * D).
    Budget is a MAXIMUM (an inequality)."""
    out = {}
    for g, samples in groups.items():
        n = len(samples)
        best_lens = []
        for s in samples:
            base_ok = s.get("initial_transition_metric") == 1.0
            cands = [num_layers] if base_ok else []
            cands += [len(p) for p in s.get("final_valid_transitions", [])]
            best_lens.append(min(cands) if cands else None)
        curve = []
        for b in DEPTH_BUDGETS_PCT:
            limit = b / 100 * num_layers
            cnt = sum(1 for L in best_lens if L is not None and L <= limit)
            curve.append(cnt / n if n else 0.0)
        base_acc = (
            sum(1 for s in samples if s.get("initial_transition_metric") == 1.0) / n if n else 0.0
        )
        out[g] = {"budgets": DEPTH_BUDGETS_PCT, "accuracy": curve, "base_accuracy": base_acc}
    return out


def compute_mean_executed_depth(groups: Dict[str, List[dict]], num_layers: int) -> Dict[str, dict]:
    """Mean executed depth (% of D) of the shortest valid program, split
    C->C (base already correct) vs. W->C (rescued), plus a "unique layers
    touched" overlay = mean # of UNIQUE layers touched (distinct from
    total executed depth once a program repeats any layer)."""
    id_path = identity(num_layers)
    out = {}
    for g, samples in groups.items():
        cc_len, cc_uniq, wc_len, wc_uniq = [], [], [], []
        for s in samples:
            base_ok = s.get("initial_transition_metric") == 1.0
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if base_ok:
                cands = [id_path] + valid
                rep = shortest_valid(cands, num_layers)
                cc_len.append(len(rep))
                cc_uniq.append(len(set(rep)))
            elif valid:
                rep = shortest_valid(valid, num_layers)
                wc_len.append(len(rep))
                wc_uniq.append(len(set(rep)))

        def _mean_pct(xs):
            return (sum(xs) / len(xs) / num_layers * 100) if xs else None

        out[g] = {
            "cc_depth_pct": _mean_pct(cc_len),
            "cc_unique_pct": _mean_pct(cc_uniq),
            "n_cc": len(cc_len),
            "wc_depth_pct": _mean_pct(wc_len),
            "wc_unique_pct": _mean_pct(wc_uniq),
            "n_wc": len(wc_len),
        }
    return out


def compute_valid_coverage_by_recurrence_budget(
    groups: Dict[str, List[dict]], num_layers: int, max_r: int = 8
) -> dict:
    """P(a valid program exists) vs. max additional latent execution steps
    via recurrence: budget r = NET extra executed layers relative to the
    model depth D, i.e. max(0, len(path) - D) -- NOT len(path)-len(set(path))
    (which would measure repeat-caused overhead relative to a program's OWN
    unique-layer count, ignoring any length saved by a simultaneous skip in
    the same program). Concretely: a program that skips 5 layers and
    repeats a different 8-layer block nets len(path) = D+3, so it counts as
    budget 3 -- "the total additional latent steps the model took", not
    "how many of those steps came specifically from recurrence". Cumulative
    ("at most r" -- the exact-length reading is NOT monotonic and doesn't
    match PoLar's smooth rising/plateauing curve; this cumulative reading
    does). Pooled across all difficulties for this model."""
    all_samples = [s for v in groups.values() for s in v]
    n = len(all_samples)
    best_len = []
    for s in all_samples:
        base_ok = s.get("initial_transition_metric") == 1.0
        cands = [num_layers] if base_ok else []
        cands += [len(p) for p in s.get("final_valid_transitions", [])]
        best_len.append(min(cands) if cands else None)
    curve = []
    for r in range(0, max_r + 1):
        target = num_layers + r
        cnt = sum(1 for L in best_len if L is not None and L <= target)
        curve.append(cnt / n if n else 0.0)
    return {"budgets": list(range(0, max_r + 1)), "p_valid": curve, "n": n}


def compute_recurrence_and_skip_requirement(
    groups: Dict[str, List[dict]], num_layers: int
) -> Dict[str, dict]:
    """P(require recurrence) / P(require skip), per difficulty. Settled
    definition: of every question SOLVABLE at all (base_ok OR >=1 valid
    program -- matches PoLar's own y-axis label "P(require | solvable)"),
    the share where the question was NOT already correct (base_ok=False)
    AND at least one RESCUING program uses that op somewhere -- not
    exclusive, a program using both skip and repeat counts toward BOTH
    shares."""
    out = {}
    for g, samples in groups.items():
        solved = need_recurrence = need_skip = 0
        for s in samples:
            base_ok = s.get("initial_transition_metric") == 1.0
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if not base_ok and not valid:
                continue
            solved += 1
            if base_ok:
                continue  # identity already valid -- never counts toward either share
            classes = {op_class(p, num_layers) for p in valid}
            if "repeat-only" in classes or "both" in classes:
                need_recurrence += 1
            if "skip-only" in classes or "both" in classes:
                need_skip += 1
        out[g] = {
            "p_require_recurrence": (need_recurrence / solved) if solved else None,
            "p_require_skip": (need_skip / solved) if solved else None,
            "n_solved": solved,
        }
    return out


def compute_accuracy_by_executed_depth(
    groups: Dict[str, List[dict]],
    num_layers: int,
    n_buckets: int = 16,
    lo_pct: float = 50.0,
    hi_pct: float = 140.0,
) -> Dict[str, list]:
    """Accuracy vs. total executed depth (% of D), bucketed. "Average
    accuracy" = pooled over ALL attempted programs (valid AND invalid, not
    just valid ones -- a single valid program's own "accuracy" is trivially
    1, so this must be a rate over many attempts per bucket) = n_valid /
    (n_valid + n_invalid) within each depth-% bucket, per difficulty."""
    bucket_w = (hi_pct - lo_pct) / n_buckets
    out = {}
    for g, samples in groups.items():
        valid_c, invalid_c = defaultdict(int), defaultdict(int)
        for s in samples:
            for p in s.get("final_valid_transitions", []):
                b = int((len(p) / num_layers * 100 - lo_pct) // bucket_w)
                valid_c[b] += 1
            for p in s.get("final_invalid_transitions", []):
                b = int((len(p) / num_layers * 100 - lo_pct) // bucket_w)
                invalid_c[b] += 1
        points = []
        for b in sorted(set(valid_c) | set(invalid_c)):
            v, iv = valid_c.get(b, 0), invalid_c.get(b, 0)
            tot = v + iv
            if tot == 0 or b < 0 or b >= n_buckets:
                continue
            points.append(
                {"depth_pct": lo_pct + (b + 0.5) * bucket_w, "accuracy": v / tot, "n": tot}
            )
        out[g] = points
    return out


def compute_segment_length_distribution(
    groups: Dict[str, List[dict]], num_layers: int, max_span: int = 4
) -> dict:
    """Segment-length histogram, over the shortest-valid representative
    program per solved question, pooled across all difficulties --
    DIAGONAL-ONLY (row==col). Off-diagonal (row < col, "gappy"/fragmented)
    cells can ONLY arise from MERGING two or more separate, non-adjacent
    edit actions into one measured "segment" -- a single atomic
    skip/repeat action is always fully contiguous, so alone it can only
    ever produce row==col. PoLar's paper never states its merging rule
    (how close two separate edits must be before counting as "one
    segment"), and our data has no recorded action sequence to recover it
    from either (only the final flattened path). So off-diagonal numbers
    are unrecoverable and are not computed at all, rather than shown as a
    guess. Diagonal cells need no such rule -- "how long is this one
    clean, unbroken edited stretch" is exactly what `decode()` already
    gives, unambiguously, same trusted tool as
    `compute_segment_recurrence_distribution`. Row marginals
    (length-1/2/3/4 shares) remain directly checkable against PoLar's own
    stated 54.5% / >66.7%."""
    counts = Counter()  # length -> count
    total = 0
    for samples in groups.values():
        for s in samples:
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if not valid:
                continue
            rep = shortest_valid(valid, num_layers)
            prog = decode(rep, num_layers)
            for seg in prog.segments:
                if seg.op.value == "keep":
                    continue
                length = min(len(seg), max_span)
                counts[length] += 1
                total += 1
    lengths = list(range(1, max_span + 1))
    matrix = [
        [(counts.get(row, 0) / total if total else 0.0) if col == row else None for col in lengths]
        for row in lengths
    ]
    return {str(k): (counts.get(k, 0) / total if total else 0.0) for k in lengths} | {
        "n": total,
        "le2_pct": ((counts.get(1, 0) + counts.get(2, 0)) / total * 100) if total else 0.0,
        "matrix": matrix,
        "row_labels": [str(v) for v in lengths],
        "col_labels": [str(v) for v in lengths],
    }


def compute_segment_recurrence_distribution(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    """Max recurrence count per segment (0/1/2+), over the SHORTEST-VALID
    representative program per solved question, pooled across all
    difficulties (an assumption about PoLar's own pooling -- not something
    its published text explicitly confirms, flagged here since it isn't)."""
    counts = Counter()
    total = 0
    for samples in groups.values():
        for s in samples:
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if not valid:
                continue
            rep = shortest_valid(valid, num_layers)
            prog = decode(rep, num_layers)
            max_times = max(
                [seg.times - 1 for seg in prog.segments if seg.op.value == "repeat"], default=0
            )
            counts[min(max_times, 2)] += 1
            total += 1
    return {
        "0": counts.get(0, 0) / total if total else 0.0,
        "1": counts.get(1, 0) / total if total else 0.0,
        "2": counts.get(2, 0) / total if total else 0.0,
        "n": total,
    }


def _shorter_fraction(
    groups: Dict[str, List[dict]], num_layers: int, want_cc: bool
) -> Optional[float]:
    """Fraction of C→C (want_cc=True) or W→C (want_cc=False) samples whose
    shortest-valid representative program executes STRICTLY FEWER than D
    layers -- the exact per-question metric PoLar's own Finding 2 text
    quotes ("among inputs already solved correctly (C→C), 75.5% admit
    shorter valid programs... for W→C, 36.2% admit shorter programs")."""
    id_path = identity(num_layers)
    n = shorter = 0
    for samples in groups.values():
        for s in samples:
            base_ok = s.get("initial_transition_metric") == 1.0
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if want_cc:
                if not base_ok:
                    continue
                rep = shortest_valid([id_path] + valid, num_layers)
            else:
                if base_ok or not valid:
                    continue
                rep = shortest_valid(valid, num_layers)
            n += 1
            if len(rep) < num_layers:
                shorter += 1
    return (shorter / n) if n else None


def _diff_sort_key(g):
    try:
        return (0, int(g))
    except ValueError:
        return (1, g)


def compute_findings(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    """One confirm/refute signal per this repo's own Findings F1-F4 (which
    mirror PoLar's own diagnostic Findings 1-4), computed from the SAME
    underlying data as the functions above (re-calls them -- cheap, this
    data is small), so a comparison run can put PoLar's own claim text
    next to a real check against our data, not just restate the claim."""
    skip_loop = compute_skip_loop_accuracy(groups, num_layers)
    recurrence_coverage = compute_valid_coverage_by_recurrence_budget(groups, num_layers)
    recurrence_req = compute_recurrence_and_skip_requirement(groups, num_layers)
    seg_len = compute_segment_length_distribution(groups, num_layers)
    seg_rec = compute_segment_recurrence_distribution(groups, num_layers)

    # F1: loop > skip, skiploop >= both, per difficulty.
    diffs = sorted(skip_loop, key=_diff_sort_key)
    loop_beats_skip = sum(1 for g in diffs if skip_loop[g]["loop"] > skip_loop[g]["skip"])
    combined_best = sum(
        1
        for g in diffs
        if skip_loop[g]["skiploop"] >= max(skip_loop[g]["skip"], skip_loop[g]["loop"])
    )
    f1_note = (
        f"loop>skip in {loop_beats_skip}/{len(diffs)} difficulties, "
        f"Skip&Loop best-or-tied in {combined_best}/{len(diffs)}"
    )

    # F2 ("Occam's razor"): fraction of C->C / W->C admitting a strictly shorter program.
    cc_frac = _shorter_fraction(groups, num_layers, want_cc=True)
    wc_frac = _shorter_fraction(groups, num_layers, want_cc=False)
    f2_note = (
        (
            f"C->C: {cc_frac * 100:.1f}% admit a shorter program (PoLar: 75.5%)"
            if cc_frac is not None
            else "C->C: no data"
        )
        + " / "
        + (
            f"W->C: {wc_frac * 100:.1f}% admit a shorter program (PoLar: 36.2%)"
            if wc_frac is not None
            else "W->C: no data"
        )
    )

    # F3: p_valid monotonic non-decreasing in recurrence budget; require-recurrence/skip
    # trending up from the easiest to the hardest difficulty.
    p_valid = recurrence_coverage["p_valid"]
    monotonic = all(p_valid[i] <= p_valid[i + 1] + 1e-9 for i in range(len(p_valid) - 1))
    easiest, hardest = diffs[0], diffs[-1]
    rec_up = (recurrence_req.get(hardest, {}).get("p_require_recurrence") or 0) >= (
        recurrence_req.get(easiest, {}).get("p_require_recurrence") or 0
    )
    skip_up = (recurrence_req.get(hardest, {}).get("p_require_skip") or 0) >= (
        recurrence_req.get(easiest, {}).get("p_require_skip") or 0
    )
    f3_note = (
        f"P(valid) monotonic in recurrence budget: {monotonic}; "
        f"require-recurrence DM-{easiest}->DM-{hardest}: "
        f"{(recurrence_req.get(easiest, {}).get('p_require_recurrence') or 0) * 100:.1f}%->"
        f"{(recurrence_req.get(hardest, {}).get('p_require_recurrence') or 0) * 100:.1f}% "
        f"({'up' if rec_up else 'down'}); require-skip "
        f"{(recurrence_req.get(easiest, {}).get('p_require_skip') or 0) * 100:.1f}%->"
        f"{(recurrence_req.get(hardest, {}).get('p_require_skip') or 0) * 100:.1f}% "
        f"({'up' if skip_up else 'down'})"
    )

    # F4: len-1 dominant, len<=2 majority, at most-one-recurrence dominant.
    at_most_one_rep = seg_rec["0"] + seg_rec["1"]
    f4_note = (
        f"len-1: {seg_len['1'] * 100:.1f}% (PoLar: 54.5%); len<=2: {seg_len['le2_pct']:.1f}% "
        f"(PoLar: >66.7%); at-most-one-recurrence: {at_most_one_rep * 100:.1f}%"
    )

    # Verdicts -- confirmed / partial / not_confirmed, one per finding.
    ratio1 = loop_beats_skip / len(diffs) if diffs else 0
    combined_ratio1 = combined_best / len(diffs) if diffs else 0
    if ratio1 == 1 and combined_ratio1 == 1:
        v1 = "confirmed"
    elif ratio1 >= 0.6 and combined_ratio1 >= 0.6:
        v1 = "partial"
    else:
        v1 = "not_confirmed"

    frac2 = [v for v in (cc_frac, wc_frac) if v is not None]
    if not frac2:
        v2 = "no_data"
    elif all(v > 0.5 for v in frac2):
        v2 = "confirmed"
    elif any(v > 0.5 for v in frac2):
        v2 = "partial"
    else:
        v2 = "not_confirmed"

    if monotonic and rec_up and skip_up:
        v3 = "confirmed"
    elif monotonic or rec_up or skip_up:
        v3 = "partial"
    else:
        v3 = "not_confirmed"

    le2 = seg_len["le2_pct"]
    if le2 > 66.7 and at_most_one_rep > 0.5:
        v4 = "confirmed"
    elif le2 > 50 or at_most_one_rep > 0.5:
        v4 = "partial"
    else:
        v4 = "not_confirmed"

    return {
        "1": {
            "loop_beats_skip": loop_beats_skip,
            "n": len(diffs),
            "combined_best": combined_best,
            "note": f1_note,
            "verdict": v1,
        },
        "2": {"cc_frac": cc_frac, "wc_frac": wc_frac, "note": f2_note, "verdict": v2},
        "3": {
            "monotonic": monotonic,
            "rec_up": rec_up,
            "skip_up": skip_up,
            "note": f3_note,
            "verdict": v3,
        },
        "4": {
            "len1_pct": seg_len["1"] * 100,
            "le2_pct": seg_len["le2_pct"],
            "at_most_one_rep": at_most_one_rep * 100,
            "note": f4_note,
            "verdict": v4,
        },
    }
