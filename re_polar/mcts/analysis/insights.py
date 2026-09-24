"""Registered insights, metrics computed directly from MCTS-discovered
programs (`merged_mcts_samples.json`), generalized to any model via
`num_layers`.

Every insight: `fn(groups: dict[str, list[sample]], num_layers: int) -> dict`,
`groups` = {group_label: [merged_mcts_samples.json sample dict, ...]}.
"""

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .registry import insight
from .segments import (
    decode,
    edit_signature,
    entropy_bits,
    gini,
    identity,
    layer_freq,
    n_edits,
    op_class,
    shortest_valid,
    sig_str,
    third,
    topk_coverage_share,
)


def _representative_paths(samples: List[dict], num_layers: int):
    """One path per solved question: shortest_valid() over its
    final_valid_transitions. Skips questions with 0 valid programs."""
    out = []
    for sample in samples:
        valid = [tuple(p) for p in sample.get("final_valid_transitions", [])]
        if valid:
            out.append(shortest_valid(valid, num_layers))
    return out


# --------------------------------------------------------------------------- #
# menu_concentration: how concentrated the search's discovered menu is
# --------------------------------------------------------------------------- #
def _ranked_coverage(samples, num_layers):
    prog_qids = defaultdict(set)
    id_ = identity(num_layers)
    for i, s in enumerate(samples):
        qid = s.get("sample_info", {}).get("query_id", i)
        for p in {tuple(p) for p in s.get("final_valid_transitions", [])}:
            if p != id_:
                prog_qids[p].add(qid)
    cover = sorted(((len(qs), p) for p, qs in prog_qids.items()), reverse=True)
    return prog_qids, cover


@insight(
    "menu_concentration",
    "Menu concentration (search-discovered non-identity programs)",
    "table",
    "How much of the solved population is covered by identity, the single "
    "most-covering non-identity program, and the top 10, plus a cumulative "
    "coverage curve for the full ranked list.",
)
def menu_concentration(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    id_ = identity(num_layers)
    table, curves = {}, {}
    for label, samples in groups.items():
        n_solved = sum(
            1
            for s in samples
            if s.get("initial_transition_metric") == 1.0 or s.get("final_valid_transitions")
        )
        if n_solved == 0:
            table[label] = {"solved": 0}
            curves[label] = []
            continue
        base_ok = sum(1 for s in samples if s.get("initial_transition_metric") == 1.0)
        prog_qids, cover = _ranked_coverage(samples, num_layers)
        top1 = cover[0][0] if cover else 0
        top10 = set()
        for _, p in cover[:10]:
            top10 |= prog_qids[p]
        n_distinct = len(prog_qids)
        unique1 = sum(1 for qs in prog_qids.values() if len(qs) == 1)
        mean_edit = (
            sum(
                len({tuple(p) for p in s.get("final_valid_transitions", [])} - {id_})
                for s in samples
            )
            / n_solved
        )
        table[label] = {
            "solved": n_solved,
            "base_ok_share": base_ok / n_solved,
            "menu_share": top1 / n_solved,
            "top10_share": len(top10) / n_solved,
            "unique1_share": (unique1 / n_distinct) if n_distinct else 0.0,
            "mean_edit_programs": mean_edit,
            "n_distinct_programs": n_distinct,
        }
        covered = set()
        curve = []
        for _, p in cover:
            covered |= prog_qids[p]
            curve.append(len(covered) / n_solved)
        curves[label] = curve
    return {"table": table, "coverage_curve": curves}


# --------------------------------------------------------------------------- #
# op_mix: segment op/length/position distribution + action-space-normalized
# repeat:skip ratio (the action-cardinality-confound fix)
# --------------------------------------------------------------------------- #
@insight(
    "op_mix",
    "Segment op-mix (skip/keep/repeat) and position distribution",
    "stackedBar",
    "Op/length/position distribution over every final_valid_transitions "
    "path (ALL valid instances, not just shortest-valid, the raw search "
    "landscape), plus the action-cardinality-normalized repeat:skip ratio "
    "(re_polar/mcts/search.py offers K_repeat distinct REPEAT actions per "
    "(start,length) vs. exactly 1 SKIP action, so a raw frequency count is "
    "not comparable across --max-repeat-times configs).",
)
def op_mix(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    op_dist, pos_dist, normalized = {}, {}, {}
    for label, samples in groups.items():
        op_counts, pos_counts, times_counts = Counter(), Counter(), Counter()
        for sample in samples:
            for path in sample.get("final_valid_transitions", []):
                prog = decode(tuple(path), num_layers)
                for seg in prog.segments:
                    op_counts[seg.op.value] += 1
                    pos_counts[third(seg.start, num_layers)] += 1
                    if seg.op.value == "repeat":
                        times_counts[seg.times] += 1
        total_segs = sum(op_counts.values())
        op_dist[label] = {k: v / total_segs for k, v in op_counts.items()} if total_segs else {}
        total_pos = sum(pos_counts.values())
        pos_dist[label] = {k: v / total_pos for k, v in pos_counts.items()} if total_pos else {}

        skip_n, repeat_n = op_counts.get("skip", 0), op_counts.get("repeat", 0)
        nonkeep = skip_n + repeat_n
        skip_cond = skip_n / nonkeep if nonkeep else 0.0
        repeat_cond = repeat_n / nonkeep if nonkeep else 0.0
        max_times = max(times_counts) if times_counts else 2
        k_repeat = max(1, max_times - 1)
        repeat_per_action = repeat_cond / k_repeat
        raw_ratio = (repeat_cond / skip_cond) if skip_cond else float("inf")
        norm_ratio = (repeat_per_action / skip_cond) if skip_cond else float("inf")
        normalized[label] = {
            "skip_conditional": skip_cond,
            "repeat_conditional": repeat_cond,
            "repeat_cardinality_k": k_repeat,
            "raw_repeat_to_skip_ratio": raw_ratio,
            "cardinality_normalized_repeat_to_skip_ratio": norm_ratio,
        }
    return {"op_distribution": op_dist, "pos_distribution": pos_dist, "normalized": normalized}


# --------------------------------------------------------------------------- #
# segmentation_structure: op-agnostic edit-signature entropy/gini/top-k
# --------------------------------------------------------------------------- #
@insight(
    "segmentation_structure",
    "Segmentation-structure diversity (op-agnostic edit signatures)",
    "line",
    "Op-agnostic partition signature (ordered (start,length) of non-KEEP "
    "segments) per valid program; entropy/gini/top-k-coverage of how "
    "concentrated the search's distinct segmentations are.",
)
def segmentation_structure(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table, topk = {}, {}
    for label, samples in groups.items():
        sig_counts = Counter()
        for sample in samples:
            for path in sample.get("final_valid_transitions", []):
                prog = decode(tuple(path), num_layers)
                sig_counts[edit_signature(prog)] += 1
        sorted_counts = [c for _, c in sig_counts.most_common()]
        total = sum(sorted_counts)
        table[label] = {
            "n_distinct_edit_signatures": len(sig_counts),
            "entropy_bits": entropy_bits(sorted_counts),
            "gini": gini(sorted_counts),
            "top_signature": sig_str(sig_counts.most_common(1)[0][0]) if sig_counts else "n/a",
        }
        topk[label] = topk_coverage_share(sorted_counts, total)
    return {"table": table, "topk_coverage": topk}


# --------------------------------------------------------------------------- #
# rescue_breakdown: baseline (identity) vs. rescued vs. unsolved
# --------------------------------------------------------------------------- #
@insight(
    "rescue_breakdown",
    "Baseline vs. rescued vs. unsolved",
    "stackedBar",
    "For each group: identity-already-correct (baseline), identity-wrong-"
    "but-search-found-a-fix (rescued), and unsolved.",
)
def rescue_breakdown(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table = {}
    for label, samples in groups.items():
        n = len(samples)
        baseline = sum(1 for s in samples if s.get("initial_transition_metric") == 1.0)
        rescued = sum(
            1
            for s in samples
            if s.get("initial_transition_metric") != 1.0 and s.get("final_valid_transitions")
        )
        unsolved = n - baseline - rescued
        table[label] = {
            "n": n,
            "baseline_solved": baseline,
            "rescued": rescued,
            "unsolved": unsolved,
            "baseline_rate": baseline / n if n else 0.0,
            "rescued_rate": rescued / n if n else 0.0,
            "solved_rate": (baseline + rescued) / n if n else 0.0,
        }
    return {"table": table}


# --------------------------------------------------------------------------- #
# op_class_coverage: "if we only had skip ops, how many Qs solvable at all"
# --------------------------------------------------------------------------- #
@insight(
    "op_class_coverage",
    "Op-class coverage (skip-only / repeat-only / both / identity)",
    "table",
    "For each question, which op-classes have >=1 valid program at all, "
    "existence, not instance share, with two denominators (of solved / of "
    "all questions in the group).",
)
def op_class_coverage(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table, mix_sv, mix_all = {}, {}, {}
    classes = ("identity", "skip-only", "repeat-only", "both")
    for label, samples in groups.items():
        counts = Counter()
        n_solved = 0
        n_all = len(samples)
        for sample in samples:
            valid = [tuple(p) for p in sample.get("final_valid_transitions", [])]
            if not valid:
                continue
            n_solved += 1
            for c in {op_class(p, num_layers) for p in valid}:
                counts[c] += 1
        table[label] = {
            "solved": {c: (counts.get(c, 0) / n_solved if n_solved else 0.0) for c in classes},
            "all": {c: (counts.get(c, 0) / n_all if n_all else 0.0) for c in classes},
            "n_solved": n_solved,
            "n_all": n_all,
        }

        # Class MIX (distribution): of every PROGRAM instance (not
        # existence-per-question), what fraction falls in each class --
        # shortest-valid vs ALL instances. A different question from the
        # coverage table above ("does >=1 program of this class exist for
        # this question")
        # -- this is "of everything MCTS tried/kept, how is it distributed".
        rep_paths = _representative_paths(samples, num_layers)
        all_paths = [tuple(p) for s in samples for p in s.get("final_valid_transitions", [])]
        sv_counts = Counter(op_class(p, num_layers) for p in rep_paths)
        all_counts = Counter(op_class(p, num_layers) for p in all_paths)
        sv_total = sum(sv_counts.values())
        all_total = sum(all_counts.values())
        mix_sv[label] = {c: (sv_counts.get(c, 0) / sv_total if sv_total else 0.0) for c in classes}
        mix_all[label] = {
            c: (all_counts.get(c, 0) / all_total if all_total else 0.0) for c in classes
        }
    return {"table": table, "mix_shortest_valid": mix_sv, "mix_all_instances": mix_all}


# --------------------------------------------------------------------------- #
# menu_size_distribution: per-question valid-program-count boxplot
# --------------------------------------------------------------------------- #
@insight(
    "menu_size_distribution",
    "Valid-program count per question",
    "boxplot",
    "Distribution (5-number summary) of how many valid programs each "
    "question has, PER SAMPLE (0 for an unsolved question, not dropped, "
    "counting solved questions only would inflate the apparent typical "
    "menu size).",
)
def menu_size_distribution(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table = {}
    for label, samples in groups.items():
        counts = np.array([len(s.get("final_valid_transitions", [])) for s in samples], dtype=float)
        if counts.size == 0:
            table[label] = None
            continue
        q1, med, q3 = np.percentile(counts, [25, 50, 75])
        table[label] = {
            "min": float(counts.min()),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "max": float(counts.max()),
            "mean": float(counts.mean()),
            "n": int(counts.size),
        }
    return {"boxplot": table}


# --------------------------------------------------------------------------- #
# layer_ops_heatmap: full per-layer skip/repeat coverage, shortest-valid vs
# ALL valid instances, plus mean segment length starting at each layer
# (shortest-valid): rows=group, cols=layer, color=% coverage.
# --------------------------------------------------------------------------- #
@insight(
    "layer_ops_heatmap",
    "Where do skip/repeat land, by layer? (full per-layer resolution)",
    "heatmap",
    "Per-layer % of programs with a SKIP / REPEAT touching that layer, two "
    "lenses (shortest-valid = one representative program per solved "
    "question; ALL valid instances = every valid path, instance-weighted), "
    "plus mean length of edit segments STARTING at each layer "
    "(shortest-valid).",
)
def layer_ops_heatmap(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table = {}
    for label, samples in groups.items():
        rep_paths = _representative_paths(samples, num_layers)
        all_paths = [tuple(p) for s in samples for p in s.get("final_valid_transitions", [])]
        n_rep, n_all = len(rep_paths), len(all_paths)

        def _rate(paths, which, n):
            freq = layer_freq(paths, which, num_layers)
            return [x / n if n else 0.0 for x in freq]

        len_at_start = [[] for _ in range(num_layers)]
        for p in rep_paths:
            prog = decode(p, num_layers)
            for seg in prog.segments:
                if seg.op.value != "keep":
                    len_at_start[seg.start].append(len(seg))
        mean_len = [(sum(v) / len(v)) if v else None for v in len_at_start]

        table[label] = {
            "n_shortest_valid": n_rep,
            "n_all_instances": n_all,
            "skip_shortest_valid": _rate(rep_paths, "skip", n_rep),
            "repeat_shortest_valid": _rate(rep_paths, "repeat", n_rep),
            "skip_all_instances": _rate(all_paths, "skip", n_all),
            "repeat_all_instances": _rate(all_paths, "repeat", n_all),
            "mean_len_starting_at_layer": mean_len,
        }
    return {"table": table}


# --------------------------------------------------------------------------- #
# shorter_than_identity: "1b": does the search find shorter-than-identity
# programs, and how often?
# --------------------------------------------------------------------------- #
@insight(
    "shorter_than_identity",
    "Shorter than identity, do we find shorter programs than the base model?",
    "hbar",
    "For shortest_valid per solved question: mean/median executed length "
    "vs. identity (num_layers), and the share of solved questions where "
    "shortest_valid is strictly shorter than identity.",
)
def shorter_than_identity(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    table = {}
    for label, samples in groups.items():
        lens = []
        for s in samples:
            valid = [tuple(p) for p in s.get("final_valid_transitions", [])]
            if not valid:
                continue
            lens.append(len(shortest_valid(valid, num_layers)))
        n_solved = len(lens)
        if n_solved == 0:
            table[label] = None
            continue
        arr = np.array(lens, dtype=float)
        shorter = sum(1 for L in lens if L < num_layers)
        table[label] = {
            "n_solved": n_solved,
            "mean_len": float(arr.mean()),
            "median_len": float(np.median(arr)),
            "identity_len": num_layers,
            "shorter_rate": shorter / n_solved,
        }
    return {"table": table}


# --------------------------------------------------------------------------- #
# dismemberment: length x start-layer heatmap, shortest-valid vs all-instances
# --------------------------------------------------------------------------- #
def _length_position_heatmap(paths_iter, which: str, num_layers: int):
    M = [[0] * num_layers for _ in range(4)]
    for path in paths_iter:
        prog = decode(tuple(path), num_layers)
        for seg in prog.segments:
            op = seg.op.value
            match = (op == which) or (which == "edit" and op in ("skip", "repeat"))
            if not match:
                continue
            length = len(seg)
            if 1 <= length <= 4:
                M[length - 1][seg.start] += 1
    return M


@insight(
    "dismemberment",
    "Segment length x start-layer heatmap (skip/repeat/edit)",
    "heatmap",
    "[length 1-4] x [start layer] RAW COUNT matrices for SKIP, REPEAT, and "
    "EDIT (=SKIP union REPEAT, i.e. skip+repeat since a segment is always "
    "exactly one of the two), in two lenses: shortest-valid (one "
    "representative program per solved question, skip-heavy by "
    "construction) and all-instances (every valid path, repeat-heavy, the "
    "raw search landscape). %-of-total display is a render-time transform "
    "of these same raw counts, not computed here.",
)
def dismemberment(groups: Dict[str, List[dict]], num_layers: int) -> dict:
    out = {}
    for label, samples in groups.items():
        rep_paths = _representative_paths(samples, num_layers)
        all_paths = [tuple(p) for s in samples for p in s.get("final_valid_transitions", [])]
        out[label] = {
            "shortest_valid": {
                "skip": _length_position_heatmap(rep_paths, "skip", num_layers),
                "repeat": _length_position_heatmap(rep_paths, "repeat", num_layers),
                "edit": _length_position_heatmap(rep_paths, "edit", num_layers),
            },
            "all_instances": {
                "skip": _length_position_heatmap(all_paths, "skip", num_layers),
                "repeat": _length_position_heatmap(all_paths, "repeat", num_layers),
                "edit": _length_position_heatmap(all_paths, "edit", num_layers),
            },
        }
    return out
