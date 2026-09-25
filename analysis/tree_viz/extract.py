#!/usr/bin/env python3
"""Turn this repo's own MCTS search results into the tree-data JSON that
template.html renders (via build.py, or directly with --html). Reads
data/mcts_results_full/ and data/generated_answers/ (see data/DATA.md).

Which (diff, sample-index) trees to include:
    --pick DIFF:IDX[:NOTE]     repeatable, e.g. --pick 1:1183:"deep, rescued"
    --picks-file picks.csv     CSV rows: diff,idx[,note]  (# comments ok)
    --all-diff DIFF            every sample in that tier (repeatable;
                                combine with --limit-per-diff for a quick look)

Examples:
    python analysis/tree_viz/extract.py --model-tag qwen3_8b \\
        --all-diff 3 --limit-per-diff 20 --html out.html

    python analysis/tree_viz/extract.py --model-tag qwen3_8b \\
        --pick 3:0:"first sample" --output data.json

Requires this repo's venv active (`re_polar` importable) — see AGENTS.md.
"""
from __future__ import annotations

import argparse
import csv
import functools
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from re_polar.core import MAX_SEGMENT_LEN, Op, Program
from re_polar.models import MODEL_REGISTRY
from re_polar.router.train import program_from_layer_path


def program_edits(program: Program) -> list[tuple[int, int, Op, int]]:
    """(start, length, op, times) for every non-KEEP segment — the shape this
    tree viewer needs. re_polar.mcts.analysis.segments.program_edits returns
    something different (raw skipped/repeated layer index *sets*, no run
    structure), so this reads Program.segments directly instead. Op is a
    (str, Enum) subclass, so it supports both `.value` and `== "skip"`."""
    return [
        (seg.start, len(seg), seg.op, seg.times) for seg in program.segments if seg.op != Op.KEEP
    ]


# ---------------------------------------------------------------- picks ----


def parse_pick(s: str) -> tuple[int, int, str]:
    parts = s.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--pick must be 'DIFF:IDX' or 'DIFF:IDX:NOTE', got {s!r}")
    diff, idx = int(parts[0]), int(parts[1])
    note = parts[2] if len(parts) == 3 else f"diff{diff} #{idx}"
    return diff, idx, note


def load_picks_file(path: str) -> list[tuple[int, int, str]]:
    picks = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            diff, idx = int(row[0]), int(row[1])
            note = row[2].strip() if len(row) > 2 and row[2].strip() else f"diff{diff} #{idx}"
            picks.append((diff, idx, note))
    return picks


# -------------------------------------------------------------- loading ----


@functools.lru_cache(maxsize=None)
def load_samples(data_root: str, model_tag: str, diff: int) -> tuple:
    # mcts_results_full/, not mcts_results/ -- the plain data/mcts_results/ dump
    # doesn't carry question/gt_ans (see data/DATA.md); mcts_results_full/ is
    # the query_id-joined version `python -m re_polar.datasets.attach_mcts_questions`
    # produces, uncompressed (unlike mcts_results/'s .json.gz).
    path = (
        Path(data_root)
        / "mcts_results_full"
        / model_tag
        / f"dart-math-diff-{diff}"
        / "merged_mcts_samples.json"
    )
    return tuple(json.load(open(path))["samples"])


@functools.lru_cache(maxsize=None)
def load_generation_index(data_root: str, model_tag: str, diff: int) -> dict:
    """{(query_id, tuple(path)): {"text":..., "reward":...}}, built in one pass
    over data/generated_answers/<model>/diff<D>.jsonl.gz (see data/DATA.md) --
    "a sample of evaluations", not necessarily every path this tier's search
    visited, so a miss here just means no generated text for that node."""
    path = Path(data_root) / "generated_answers" / model_tag / f"diff{diff}.jsonl.gz"
    index = {}
    if not path.exists():
        return index
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            key = (d["query_id"], tuple(d["path"]))
            if key not in index:
                index[key] = {"text": d["text"], "reward": d["reward"]}
    return index


def all_picks_for_diff(
    data_root: str, model_tag: str, diff: int, limit: int | None
) -> list[tuple[int, int, str]]:
    n = len(load_samples(data_root, model_tag, diff))
    if limit:
        n = min(n, limit)
    return [(diff, i, f"diff{diff} #{i}") for i in range(n)]


# --------------------------------------------------------- edit-set math ----
# Reconstructs the parent/child edit-set relationships from the flat list of
# layer-paths the search actually visited. A canonical edit list can hold two
# (or more) adjacent same-op, same-times edits that only exist because a
# single content run longer than MAX_SEGMENT_LEN had to be built from
# multiple <=MAX_SEGMENT_LEN actions -- merge_super_runs/candidate_parents
# cast a wide enough net over alternate splits to find the real parent
# instead of assuming the parser's own split is the only valid one.


def edits_key(edits):
    return tuple((s, l, op.value, t) for s, l, op, t in edits)


def fmt_edit(s, l, op, t):
    span = f"{s}" if l == 1 else f"{s}–{s + l - 1}"
    if op == "skip":
        return f"skip {span}"
    return f"repeat {span} ×{t}"


def fmt_edits(edits):
    if not edits:
        return "identity (0 edits)"
    return "  ·  ".join(fmt_edit(*e) for e in edits)


def merge_super_runs(edits):
    edits_sorted = sorted(edits)
    runs, i, n = [], 0, len(edits_sorted)
    while i < n:
        s, l, op, t = edits_sorted[i]
        piece_edits = [edits_sorted[i]]
        end = s + l
        j = i + 1
        while j < n:
            s2, l2, op2, t2 = edits_sorted[j]
            if op2 == op and t2 == t and s2 == end:
                piece_edits.append(edits_sorted[j])
                end = s2 + l2
                j += 1
            else:
                break
        runs.append(dict(pieces=piece_edits, start=s, length=end - s, op=op, times=t))
        i = j
    return runs


def split_options(start, length, op, times, max_segment_len):
    if length <= max_segment_len:
        return []
    lo, hi = max(1, length - max_segment_len), min(length - 1, max_segment_len)
    return [((start, k, op, times), (start + k, length - k, op, times)) for k in range(lo, hi + 1)]


def candidate_parents(edits, max_segment_len):
    edits = tuple(edits)
    N = len(edits)
    candidates = set()
    for i in range(N):
        candidates.add(tuple(sorted(edits[:i] + edits[i + 1 :])))
    for run in merge_super_runs(edits):
        if run["length"] <= max_segment_len:
            continue
        run_pieces = set(run["pieces"])
        others = tuple(e for e in edits if e not in run_pieces)
        for pieceA, pieceB in split_options(
            run["start"], run["length"], run["op"], run["times"], max_segment_len
        ):
            candidates.add(tuple(sorted(others + (pieceA,))))
            candidates.add(tuple(sorted(others + (pieceB,))))
    return candidates


# ------------------------------------------------------------- per-tree ----


def build_tree(data_root, diff, idx, note, depth, text_cap, model_tag, strict_repeat_2x):
    s = load_samples(data_root, model_tag, diff)[idx]
    valid = s["final_valid_transitions"]
    invalid = s["final_invalid_transitions"]
    query_id = s["sample_info"]["query_id"]
    gen_index = load_generation_index(data_root, model_tag, diff)

    nodes = {}
    dupe_hits = 0
    for kind, plist in (("valid", valid), ("invalid", invalid)):
        for path in plist:
            program = program_from_layer_path(path, depth, strict_repeat_2x=strict_repeat_2x)
            edits = program_edits(program)
            k = edits_key(edits)
            if k in nodes:
                dupe_hits += 1
                continue
            hit = gen_index.get((query_id, tuple(path)))
            nodes[k] = dict(
                edits=list(k),
                edit_count=len(k),
                is_valid=(kind == "valid"),
                text=(hit["text"][:text_cap] if hit else None),
                reward=(hit["reward"] if hit else None),
            )
    if () not in nodes:
        hit = gen_index.get((query_id, tuple(range(depth))))
        nodes[()] = dict(
            edits=[],
            edit_count=0,
            is_valid=(s["initial_transition_metric"] == 1),
            text=(hit["text"][:text_cap] if hit else None),
            reward=(hit["reward"] if hit else None),
        )

    # Search trajectory (re_polar/mcts/search.py's ProgramMCTS.trajectory):
    # {path, parent_path, reward} per simulation, in visit order. Parsed ONCE
    # here, used for two things below: (1) real_parent_of -- parent_path is
    # chain[-2] in search.py's update(), i.e. the ACTUAL internal MCTS-tree
    # parent, not a guess -- preferred over candidate_parents' post-hoc
    # inference wherever it resolves to a saved node; (2) the trajectory
    # replay list further down.
    raw_traj = s.get("search_trajectory") or []
    traj_parsed = []  # (child_key_or_None, parent_key_or_None, reward), in visit order
    real_parent_of = {}  # child edits_key -> parent edits_key, ground truth, first occurrence wins
    for entry in raw_traj:
        path = entry.get("path")
        child_key = None
        if path:
            try:
                child_key = edits_key(
                    program_edits(
                        program_from_layer_path(path, depth, strict_repeat_2x=strict_repeat_2x)
                    )
                )
            except Exception:
                child_key = None
        parent_path = entry.get("parent_path")
        if parent_path is None:
            parent_key = () if child_key is not None else None
        else:
            try:
                parent_key = edits_key(
                    program_edits(
                        program_from_layer_path(
                            parent_path, depth, strict_repeat_2x=strict_repeat_2x
                        )
                    )
                )
            except Exception:
                parent_key = None
        traj_parsed.append((child_key, parent_key, entry.get("reward")))
        if child_key is not None and parent_key is not None and child_key not in real_parent_of:
            real_parent_of[child_key] = parent_key

    by_count = defaultdict(list)
    for k in nodes:
        by_count[len(k)].append(k)

    parent_of = {}
    parent_ambiguous = {}
    parent_from_trajectory = {}
    for N in sorted(c for c in by_count if c > 0):
        for k in by_count[N]:
            real_parent = real_parent_of.get(k)
            if real_parent is not None and real_parent in nodes:
                # Ground truth from the actual search, not a "drop one edit" guess --
                # e.g. progressive widening can GROW an existing skip/repeat run by one
                # layer at a boundary (child edit (7,4,skip,1) from parent (8,3,skip,1)),
                # which candidate_parents' subset/re-split search below never considers
                # and used to show as a dashed "no ancestor found" fallback edge.
                parent_of[k] = real_parent
                parent_ambiguous[k] = False
                parent_from_trajectory[k] = True
                continue
            edits = nodes[k]["edits"]
            matches = sorted(c for c in candidate_parents(edits, MAX_SEGMENT_LEN) if c in nodes)
            simple_drop = tuple(edits[:-1]) if edits else ()
            found = simple_drop if simple_drop in matches else (matches[0] if matches else None)
            parent_of[k] = found if found is not None else ()
            parent_ambiguous[k] = len(matches) > 1
            parent_from_trajectory[k] = False

    id_of = {k: i for i, k in enumerate(nodes)}
    children = defaultdict(list)
    for k in nodes:
        if k == ():
            continue
        children[id_of[parent_of[k]]].append(id_of[k])

    node_records = {}
    for k, rec in nodes.items():
        i = id_of[k]
        node_records[i] = dict(
            id=i,
            edit_count=rec["edit_count"],
            edits=rec["edits"],
            edits_label=fmt_edits(rec["edits"]),
            is_valid=rec["is_valid"],
            reward=rec["reward"],
            text=rec["text"],
            children=sorted(children.get(i, [])),
            parent_fallback=(
                k != ()
                and not parent_from_trajectory.get(k, False)
                and parent_of[k] == ()
                and rec["edit_count"] > 1
            ),
            parent_ambiguous=parent_ambiguous.get(k, False),
            # a trajectory-resolved parent with the SAME edit_count as its child (~0.3%
            # of edges): progressive widening GREW one of the parent's own edits by a
            # layer at its boundary
            # (e.g. (8,3,skip,1) -> (7,4,skip,1)) rather than adding a wholly new edit
            # -- so depth (edit count) does NOT always advance by exactly one from
            # parent to child, in contradiction of this viewer's original one-edit-
            # per-row assumption (see the header disclosure). Real edge, correctly
            # resolved; just doesn't fit the row-per-edit-count layout, so it's
            # flagged for a distinct style.
            parent_same_level=(
                k != ()
                and parent_from_trajectory.get(k, False)
                and nodes[parent_of[k]]["edit_count"] == rec["edit_count"]
            ),
        )

    n_traj_parents = sum(parent_from_trajectory.values())
    if n_traj_parents:
        print(
            f"  {n_traj_parents}/{len(nodes) - 1} parents resolved exactly via search_trajectory "
            f"(vs. candidate_parents guessing)",
            file=sys.stderr,
        )

    # trajectory replay list for the viewer: maps each visited path onto the SAME node
    # ids as the static tree above (an unmatched step -- e.g. a degenerate all-skip
    # retry logged as path: [] -- is kept with node_id=None so the round count/scrubber
    # still line up; the viewer just skips revealing it).
    trajectory = None
    if raw_traj:
        trajectory = []
        unmatched = 0
        for child_key, _parent_key, reward in traj_parsed:
            node_id = id_of.get(child_key) if child_key is not None else None
            if node_id is None:
                unmatched += 1
            trajectory.append(dict(node_id=node_id, reward=reward))
        if unmatched:
            print(
                f"  trajectory: {unmatched}/{len(raw_traj)} steps didn't match a saved node",
                file=sys.stderr,
            )

    max_edit_count = max((r["edit_count"] for r in node_records.values()), default=0)
    counts_by_level = {lvl: len(by_count[lvl]) for lvl in sorted(by_count)}
    info = s["sample_info"]
    tree = dict(
        label=f"diff{diff} #{idx}",
        note=note,
        diff=diff,
        idx=idx,
        query_id=query_id,
        question=s.get("question"),
        gt_ans=s.get("gt_ans"),
        math_level=info.get("math_level"),
        domain=info.get("domain"),
        n_valid=len(valid),
        n_invalid=len(invalid),
        root_id=id_of[()],
        max_edit_count=max_edit_count,
        counts_by_level=counts_by_level,
        nodes=node_records,
        trajectory=trajectory,
    )
    print(
        f"diff{diff} #{idx}: {len(nodes)} distinct programs ({dupe_hits} dupes folded), "
        f"levels={counts_by_level}"
        + (f", trajectory={len(trajectory)} steps" if trajectory else ""),
        file=sys.stderr,
    )
    return tree


# ------------------------------------------------------------------ main ----


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-root",
        default="data",
        help="dir containing mcts_results_full/ and generated_answers/ (default: data)",
    )
    ap.add_argument(
        "--model-tag", default="qwen3_8b", choices=sorted(MODEL_REGISTRY), help="default: qwen3_8b"
    )
    ap.add_argument(
        "--depth",
        type=int,
        help="number of model layers (default: MODEL_REGISTRY[model-tag]['num_layers'])",
    )
    ap.add_argument(
        "--text-cap", type=int, default=550, help="max chars of generated text kept per node"
    )
    ap.add_argument(
        "--pick", action="append", default=[], type=parse_pick, metavar="DIFF:IDX[:NOTE]"
    )
    ap.add_argument("--picks-file", help="CSV file: diff,idx[,note] per row")
    ap.add_argument("--all-diff", action="append", default=[], type=int, metavar="DIFF")
    ap.add_argument("--limit-per-diff", type=int, help="cap on --all-diff (default: no cap)")
    ap.add_argument("--output", help="write tree-data JSON here")
    ap.add_argument("--html", help="also (or instead) write a standalone HTML file here")
    ap.add_argument("--template", default=str(Path(__file__).parent / "template.html"))
    ap.add_argument(
        "--strict-repeat-2x",
        action="store_true",
        help="reject REPEAT segments with times!=2 (PoLar's own parser, matches the "
        "router-training default). Default off: the tree should show what MCTS "
        "actually explored, not a filtered subset.",
    )
    args = ap.parse_args()
    depth = args.depth or MODEL_REGISTRY[args.model_tag]["num_layers"]

    picks = list(args.pick)
    if args.picks_file:
        picks += load_picks_file(args.picks_file)
    for diff in args.all_diff:
        picks += all_picks_for_diff(args.data_root, args.model_tag, diff, args.limit_per_diff)

    if not picks:
        ap.error("no trees selected — use --pick, --picks-file, and/or --all-diff")
    if not args.output and not args.html:
        ap.error("nothing to do — pass --output and/or --html")

    trees = [
        build_tree(
            args.data_root,
            diff,
            idx,
            note,
            depth,
            args.text_cap,
            args.model_tag,
            args.strict_repeat_2x,
        )
        for diff, idx, note in picks
    ]
    data_json = json.dumps(dict(base_depth=depth, trees=trees))

    if args.output:
        Path(args.output).write_text(data_json)
        print(f"wrote {args.output} ({len(data_json) / 1024:.1f} KB)", file=sys.stderr)

    if args.html:
        sys.path.insert(0, str(Path(__file__).parent))
        import build  # local build.py

        html = build.build(data_json, Path(args.template))
        Path(args.html).write_text(html)
        print(f"wrote {args.html} ({len(html) / 1024:.1f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
