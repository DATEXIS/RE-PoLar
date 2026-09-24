"""How load-bearing is a single edit inside a working (RESCUE) program?

For every RESCUE program (identity wrong, this program right), revert one
edited segment at a time back to KEEP (identity behavior at that segment,
holding every other segment fixed) and check whether the program still
solves the question. This is DIRECTIONAL and restricted to programs that
actually work -- of a RESCUE program's own edits, how many are load-bearing?

**No new model inference**: MCTS's own tree search already visits many
single-segment-reverted siblings of the programs it finds, as ordinary UCB
exploration -- this re-reads already-computed rewards from the raw search
log, never re-executes anything. Coverage (how many of a RESCUE program's
edited segments have a logged keep-reverted sibling) is empirical, reported
explicitly, not assumed complete.

    python -m analysis.structural_analysis.reversion_robustness \\
        --search-log /path/to/qwen3_8b_diff2.jsonl \\
        --num-layers 36 \\
        --output results/structural_analysis/dart_reversion_qwen3_8b_diff2.json

``--search-log`` accepts EITHER of two real, equivalent formats:
  1. The raw per-simulation MCTS cache log ({query_id, path, reward} per
     line) written during search -- an internal artifact, not part of the
     released dataset.
  2. The RELEASED merged_mcts_samples.json itself (see data/DATA.md), read
     from each record's own `search_trajectory` field (`{path, parent_path,
     reward}`, tagged by that record's own `sample_info.query_id`), PLUS a
     synthetic identity-path record built from that same record's
     `initial_transition_metric` (documented in data/DATA.md as "reward of
     the identity program") -- `search_trajectory` does not always log an
     explicit identity-path visit, so this is the correct, documented source
     for it. Same underlying simulation data as format 1, just nested
     per-question instead of flat. This is what lets this check run from
     ONLY the data re_polar actually publishes, with no separate/undocumented
     cache file needed.
Auto-detected: format 2 if the file is a single JSON document with a
top-level "samples" key, format 1 otherwise (line-delimited JSON).
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

FullBoundary = Tuple[Tuple[int, int], ...]  # ((start,end), ...) for ALL segments, KEEP included
OpSeq = Tuple[str, ...]  # op name per segment position, aligned to FullBoundary


def is_identity_path(path: Sequence[int]) -> bool:
    return list(path) == list(range(len(path)))


def parse_program(path: Sequence[int], num_layers: int):
    """path -> Program, via the proven, re-execution-asserted parser. Lazy
    torch-touching import, kept local so this module stays torch-free at
    import time."""
    from re_polar.router.train import program_from_layer_path

    return program_from_layer_path(list(path), num_layers, strict_repeat_2x=False)


def full_boundary(program) -> FullBoundary:
    return tuple((s.start, s.end) for s in program.segments)


def op_seq(program) -> OpSeq:
    return tuple(s.op.value for s in program.segments)


def classify_reversions(
    records: List[dict], num_layers: int, max_group_size: int = 40
) -> Dict[str, object]:
    """For one query_id's logged (path, reward) records: find its identity
    reward, then for every single-segment keep-reversion pair where the
    non-keep side is a genuine RESCUE program (identity reward 0, this
    program's reward 1), check whether the keep-reverted sibling still has
    reward 1 (rescue survives) or 0 (this edit was load-bearing)."""
    identity_reward = None
    for r in records:
        if is_identity_path(r["path"]):
            identity_reward = r["reward"]
            break
    result = {"eligible": False, "reversions": [], "skipped_groups": 0}
    if identity_reward != 0.0:
        return result  # not a RESCUE-eligible query (identity already correct, or no identity record logged)
    result["eligible"] = True

    parsed = []
    for r in records:
        try:
            program = parse_program(r["path"], num_layers)
        except ValueError:
            continue
        parsed.append((r, program))

    by_full_boundary: Dict[tuple, List[Tuple[dict, tuple]]] = defaultdict(list)
    for r, program in parsed:
        fb = full_boundary(program)
        by_full_boundary[fb].append((r, op_seq(program)))

    for _fb, rows in by_full_boundary.items():
        if len(rows) < 2:
            continue
        if len(rows) > max_group_size:
            result["skipped_groups"] += 1
            continue
        for (a, ops_a), (b, ops_b) in itertools.combinations(rows, 2):
            diff_positions = [i for i, (oa, ob) in enumerate(zip(ops_a, ops_b)) if oa != ob]
            if len(diff_positions) != 1:
                continue
            i = diff_positions[0]
            op_a, op_b = ops_a[i], ops_b[i]
            if op_a == "keep" and op_b != "keep":
                keep_side, edit_side = a, b
            elif op_b == "keep" and op_a != "keep":
                keep_side, edit_side = b, a
            else:
                continue  # neither side is keep (e.g. repeat<->skip) -- not a "reversion"
            if edit_side["reward"] != 1.0:
                continue  # not a confirmed-working RESCUE program, nothing to break
            broke = keep_side["reward"] == 0.0
            result["reversions"].append(
                {
                    "broke": broke,
                    "edit_side_path": tuple(edit_side["path"]),
                }
            )

    return result


def aggregate(all_query_results: List[dict]) -> dict:
    reversions = [rev for qres in all_query_results for rev in qres["reversions"]]
    n = len(reversions)
    n_broken = sum(1 for rev in reversions if rev["broke"])
    n_distinct_programs = len(
        {
            (qidx, rev["edit_side_path"])
            for qidx, qres in enumerate(all_query_results)
            for rev in qres["reversions"]
        }
    )
    return {
        "n_queries_eligible": sum(1 for qres in all_query_results if qres["eligible"]),
        "n_reversions_tested": n,
        "n_broken": n_broken,
        "break_rate": (n_broken / n) if n else None,
        "n_distinct_rescue_programs_covered": n_distinct_programs,
    }


def load_by_query(path: str, num_layers: int = None) -> Dict[str, List[dict]]:
    """Reads --search-log in either supported format (see module docstring)
    and returns {query_id: [{"path", "reward"}, ...]}. Peeks at the first
    line only to decide which format it is (cheap): a line-delimited raw
    cache record parses to a dict with a top-level "query_id" key; a
    merged_mcts_samples.json is written as ONE line (json.dump, no indent,
    see re_polar/datasets/schemas.py) whose top-level key is "samples" instead
    -- the two are never ambiguous.

    ``num_layers`` is required for the merged format: `search_trajectory`
    (see re_polar/datasets/schemas.py) does NOT always contain an explicit
    identity-path record (identity only shows up as some entries' parent_path,
    never necessarily its own path) -- the identity program's reward is
    instead carried on the sample itself, as `initial_transition_metric`
    (documented in data/DATA.md). A synthetic identity record is added from
    that field for every sample, using the same identity-path definition
    `is_identity_path` uses (`list(range(num_layers))`), so classify_reversions
    sees the same identity-eligibility signal it would from a raw cache log.
    """
    by_query: Dict[str, List[dict]] = defaultdict(list)
    n_lines = 0
    with open(path) as f:
        first_line = f.readline()
    first = json.loads(first_line)
    is_raw_cache = isinstance(first, dict) and "query_id" in first

    if is_raw_cache:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_query[r["query_id"]].append(r)
                n_lines += 1
                if n_lines % 200000 == 0:
                    print(
                        f"read {n_lines} lines, {len(by_query)} distinct query_ids so far",
                        flush=True,
                    )
        print(
            f"Loaded {n_lines} records across {len(by_query)} distinct query_ids (raw cache log)",
            flush=True,
        )
        return by_query

    if num_layers is None:
        raise ValueError(
            "--num-layers is required to read the released merged_mcts_samples.json format "
            "(needed to construct the identity path for initial_transition_metric)"
        )
    identity_path = list(range(num_layers))

    with open(path) as f:
        data = json.load(f)
    n_no_traj = 0
    for sample in data["samples"]:
        qid = sample["sample_info"]["query_id"]
        by_query[qid].append({"path": identity_path, "reward": sample["initial_transition_metric"]})
        n_lines += 1
        traj = sample.get("search_trajectory")
        if not traj:
            n_no_traj += 1
            continue
        for r in traj:
            by_query[qid].append({"path": r["path"], "reward": r["reward"]})
            n_lines += 1
    if n_no_traj:
        print(
            f"{n_no_traj}/{len(data['samples'])} samples had no search_trajectory "
            f"(identity-only coverage for those, from initial_transition_metric)",
            flush=True,
        )
    print(
        f"Loaded {n_lines} records across {len(by_query)} distinct query_ids "
        f"(released merged_mcts_samples.json)",
        flush=True,
    )
    return by_query


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--search-log", required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument(
        "--max-queries", type=int, default=None, help="debug: first N distinct query_ids"
    )
    parser.add_argument("--max-group-size", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    by_query = load_by_query(args.search_log, num_layers=args.num_layers)

    query_ids = list(by_query)
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]

    all_query_results = []
    total_skipped_groups = 0
    for i, qid in enumerate(query_ids):
        qres = classify_reversions(by_query[qid], args.num_layers, args.max_group_size)
        all_query_results.append(qres)
        total_skipped_groups += qres["skipped_groups"]
        if (i + 1) % 500 == 0:
            n_so_far = sum(len(r["reversions"]) for r in all_query_results)
            print(
                f"processed {i + 1}/{len(query_ids)} queries -- {n_so_far} reversion pairs so far",
                flush=True,
            )

    agg = aggregate(all_query_results)
    result = {
        "n_queries": len(query_ids),
        "skipped_groups_over_cap": total_skipped_groups,
        **agg,
    }

    print(
        f"\n=== DONE: {len(query_ids)} queries, {total_skipped_groups} groups skipped "
        f"(over --max-group-size={args.max_group_size}) ===",
        flush=True,
    )
    print(f"queries with identity wrong (RESCUE-eligible): {agg['n_queries_eligible']}", flush=True)
    print(f"single-segment reversions tested: {agg['n_reversions_tested']}", flush=True)
    print(
        f"distinct RESCUE programs covered: {agg['n_distinct_rescue_programs_covered']}", flush=True
    )
    if agg["break_rate"] is not None:
        print(
            f"break_rate: {agg['break_rate']*100:.1f}% ({agg['n_broken']}/{agg['n_reversions_tested']})",
            flush=True,
        )
    else:
        print("no reversion pairs found in this log -- 0 coverage", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
