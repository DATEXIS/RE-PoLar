"""Menu coverage via real cross-execution: for a model/difficulty, rank
non-identity programs by how many questions they solved DURING the original
MCTS search, then actually EXECUTE the top-K of those programs against every
question in the full train+val+test pool (not just the ones the search
happened to already try each program on). This is what produces the paper's
Finding 5 menu-coverage numbers (coverage vs. menu size K): ranking alone
(without real execution) understates coverage, since the original search
never tried every top-K program against every question -- only real
execution gives the true coverage curve.

Free-reuse rule: any (program, question) pair the original MCTS search
already recorded as valid/invalid is reused as-is, never re-executed; only
genuinely untried pairs are run fresh. Uses `GenerationReward.__call__`
(serial, not masked-batch) so every reused/generated label stays directly
comparable to the ones already on record from the original search.

Output is a full, appendable per-program x per-question grid, compact-
encoded: the qid list is written ONCE per difficulty, and each program's
results are a same-length list of {0,1} rewards + a parallel provenance
string (one char per qid: 'v'=known valid from the original MCTS search,
'i'=known invalid from the original MCTS search, 'f'=freshly executed this
run) aligned positionally against that qid list -- no repeated qid keys.
This lets a later pass (a) exclude identity, (b) add more programs, or (c)
add more questions, as pure post-processing / targeted appends, without
re-deriving or re-running anything already stored.

  python -m analysis.mcts_design.menu_coverage \
      --model qwen25_3b --difficulty 1 --k 100 --batch-size 128 \
      --data-dir ./data/dart_math --mcts-dir ./data/mcts/qwen25_3b \
      --output ./results/menu_crossexec/qwen25_3b/diff1.json

Top level is stdlib-only so nothing here needs torch until `main()` runs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(p):
    return json.loads(Path(p).read_text())


def resolve_difficulties(vals):
    if not vals or "all" in vals:
        return [1, 2, 3, 4, 5]
    return sorted(int(v) for v in vals)


def build_candidate_pool(mcts_samples, pool_qids, k):
    """Rank non-identity programs by how many pool questions they were
    recorded valid for during the original search ("most found" first), and
    return the top-k along with the full set of (qid -> known reward) pairs
    already on record for each -- reused for free, never re-generated."""
    valid_qids = defaultdict(set)
    known = defaultdict(dict)  # path tuple -> {qid: reward}
    for sample in mcts_samples:
        qid = sample.get("sample_info", {}).get("query_id")
        if qid not in pool_qids:
            continue
        for p in sample.get("final_valid_transitions", []):
            t = tuple(p)
            valid_qids[t].add(qid)
            known[t][qid] = 1.0
        for p in sample.get("final_invalid_transitions", []):
            t = tuple(p)
            known[t].setdefault(qid, 0.0)  # a program can't be both; valid wins if both logged
    ranked = sorted(valid_qids.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:k]
    return [(path, known[path]) for path, _ in ranked]


def main(argv=None):
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.router.train import program_from_layer_path
    from re_polar.datasets.schemas import load_samples

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument("--difficulty", action="append", default=None)
    p.add_argument("--data-dir", required=True, help="DART-Math root (has diff{N}/{train,val,test}.json)")
    p.add_argument("--mcts-dir", required=True,
                   help="MCTS output root (has diff{N}/merged_mcts_samples.json)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="cap n_pool questions, smoke-test only")
    p.add_argument("--fresh", action="store_true",
                   help="ignore already-known (program, question) labels from --mcts-dir and "
                        "generate every cell fresh instead of reusing them.")
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    diffs = resolve_difficulties(args.difficulty)

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    reward = GenerationReward(executor, batch_size=args.batch_size)  # prompt_style="raw" default

    results = []
    for diff in diffs:
        train_data = load_json(Path(args.data_dir) / f"diff{diff}" / "train.json")
        val_data = load_json(Path(args.data_dir) / f"diff{diff}" / "val.json")
        test_data = load_json(Path(args.data_dir) / f"diff{diff}" / "test.json")
        pool_data = train_data + val_data + test_data
        if args.limit:
            pool_data = pool_data[: args.limit]
        pool_qids = [d["query_id"] for d in pool_data]
        pool_qid_set = set(pool_qids)
        assert len(pool_qids) == len(pool_qid_set), f"diff{diff}: duplicate query_id across train/val/test"
        gt_by_qid = {d["query_id"]: d["gt_ans"] for d in pool_data}
        q_by_qid = {d["query_id"]: d["question"] for d in pool_data}
        n_pool = len(pool_qids)
        print(f"[diff {diff}] {n_pool} pool questions (train={len(train_data)}, "
              f"val={len(val_data)}, test={len(test_data)})", flush=True)

        mcts_samples = load_samples(
            Path(args.mcts_dir) / cfg["model_id"] / f"dart-math-diff-{diff}" / "merged_mcts_samples.json")
        candidates = build_candidate_pool(mcts_samples, pool_qid_set, args.k)
        print(f"[diff {diff}] top-{len(candidates)} candidate programs selected "
              f"(ranked over the full {n_pool}-question pool)", flush=True)

        n_reused_total = 0
        n_new_total = 0
        per_program_rows = []  # {"path": [...], "rewards": [...], "provenance": "..."}
        for path, known in candidates:
            row = {} if args.fresh else dict(known)
            provenance = {}  # qid -> 'v'/'i' for known cells, filled in below
            if not args.fresh:
                for qid, r in known.items():
                    provenance[qid] = "v" if r > 0 else "i"
            todo_qids = [q for q in pool_qids if q not in row]
            n_reused_total += n_pool - len(todo_qids)
            n_new_total += len(todo_qids)
            if todo_qids:
                # strict_repeat_2x=False: analyzing real MCTS menu paths, not
                # generating router training labels -- want the true `times`.
                program = program_from_layer_path(list(path), D, strict_repeat_2x=False)
                qs = [q_by_qid[q] for q in todo_qids]
                gts = [gt_by_qid[q] for q in todo_qids]
                rewards = reward(program, qs, gts)
                for q, r in zip(todo_qids, rewards):
                    row[q] = r
                    provenance[q] = "f"
            rewards_list = [int(row.get(q, 0.0) > 0) for q in pool_qids]
            provenance_str = "".join(provenance.get(q, "?") for q in pool_qids)
            per_program_rows.append({"path": list(path), "rewards": rewards_list, "provenance": provenance_str})
            solved_now = sum(rewards_list)
            print(f"  program {list(path)}: reused {n_pool - len(todo_qids)}, "
                  f"new {len(todo_qids)}, total solved {solved_now}/{n_pool}", flush=True)

        # real top-k coverage curve: in the candidates' rank order (by search-frequency
        # over the full pool), how much of the pool does ANY of the top-1..top-K solve,
        # using the REAL (executed) labels.
        any_solved = [0] * n_pool
        coverage_curve = []
        for i, row in enumerate(per_program_rows, start=1):
            for j, r in enumerate(row["rewards"]):
                if r:
                    any_solved[j] = 1
            if i in (1, 3, 5, 10, 20, 30, 50, 100) or i == len(per_program_rows):
                coverage_curve.append({"k": i, "real_coverage": sum(any_solved) / n_pool})

        row_out = {
            "difficulty": diff,
            "n_pool": n_pool, "n_train": len(train_data), "n_val": len(val_data), "n_test": len(test_data),
            "k_requested": args.k, "k_actual": len(candidates),
            "n_reused_cells": n_reused_total, "n_new_cells": n_new_total,
            "real_topk_coverage_curve": coverage_curve,
            "final_real_coverage": coverage_curve[-1]["real_coverage"] if coverage_curve else 0.0,
            "qids": pool_qids,  # written ONCE per difficulty; per_program rows are positional against this
            "provenance_legend": {"v": "known valid from original MCTS search (reused, not re-executed)",
                                   "i": "known invalid from original MCTS search (reused, not re-executed)",
                                   "f": "freshly executed this run"},
            "per_program": per_program_rows,
        }
        results.append(row_out)
        print(f"[diff {diff}] DONE. real top-{len(candidates)} coverage = "
              f"{row_out['final_real_coverage']:.4f} "
              f"({sum(any_solved)}/{n_pool} pool questions), "
              f"reused {n_reused_total} cells, ran {n_new_total} new evaluations", flush=True)

    out = {"model": args.model, "k": args.k, "schema": "menu_coverage", "per_difficulty": results}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out))
    print(f"\nwrote {outp}", flush=True)

    print(f"\n{'diff':>4} {'k':>4} {'real_cov':>9} {'new_evals':>10}")
    for r in results:
        print(f"{r['difficulty']:>4} {r['k_actual']:>4} {r['final_real_coverage']:>9.4f} "
              f"{r['n_new_cells']:>10}")
    return outp


if __name__ == "__main__":
    main()
