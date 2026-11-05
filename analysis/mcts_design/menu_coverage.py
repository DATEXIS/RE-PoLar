"""Menu coverage (paper Finding 5): rank programs, identity included, by how many questions
they solved during the MCTS search, execute the top-K on a set of questions, and report which
share of those questions at least one of the top-1..K programs solves (coverage vs. menu size K).
Every top-K program is executed on every question, since the search tried each program only on
some of them.

Which questions:
  --rank-pool all --eval-pool all         (default) rank and execute on train+val+test
                                          (in-sample; paper Fig. 4a and Fig. 9, the grids in
                                          data/menu_crossexec/)
  --rank-pool trainval --eval-pool test   rank on train+val, execute on test: the menu is chosen
                                          without seeing the questions it is scored on (the
                                          held-out check, data/menu_crossexec_heldout/;
                                          compare both with analysis.mcts_design.menu_heldout)

Reuse: a (program, question) cell the search already graded valid/invalid is taken from the
search, and with --reuse-grid every cell of another output grid of this script is taken from
that grid; only the remaining cells are executed, with `GenerationReward.__call__` (serial,
not masked-batch).

Output: one per-program x per-question grid per difficulty. `qids` is written once; each program
has a same-length list of {0,1} rewards and a provenance string, one letter per question:
'v' / 'i' = valid / invalid in the search, 'g' = from --reuse-grid, 'f' = executed in this run.

The paper's grids use the seed-42 search of each model and the oneshot prompt, the search's own:

  python -m analysis.mcts_design.menu_coverage \
      --model qwen25_3b --difficulty 1 --k 100 --batch-size 128 \
      --data-dir ./data/dart_math --mcts-dir ./data/mcts_results/qwen25_3b/seed42 \
      --output ./results/menu_crossexec/qwen25_3b/diff1.json

Held-out grid of the same cell (cells already in the in-sample grid are reused):

  python -m analysis.mcts_design.menu_coverage \
      --model qwen25_3b --difficulty 1 --k 100 --batch-size 128 \
      --data-dir ./data/dart_math --mcts-dir ./data/mcts_results/qwen25_3b/seed42 \
      --rank-pool trainval --eval-pool test \
      --reuse-grid ./data/menu_crossexec/qwen25_3b/diff1.json.gz \
      --output ./results/menu_crossexec_heldout/qwen25_3b/diff1.json

Top level is stdlib-only so nothing here needs torch until `main()` runs.
"""

import argparse
import gzip
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
    """Rank programs, identity included, by how many questions in `pool_qids` the search
    graded them valid on (most first), and return the top-k with their known
    (qid -> reward) cells on those questions. Identity is valid for every question the
    standard forward pass answers correctly, so it typically ranks first."""
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


def known_labels(mcts_samples, qids, paths):
    """(path -> {qid: reward}) for `paths` on `qids`, from the search's valid/invalid logs."""
    paths = set(paths)
    known = defaultdict(dict)
    for sample in mcts_samples:
        qid = sample.get("sample_info", {}).get("query_id")
        if qid not in qids:
            continue
        for p in sample.get("final_valid_transitions", []):
            if tuple(p) in paths:
                known[tuple(p)][qid] = 1.0
        for p in sample.get("final_invalid_transitions", []):
            if tuple(p) in paths:
                known[tuple(p)].setdefault(qid, 0.0)
    return known


def grid_labels(grid_path, difficulty, drop=""):
    """(path -> {qid: reward}) of an output grid of this script, for `difficulty`; cells whose
    provenance letter is in `drop` are left out (and so re-executed)."""
    p = Path(grid_path)
    grid = json.loads(gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text())
    (pd,) = [d for d in grid["per_difficulty"] if d["difficulty"] == difficulty]
    return {
        tuple(row["path"]): {
            q: float(r)
            for q, r, pv in zip(pd["qids"], row["rewards"], row["provenance"])
            if pv not in drop
        }
        for row in pd["per_program"]
    }


def find_mcts_file(mcts_dir, model_id, diff) -> Path:
    """merged_mcts_samples.json[.gz] of one level, in either layout: a released
    <model>/ directory (mcts_dir/dart-math-diff-N/...) or a run_search_dart_math
    output root (mcts_dir/<model_id>/dart-math-diff-N/...)."""
    ns = f"dart-math-diff-{diff}"
    for d in (Path(mcts_dir) / ns, Path(mcts_dir) / model_id / ns):
        for name in ("merged_mcts_samples.json", "merged_mcts_samples.json.gz"):
            if (d / name).exists():
                return d / name
    raise FileNotFoundError(f"no {ns}/merged_mcts_samples.json[.gz] under {mcts_dir}")


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
    p.add_argument(
        "--data-dir", required=True, help="DART-Math root (has diff{N}/{train,val,test}.json)"
    )
    p.add_argument(
        "--mcts-dir",
        required=True,
        help="one model's MCTS results (dart-math-diff-N/merged_mcts_samples.json[.gz]), or a "
        "run_search_dart_math --output-root",
    )
    p.add_argument(
        "--prompt-style",
        default="oneshot",
        help="GenerationReward prompt style; must be the prompt the search ran with, since "
        "known (program, question) labels are reused from it. The paper's grids: oneshot.",
    )
    p.add_argument(
        "--rank-pool",
        choices=("all", "trainval"),
        default="all",
        help="questions whose search counts rank the programs: all = train+val+test (in-sample, "
        "the grids in data/menu_crossexec/); trainval = train+val only (held-out menu)",
    )
    p.add_argument(
        "--eval-pool",
        choices=("all", "test"),
        default="all",
        help="questions the top-K programs are executed on: all = train+val+test; "
        "test = the test questions only (held-out menu)",
    )
    p.add_argument(
        "--reuse-grid",
        default=None,
        help="output grid of this script for the same model and prompt (.json or .json.gz): "
        "its cells are reused (provenance 'g') instead of re-executed",
    )
    p.add_argument(
        "--reuse-grid-drop",
        default="",
        help="provenance letters of --reuse-grid cells NOT to reuse, e.g. 'i' when the grid's "
        "known-invalid cells come from an older search; they are re-executed unless the "
        "--mcts-dir search knows the cell itself",
    )
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="cap n_pool questions, smoke-test only")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore already-known (program, question) labels from --mcts-dir and "
        "generate every cell fresh instead of reusing them.",
    )
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    diffs = resolve_difficulties(args.difficulty)

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    reward = GenerationReward(executor, batch_size=args.batch_size, prompt_style=args.prompt_style)

    results = []
    for diff in diffs:
        train_data = load_json(Path(args.data_dir) / f"diff{diff}" / "train.json")
        val_data = load_json(Path(args.data_dir) / f"diff{diff}" / "val.json")
        test_data = load_json(Path(args.data_dir) / f"diff{diff}" / "test.json")
        rank_qid_set = {
            d["query_id"]
            for d in (
                train_data + val_data
                if args.rank_pool == "trainval"
                else train_data + val_data + test_data
            )
        }
        pool_data = test_data if args.eval_pool == "test" else train_data + val_data + test_data
        if args.limit:
            pool_data = pool_data[: args.limit]
        pool_qids = [d["query_id"] for d in pool_data]
        pool_qid_set = set(pool_qids)
        assert len(pool_qids) == len(
            pool_qid_set
        ), f"diff{diff}: duplicate query_id across train/val/test"
        gt_by_qid = {d["query_id"]: d["gt_ans"] for d in pool_data}
        q_by_qid = {d["query_id"]: d["question"] for d in pool_data}
        n_pool = len(pool_qids)
        print(
            f"[diff {diff}] {n_pool} {args.eval_pool} questions (train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)})",
            flush=True,
        )

        mcts_samples = load_samples(find_mcts_file(args.mcts_dir, cfg["model_id"], diff))
        ranked = build_candidate_pool(mcts_samples, rank_qid_set, args.k)
        known_eval = known_labels(mcts_samples, pool_qid_set, [path for path, _ in ranked])
        candidates = [(path, known_eval.get(path, {})) for path, _ in ranked]
        reused_grid = (
            grid_labels(args.reuse_grid, diff, args.reuse_grid_drop) if args.reuse_grid else {}
        )
        print(
            f"[diff {diff}] top-{len(candidates)} candidate programs selected (ranked over "
            f"{len(rank_qid_set)} {args.rank_pool} questions, executed on {n_pool} "
            f"{args.eval_pool} questions; {sum(p in reused_grid for p, _ in candidates)} "
            f"of them in --reuse-grid)",
            flush=True,
        )

        n_reused_total = 0
        n_new_total = 0
        per_program_rows = []  # {"path": [...], "rewards": [...], "provenance": "..."}
        for path, known in candidates:
            row = {} if args.fresh else dict(known)
            provenance = {}  # qid -> 'v'/'i' for known cells, filled in below
            if not args.fresh:
                for qid, r in known.items():
                    provenance[qid] = "v" if r > 0 else "i"
                for qid, r in reused_grid.get(path, {}).items():
                    if qid in pool_qid_set and qid not in row:
                        row[qid] = r
                        provenance[qid] = "g"
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
            per_program_rows.append(
                {"path": list(path), "rewards": rewards_list, "provenance": provenance_str}
            )
            solved_now = sum(rewards_list)
            print(
                f"  program {list(path)}: reused {n_pool - len(todo_qids)}, "
                f"new {len(todo_qids)}, total solved {solved_now}/{n_pool}",
                flush=True,
            )

        # real top-k coverage curve: in the candidates' rank order (by search-frequency
        # over --rank-pool), how much of --eval-pool does ANY of the top-1..top-K solve,
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
            "n_pool": n_pool,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "k_requested": args.k,
            "k_actual": len(candidates),
            "n_reused_cells": n_reused_total,
            "n_new_cells": n_new_total,
            "real_topk_coverage_curve": coverage_curve,
            "final_real_coverage": coverage_curve[-1]["real_coverage"] if coverage_curve else 0.0,
            "qids": pool_qids,  # written ONCE per difficulty; per_program rows are positional against this
            "provenance_legend": {
                "v": "known valid from original MCTS search (reused, not re-executed)",
                "i": "known invalid from original MCTS search (reused, not re-executed)",
                "g": "reused from --reuse-grid (not re-executed)",
                "f": "freshly executed this run",
            },
            "rank_pool": args.rank_pool,
            "eval_pool": args.eval_pool,
            "reuse_grid": args.reuse_grid,
            "reuse_grid_drop": args.reuse_grid_drop,
            "per_program": per_program_rows,
        }
        results.append(row_out)
        print(
            f"[diff {diff}] DONE. real top-{len(candidates)} coverage = "
            f"{row_out['final_real_coverage']:.4f} "
            f"({sum(any_solved)}/{n_pool} pool questions), "
            f"reused {n_reused_total} cells, ran {n_new_total} new evaluations",
            flush=True,
        )

    out = {
        "model": args.model,
        "k": args.k,
        "prompt_style": args.prompt_style,
        "schema": "menu_crossexec_v2",
        "per_difficulty": results,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out))
    print(f"\nwrote {outp}", flush=True)

    print(f"\n{'diff':>4} {'k':>4} {'real_cov':>9} {'new_evals':>10}")
    for r in results:
        print(
            f"{r['difficulty']:>4} {r['k_actual']:>4} {r['final_real_coverage']:>9.4f} "
            f"{r['n_new_cells']:>10}"
        )
    return outp


if __name__ == "__main__":
    main()
