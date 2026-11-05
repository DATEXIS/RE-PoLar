"""Fixed-program baseline: one program-of-layers per difficulty level, the same for every input.

The paper's "fixed program" (Section 4.1 metrics, Table 2, Appendix B): a menu of size
one, picked on validation questions, then applied to every test question.

  candidates  the --fixed-top non-identity programs with the highest shrunk success rate
              on the TRAIN questions' MCTS verdicts (MCTS only tries promising programs,
              so raw observed rates are biased upward; each program's rate is shrunk
              toward the global non-identity success rate with --prior-strength
              pseudo-observations). Menu = programs valid on >= --min-count train questions.
  selection   each candidate is run on every VAL question (real generation, the MCTS
              reward and prompt); the best val pass@1 wins (ties: pass@k, where the
              candidate's top-k list is it plus the next k-1 candidates by train prior).
  test        the winner on every test question: identity accuracy, pass@1..k, rescue@1
              (identity wrong, program right), harm@1 (identity right, program wrong),
              exact McNemar p, per-question verdicts.

Protocol (no test leakage): train = samples[:train_n], val = the next val_n (the order of
a --split all MCTS file, or of re_polar.router.fold_data's --out-samples, asserted
against --data-dir's split files); test = --data-dir/diffN/test.json, or --test-json
(flat OOD sets, one program for all levels with --difficulties / --train-val-counts).

Main track (our split, one level):
  python -m re_polar.router.fixed_program --model qwen3_8b --difficulty 1 \\
      --samples <mcts>/dart-math-diff-1/merged_mcts_samples.json --data-dir ./data/dart_math \\
      --out ./results/fixed/qwen3_8b_diff1.json --text-log ./results/fixed/qwen3_8b_diff1.texts.jsonl
PoLar split (one fold): --samples / --data-dir from re_polar.router.fold_data,
--train-n N_TRAIN --val-n N_VAL, --prompt-style polar_default or oneshot.

The val questions are generated on, so --samples must carry question text: the released
MCTS files don't, run re_polar.datasets.attach_mcts_questions first (fold_data's
--out-samples already has it).
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main(argv=None) -> None:
    import numpy as np

    from re_polar.core import ProgramExecutor
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.datasets.schemas import load_samples
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.models import MODEL_REGISTRY
    from re_polar.router.infer import (
        grade_topk_pairs,
        load_test_json,
        mcnemar_exact_p,
        pass_at_j_curve,
    )
    from re_polar.router.train import program_from_layer_path

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="qwen3_8b", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--difficulty", type=int, default=None, help="level (single-level runs)")
    p.add_argument(
        "--samples",
        required=True,
        nargs="+",
        help="merged_mcts_samples.json in train/val/test order (several: one per --difficulties entry)",
    )
    p.add_argument(
        "--difficulties", default=None, help="comma-sep level of each --samples file (OOD runs)"
    )
    p.add_argument(
        "--train-val-counts", default=None, help="comma-sep NTRAIN:NVAL per --samples file"
    )
    p.add_argument(
        "--test-json",
        nargs="+",
        default=None,
        help="flat eval sets used as the test set instead of --data-dir/diffN/test.json (concatenated)",
    )
    p.add_argument("--data-dir", required=True, help="dir with diffN/{train,val,test}.json")
    p.add_argument("--train-n", type=int, default=1250)
    p.add_argument("--val-n", type=int, default=250)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--prior-strength", type=float, default=10.0)
    p.add_argument(
        "--fixed-top",
        type=int,
        default=20,
        help="val-evaluate this many non-identity programs with the highest shrunk train prior",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--test-limit", type=int, default=None, help="smoke tests only")
    p.add_argument("--prompt-style", default="oneshot")
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument(
        "--text-log", default=None, help="JSONL of every generated answer (re_polar.mcts.textlog)"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    def qids(xs):
        return [x.get("query_id") or x["sample_info"]["query_id"] for x in xs]

    if args.difficulty is None and not args.difficulties:
        raise SystemExit("pass --difficulty (or --difficulties)")
    if args.difficulties and len(args.difficulties.split(",")) > 1 and not args.test_json:
        raise SystemExit("a multi-level program needs --test-json (one test set for all levels)")
    levels = (
        [int(d) for d in args.difficulties.split(",")] if args.difficulties else [args.difficulty]
    )
    counts = (
        [tuple(int(x) for x in c.split(":")) for c in args.train_val_counts.split(",")]
        if args.train_val_counts
        else [(args.train_n, args.val_n)] * len(levels)
    )
    if not len(levels) == len(counts) == len(args.samples):
        raise SystemExit("--samples / --difficulties / --train-val-counts lengths differ")
    train, val = [], []
    for f, lvl, (n_tr, n_va) in zip(args.samples, levels, counts):
        samples = load_samples(Path(f))
        tr, va = samples[:n_tr], samples[n_tr : n_tr + n_va]
        ddir = Path(args.data_dir) / f"diff{lvl}"
        split = {n: json.load(open(ddir / f"{n}.json")) for n in ("train", "val", "test")}
        assert qids(tr) == qids(split["train"])[: len(tr)], f"samples order != data-dir train ({f})"
        assert qids(va) == qids(split["val"])[: len(va)], f"samples order != data-dir val ({f})"
        train, val = train + tr, val + va
    if args.test_json:
        test = [d for t in args.test_json for d in load_test_json(t)][: args.test_limit]
    else:
        test = split["test"][: args.test_limit]
    assert not set(qids(test)) & set(qids(train) + qids(val)), "test overlaps train/val"
    if any("question" not in x for x in val):
        raise SystemExit(
            "val samples carry no question text; run re_polar.datasets.attach_mcts_questions "
            "(or re_polar.router.fold_data) first"
        )
    print(f"levels {levels}: train {len(train)} / val {len(val)} / test {len(test)}", flush=True)

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    D = engine.num_layers
    reward_fn = GenerationReward(
        ProgramExecutor(engine),
        batch_size=args.gen_batch_size,
        prompt_style=args.prompt_style,
        text_log_path=args.text_log,
    )

    # ---------------------------------------------------------------- menu + train prior
    ident = tuple(range(D))
    valid_counts = Counter(tuple(v) for s in train for v in s["final_valid_transitions"])
    cands = sorted(
        (pth for pth, c in valid_counts.items() if c >= args.min_count and pth != ident),
        key=lambda pth: -valid_counts[pth],
    )

    def parse(pth):
        try:
            return program_from_layer_path(list(pth), D, strict_repeat_2x=False)
        except (
            ValueError
        ):  # Li et al.'s search can reach nested repeats, which have no Program form
            return None

    parsed = {pth: parse(pth) for pth in cands}
    skipped = [pth for pth in cands if parsed[pth] is None]
    menu = [ident] + [pth for pth in cands if parsed[pth] is not None]
    programs = [program_from_layer_path(list(ident), D, strict_repeat_2x=False)] + [
        parsed[pth] for pth in menu[1:]
    ]
    col = {pth: j for j, pth in enumerate(menu)}
    Y = np.full((len(train), len(menu)), np.nan, dtype=np.float32)
    for i, s in enumerate(train):
        for v in s["final_invalid_transitions"]:
            if tuple(v) in col:
                Y[i, col[tuple(v)]] = 0.0
        for v in s["final_valid_transitions"]:
            if tuple(v) in col:
                Y[i, col[tuple(v)]] = 1.0
        Y[i, 0] = float(s["initial_transition_metric"] == 1)
    obs = ~np.isnan(Y)
    mu = float(np.nansum(Y[:, 1:]) / max(obs[:, 1:].sum(), 1))
    prior = (np.nansum(Y, 0) + args.prior_strength * mu) / (obs.sum(0) + args.prior_strength)
    prior[0] = Y[:, 0].mean()
    print(
        f"menu: {len(menu) - 1} programs valid on >= {args.min_count} train questions (+identity; "
        f"{len(skipped)} nested paths skipped); identity correct on train {Y[:, 0].mean():.3f}; "
        f"non-identity success rate mu {mu:.3f}",
        flush=True,
    )

    # ---------------------------------------------------------------- grading (memoised)
    memo = {}

    def memo_reward(program, qs, gts):
        key = tuple(program.to_layer_path())
        todo = [(q, g) for q, g in zip(qs, gts) if (key, q) not in memo]
        if todo:
            for (q, _), r in zip(
                todo, reward_fn(program, [q for q, _ in todo], [g for _, g in todo])
            ):
                memo[(key, q)] = float(r)
        return [memo[(key, q)] for q in qs]

    def evaluate(lst, xs, detail=False):
        topk = [[programs[j] for j in lst]] * len(xs)
        qs, gts = [x["question"] for x in xs], [x["gt_ans"] for x in xs]
        pairs = grade_topk_pairs(topk, qs, gts, memo_reward)
        ident_r = memo_reward(programs[0], qs, gts)
        top1 = [pairs[(tuple(programs[lst[0]].to_layer_path()), i)] for i in range(len(xs))]
        n = len(xs)
        rescue = sum(t >= 1 and r < 1 for t, r in zip(top1, ident_r))
        harm = sum(t < 1 and r >= 1 for t, r in zip(top1, ident_r))
        out = {
            "pass_at": pass_at_j_curve(topk, pairs),
            "identity_acc": sum(ident_r) / n,
            "rescue_at1": rescue / n,
            "harm_at1": harm / n,
            "nonident_top1": float(lst[0] != 0),
        }
        if detail:
            out.update(
                {
                    "mcnemar_p": mcnemar_exact_p(rescue, harm),
                    "n_distinct_top1": 1,
                    "top_nonidentity": (
                        [{"layer_path": programs[lst[0]].to_layer_path(), "n": n}]
                        if lst[0] != 0
                        else []
                    ),
                    "per_question": [
                        {
                            "query_id": x.get("query_id"),
                            "top1": programs[lst[0]].to_layer_path(),
                            "identity_correct": bool(r >= 1),
                            "top1_correct": bool(tp >= 1),
                        }
                        for x, tp, r in zip(xs, top1, ident_r)
                    ],
                }
            )
        return out

    # ---------------------------------------------------------------- candidates on val
    order = [int(j) for j in np.argsort(-prior[1:], kind="stable") + 1]
    results = []
    for c, j in enumerate(order[: args.fixed_top], 1):
        lst = [j] + [x for x in order if x != j][: args.k - 1]
        r = evaluate(lst, val)
        params = {
            "program": j,
            "layer_path": list(menu[j]),
            "train_prior": round(float(prior[j]), 4),
        }
        results.append({"params": params, "val": r, "list": lst})
        print(
            f"[val {c}/{min(args.fixed_top, len(order))}] program {j} (len {len(menu[j])}, train "
            f"prior {prior[j]:.3f}): pass@1 {r['pass_at'][0]:.4f} pass@{args.k} "
            f"{r['pass_at'][-1]:.4f} identity {r['identity_acc']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    w = max(results, key=lambda x: (x["val"]["pass_at"][0], x["val"]["pass_at"][-1]))

    # ---------------------------------------------------------------- winner on test
    r = evaluate(w["list"], test, detail=True)
    out = {
        "difficulty": args.difficulty,
        "menu_size": len(menu),
        "prompt_style": args.prompt_style,
        "samples": args.samples[0] if len(args.samples) == 1 else args.samples,
        "difficulties": levels,
        "test_json": args.test_json,
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "methods": {
            "fixed": {
                "winner": w["params"],
                "val": w["val"],
                "test": r,
                "all_val": [{"params": x["params"], "val": x["val"]} for x in results],
            }
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(
        f"\nSUMMARY levels {levels}, test {len(test)} questions: fixed program "
        f"{w['params']['layer_path']} -> pass@1 {r['pass_at'][0]:.4f} vs identity "
        f"{r['identity_acc']:.4f} (delta {r['pass_at'][0] - r['identity_acc']:+.4f}), "
        f"rescue@1 {r['rescue_at1']:.4f} harm@1 {r['harm_at1']:.4f} McNemar p {r['mcnemar_p']:.3f} "
        f"({time.time() - t0:.0f}s) -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
