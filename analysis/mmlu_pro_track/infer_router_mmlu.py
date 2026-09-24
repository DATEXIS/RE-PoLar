"""Held-out TEST eval for an mmlu_pro_domains router checkpoint -- the
LogLikReward analog of re_polar/router/infer.py.

TEST is always `<data_dir>/test.json`, the carved held-out split
`re_polar.datasets.mmlu_pro_domains.build_train_split` produces from the local
`mmlu_pro_official` pool (all 14 native MMLU-Pro categories by default).
TEST is DISJOINT from `re_polar/datasets/mmlu_pro_domains.py`'s TRAIN/VAL split by
construction (carved out of the pool before train/val are drawn) -- never
regenerated or modified here.

Reuses `re_polar.router.infer`'s reward-fn-agnostic helpers UNCHANGED
(`predict_programs`/`predict_programs_topk`/`assert_roundtrips`/
`grade_router`/`grade_router_topk`/`summarize`) -- only the data source
(fixture rows, not diff{N}/test.json) and the reward function (LogLikReward,
not GenerationReward) differ, exactly as `re_polar/router/mmlu_bridge.py` bridges
`re_polar.router.train` for the training side. Two parallel views of each
question are threaded through: a formatted STRING (`format_mmlu_question`,
router forward / encoder input) and the structured dict + int answer_index
(`LogLikReward` grading input) -- they must stay index-aligned.

`--random-baseline-seeds` is this script's other real job (beyond a plain
router eval): grades N sets of 5 SHARED random valid programs (same action
grammar the router itself decodes from) against the identical test set and
protocol, to answer whether a router's pass@5 margin over identity reflects
learned, per-question signal or just "trying 5 candidates on a forced-choice
benchmark is a free lift regardless of which 5."

Run::

    python -m analysis.mmlu_pro_track.infer_router_mmlu \\
        --checkpoint router_qwen3_8b_mmlu.pt \\
        --model qwen3_8b --data-dir mmlu_pro_domains \\
        --output test_eval_mmlu.json \\
        --top-k-paths 5 --random-baseline-seeds 5

`identity_acc` printed/written here for the FULL test set is the reference
point for the router pass@1 comparison (`overall_summary["delta"]`,
`gate1_pass`).
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from re_polar.models import MODEL_REGISTRY

__all__ = ["load_test_records", "random_program", "pass_at_k_table", "main"]


def random_program(num_layers, rng):
    """One random valid program: contiguous segments, ops SKIP/KEEP/REPEAT
    uniform, same grammar as the router's own action space."""
    from re_polar.core import Op, Program, Segment
    from re_polar.core.grammar import is_valid

    while True:
        segs = []
        cursor = 0
        while cursor < num_layers:
            length = rng.randint(1, min(4, num_layers - cursor))
            op = rng.choice([Op.SKIP, Op.KEEP, Op.REPEAT])
            params = {"times": 2} if op is Op.REPEAT else {}
            segs.append(Segment(start=cursor, end=cursor + length, op=op, params=params))
            cursor += length
        program = Program(num_layers=num_layers, segments=segs)
        if is_valid(program):  # rejects all-skip
            return program


def pass_at_k_table(topk_programs, question_dicts, gt_answers, reward_fn, k_max=5):
    """{k: pass@k} for k=1..k_max, via grade_router_topk on progressively
    truncated candidate lists -- lets the random-baseline control reuse
    grade_router_topk without re-running the model per k."""
    from re_polar.router.infer import grade_router_topk

    result = {}
    for k in range(1, k_max + 1):
        truncated = [cands[:k] for cands in topk_programs]
        r_at_k, _, _ = grade_router_topk(truncated, question_dicts, gt_answers, reward_fn)
        result[k] = sum(r_at_k) / len(r_at_k) if r_at_k else 0.0
    return result


def load_test_records(data_dir: str, domains: Optional[List[str]] = None) -> List[dict]:
    """`<data_dir>/test.json`, filtered to `domains` if a subset was requested."""
    records = json.loads((Path(data_dir) / "test.json").read_text())
    if domains is not None:
        wanted = set(domains)
        records = [r for r in records if str(r["category"]) in wanted]
    return records


def per_domain_summaries(
    categories: List[str],
    domains: List[str],
    rewards_at_1: List[float],
    programs: List,
    identity_rewards: List[float],
    num_layers: int,
) -> List[dict]:
    """Per-domain slice of `re_polar.router.infer.summarize`, the mmlu analog of
    infer.py's per-difficulty loop. Domains absent from `categories` are
    skipped (not reported as an empty/zero row)."""
    from re_polar.router.infer import summarize

    out = []
    for domain in domains:
        idxs = [i for i, c in enumerate(categories) if c == domain]
        if not idxs:
            continue
        d_summary = summarize(
            [rewards_at_1[i] for i in idxs],
            [programs[i] for i in idxs],
            [identity_rewards[i] for i in idxs],
            num_layers,
        )
        d_summary["domain"] = domain
        d_summary["gate1_pass"] = d_summary["router_acc"] >= d_summary["identity_acc"]
        out.append(d_summary)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mmlu_pro_domains router held-out TEST eval.")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="router .pt from sweep_router_mmlu.py / re_polar.router.train",
    )
    p.add_argument(
        "--model",
        default="qwen3_8b",
        choices=sorted(MODEL_REGISTRY),
        help="target model in MODEL_REGISTRY (sets router D); default qwen3_8b",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="mmlu_pro_domains split dir (re_polar/datasets/mmlu_pro_domains.py "
        "build_train_split output). TEST = <data-dir>/test.json.",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="subset of domains to evaluate; default = all domains present "
        "in <data-dir>/test.json",
    )
    p.add_argument("--output", required=True, help="path to write the results JSON")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="LogLikReward forward batch (also router forward batch)",
    )
    p.add_argument(
        "--top-k-paths",
        type=int,
        default=1,
        help="decode top-k programs per question and score pass@k; default 1 == top-1 pass@1",
    )
    p.add_argument("--limit", type=int, default=None, help="first N test questions (smoke)")
    p.add_argument("--device", default=None, help="cpu / cuda / mps for the router (default: auto)")
    p.add_argument(
        "--random-baseline-seeds",
        type=int,
        default=0,
        help="add N random-baseline rows: 5 SHARED random valid programs (same "
        "grammar as the router's own search space), reused for every TEST "
        "question -- what pass@1..5 looks like with zero learned signal. "
        "Same reward_fn/protocol/test set as the router eval above. Prints "
        "per-seed + mean/std and adds a 'random_baseline' block to the "
        "output JSON.",
    )
    p.add_argument("--seed", type=int, default=0, help="base seed for --random-baseline-seeds")
    return p


def main(argv: Optional[List[str]] = None) -> Path:
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import LogLikReward
    from re_polar.core import Program
    from re_polar.router.infer import (
        assert_roundtrips,
        grade_router,
        grade_router_topk,
        predict_programs,
        predict_programs_topk,
        summarize,
    )
    from re_polar.router.mmlu_bridge import format_mmlu_question
    from re_polar.router.train import load_checkpoint

    args = _build_arg_parser().parse_args(argv)

    fixture = load_test_records(args.data_dir, args.domains)
    if args.limit is not None:
        fixture = fixture[: args.limit]
    domains = args.domains or sorted({str(r["category"]) for r in fixture})
    print(
        f"Loaded {len(fixture)} TEST questions from {args.data_dir}/test.json (domains={domains})."
    )

    def _resolve_device(name):
        import torch

        if name:
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = _resolve_device(args.device)
    cfg = MODEL_REGISTRY[args.model]
    no_think = cfg.get("no_think", True)
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    reward_fn = LogLikReward(executor, batch_size=args.batch_size, no_think=no_think)

    router = load_checkpoint(args.checkpoint)
    if router.num_layers != D:
        raise SystemExit(
            f"Router D={router.num_layers} != target model {args.model} D={D}; "
            "the checkpoint was trained for a different model."
        )
    router.eval()
    router.encode_questions(["warmup"])
    router.to(device)

    question_dicts = [
        {"question": r["question"], "options": r["options"], "category": r["category"]}
        for r in fixture
    ]
    question_strs = [format_mmlu_question(q, no_think=no_think) for q in question_dicts]
    gt_answers = [int(r["answer_index"]) for r in fixture]
    categories = [str(r["category"]) for r in fixture]
    identity_program = Program.identity(D)

    k = max(1, args.top_k_paths)
    print(
        f"Predicting programs{f' (top-{k})' if k > 1 else ''} for {len(question_strs)} questions..."
    )
    identity_rewards = reward_fn(identity_program, question_dicts, gt_answers)
    if k == 1:
        programs = predict_programs(router, question_strs, batch_size=args.batch_size)
        assert_roundtrips(programs, D)
        router_rewards = grade_router(programs, question_dicts, gt_answers, reward_fn)
        overall_summary = summarize(router_rewards, programs, identity_rewards, D)
        rewards_at_1 = router_rewards
    else:
        topk = predict_programs_topk(router, question_strs, k=k, batch_size=args.batch_size)
        programs = [cands[0] for cands in topk]
        assert_roundtrips(programs, D)
        rewards_at_k, rewards_at_1, _chosen = grade_router_topk(
            topk, question_dicts, gt_answers, reward_fn
        )
        overall_summary = summarize(rewards_at_1, programs, identity_rewards, D)
        overall_summary["top_k"] = k
        overall_summary["router_acc_at_k"] = (
            sum(rewards_at_k) / len(rewards_at_k) if rewards_at_k else 0.0
        )
        overall_summary["delta_at_k"] = (
            overall_summary["router_acc_at_k"] - overall_summary["identity_acc"]
        )
        overall_summary["gate1_at_k_pass"] = (
            overall_summary["router_acc_at_k"] >= overall_summary["identity_acc"]
        )
        overall_summary["mean_candidates"] = sum(len(c) for c in topk) / len(topk) if topk else 0.0
    overall_summary["gate1_pass"] = overall_summary["router_acc"] >= overall_summary["identity_acc"]

    # per-domain breakdown (the mmlu analog of infer.py's per-difficulty loop)
    per_domain = per_domain_summaries(
        categories, domains, rewards_at_1, programs, identity_rewards, D
    )

    random_baseline = None
    if args.random_baseline_seeds > 0:
        import random as _random
        import statistics

        print(
            f"\nRandom-valid-program baseline ({args.random_baseline_seeds} seed(s), 5 shared "
            f"random programs per seed, same protocol/test set as the router above)..."
        )
        per_seed = {}
        for i in range(args.random_baseline_seeds):
            rng = _random.Random(args.seed + i)
            five = [random_program(D, rng) for _ in range(5)]
            topk_random = [five for _ in question_dicts]
            table = pass_at_k_table(topk_random, question_dicts, gt_answers, reward_fn)
            per_seed[i] = table
            print(f"  seed{i}: " + " ".join(f"@{kk}={table[kk]:.4f}" for kk in range(1, 6)))
        random_baseline = {"per_seed": per_seed}
        if args.random_baseline_seeds > 1:
            mean_std = {}
            for kk in range(1, 6):
                vals = [per_seed[i][kk] for i in range(args.random_baseline_seeds)]
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                mean_std[kk] = {"mean": m, "std": s}
            random_baseline["mean_std"] = mean_std
            print(
                "  mean +/- std across seeds: "
                + " ".join(
                    f"@{kk}={mean_std[kk]['mean']:.4f}+/-{mean_std[kk]['std']:.4f}"
                    for kk in range(1, 6)
                )
            )

    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "domains": domains,
        "data_dir": args.data_dir,
        "test_source": f"{args.data_dir}/test.json",
        "overall": overall_summary,
        "per_domain": per_domain,
        "random_baseline": random_baseline,
        "programs": [
            {"category": cat, "layer_path": p.to_layer_path()}
            for cat, p in zip(categories, programs)
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(
        f"\nRouter inference (mmlu_pro_domains), {args.model} (D={D}), checkpoint {args.checkpoint}"
    )
    print(
        f"OVERALL: router_acc={overall_summary['router_acc']:.4f} identity_acc={overall_summary['identity_acc']:.4f} "
        f"delta={overall_summary['delta']:+.4f}  {'PASS' if overall_summary['gate1_pass'] else 'FAIL'}"
    )
    for d in per_domain:
        print(
            f"  {d['domain']:>18}: router={d['router_acc']:.4f} identity={d['identity_acc']:.4f} "
            f"delta={d['delta']:+.4f}  {'PASS' if d['gate1_pass'] else 'FAIL'}"
        )
    if random_baseline is not None and "mean_std" in random_baseline:
        ms = random_baseline["mean_std"]
        router_at_k = overall_summary.get("router_acc_at_k")
        print(
            f"\nSIGNAL CHECK (identity vs router@{k} vs random-5-programs@{k}, same TEST set/protocol):"
        )
        print(f"  identity_acc  = {overall_summary['identity_acc']:.4f}")
        if router_at_k is not None:
            print(f"  router_acc@{k}  = {router_at_k:.4f}")
        print(
            f"  random_acc@{k}  = {ms[k]['mean']:.4f} +/- {ms[k]['std']:.4f}  "
            f"(n_seeds={args.random_baseline_seeds})"
        )
        if router_at_k is not None:
            verdict = (
                "ROUTER BEATS RANDOM"
                if router_at_k > ms[k]["mean"] + ms[k]["std"]
                else "ROUTER <= RANDOM (no clear signal beyond try-5-candidates luck)"
            )
            print(f"  -> {verdict}")
    print(f"\nWrote results -> {out_path}")

    return out_path


if __name__ == "__main__":
    main()
