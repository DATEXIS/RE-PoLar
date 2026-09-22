"""MCTS program discovery over mmlu_pro_domains (LogLikReward), the
"own data + models" analog of run_search_dart_math.py.

Differs from the DART-Math runner in three ways:
  - reward = LogLikReward (one log-likelihood forward pass, no generation/
    grading loop) instead of GenerationReward -- ~100x cheaper, what makes
    27-32B MCTS feasible (re_polar/mcts/rewards.py).
  - split by --domains (mmlu_pro_domains categories: math, physics,
    chemistry, law, history, "computer science", ...) instead of
    --difficulty.
  - --n-per-domain subsamples EACH requested domain (seeded, without
    replacement) instead of a flat --n-inputs prefix -- keeps a pilot
    balanced across domains instead of favoring whichever domain sorts first
    in train.json.

Faithful search config: --max-repeat-times defaults to 5 (r<=4 == a segment
executed up to 5x), unlike the DART-Math runner's r=2 default (kept there
for backward compatibility with an earlier, under-faithful search).

Pilot (do valid SHORTER programs exist on mmlu_pro_domains at all):
  python -m re_polar.mcts.run_search_mmlu_pro_domains --model qwen3_8b \
      --n-per-domain 50 --budget 100 \
      --data-dir ./data/mmlu_pro_domains --output-root ./data/mcts \
      --cache-dir ./data/mcts_cache

Full run (all rows, no holdout): omit --n-per-domain, pass --split all, and
make sure the mmlu_pro_domains split itself was built UNCAPPED
(`re_polar.datasets.mmlu_pro_domains --n-per-domain` large enough that no
domain's pool got truncated -- otherwise "all" is a subset, not the whole
dataset; see that module's docstring). Raise --budget as needed.
Output uses PoLar's exact merged_mcts_samples.json schema (reuses
re_polar/datasets/schemas.py unchanged) under namespace "mmlu-pro-domains".
"""

import argparse
import json
import random
import time
from pathlib import Path

# NOTE: keep this module's top level TORCH-FREE (mirrors run_search_dart_math.py --
# the grader path there re-imports __main__ in spawned workers; LogLikReward has no
# such worker pool, but the pattern costs nothing to keep and stays consistent).
from re_polar.models import MODEL_REGISTRY
from re_polar.datasets.mmlu_pro_domains import ALL_MMLU_PRO_DOMAINS, PAPER_MMLU_PRO_DOMAINS  # noqa: F401  (torch-free)

# kept so older runs stay reproducible: PAPER_MMLU_PRO_DOMAINS (13, no computer
# science) or this even older 6-domain set.
LEGACY_MMLU_PRO_DOMAINS = ["math", "physics", "chemistry", "law", "history", "computer science"]


def _load_inputs(data_dir: Path, domains, n_per_domain, seed: int,
                 split: str = "train") -> list:
    """train.json (flat, see re_polar/datasets/mmlu_pro_domains.py) -> MCTSRunner inputs:
    {"query_id", "question": {"question", "options", "category"}, "gt_ans", "category"}.
    LogLikReward expects `question` to carry this nested dict (mirrors the shape
    run_mmlu_pro_domains/MMLUProSample need) -- see rewards.py's LogLikReward docstring.

    `split` mirrors the DART-Math runner's own choices (including "all"):
    "trainval" reads train.json then val.json as one ordered pool. "traintest"
    reads train.json then test.json instead -- the default for a held-out-val
    full run: MCTS full runs search train+test, NOT val (val is held out for
    the router/other downstream uses rather than spent as extra MCTS search
    budget). "all" unions train+val+test -- literally the whole pool, no
    holdout at all (only correct if the split itself was built uncapped, i.e.
    `--n-per-domain` large enough that no domain's pool was truncated; see
    re_polar/datasets/mmlu_pro_domains.py). NOTE train.json/val.json/test.json
    each number their `id` field from 0, so the query_id is prefixed with the
    split -- without that, concatenating them would silently collide every
    row onto a same-index row of another split, and MCTSRunner keys its trees
    and its eval cache by query_id."""
    if split == "trainval":
        splits = ["train", "val"]
    elif split == "traintest":
        splits = ["train", "test"]
    elif split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [split]
    records = []
    for s in splits:
        path = data_dir / f"{s}.json"
        if not path.exists():
            raise SystemExit(f"--split {split} needs {path}, which does not exist "
                             f"(rebuild with re_polar.datasets.mmlu_pro_domains --val-frac > 0)")
        for rec in json.load(open(path)):
            records.append((s, rec))
    by_domain: dict = {d: [] for d in domains}
    for s, rec in records:
        cat = str(rec["category"])
        if cat in by_domain:
            by_domain[cat].append((s, rec))

    rng = random.Random(seed)
    inputs = []
    for domain in domains:
        pool = by_domain[domain]
        if not pool:
            raise SystemExit(f"No {split} rows for domain {domain!r} in {data_dir}")
        chosen = rng.sample(pool, min(n_per_domain, len(pool))) if n_per_domain else pool
        for s, rec in chosen:
            qid = f"mmlu-{s}-{rec['id']}"
            inputs.append({
                "query_id": qid,
                # "query_id" here too (not just on the outer dict): LogLikReward's
                # probs-log side-channel (re_polar/mcts/rewards.py) only ever sees this
                # nested dict, not the outer one -- purely additive, ignored by
                # scoring (MMLUProSample only reads question/options/category).
                "question": {"question": rec["question"], "options": rec["options"],
                            "category": rec["category"], "query_id": qid},
                "gt_ans": int(rec["answer_index"]),
                "category": rec["category"],
            })
    return inputs


def main():
    from re_polar.datasets.schemas import sample_record, write_merged_samples
    from re_polar.mcts.scheduler import EvalCache, MCTSRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_8b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test", "trainval", "traintest", "all"],
                        help="all = train+val+test unioned, literally the whole pool, no "
                             "holdout at all (only correct if the split was built uncapped -- "
                             "see re_polar/datasets/mmlu_pro_domains.py). traintest = train then "
                             "test as one ordered pool -- the default for a HELD-OUT-val full "
                             "run (train+test, NOT val -- val held out for the router/other "
                             "downstream uses). trainval = train then val, an earlier full-run "
                             "convention. Mirrors the DART-Math runner's own --split trainval/"
                             "traintest/all exactly. query_ids are split-prefixed so rows from "
                             "different split files cannot collide.")
    parser.add_argument("--domains", nargs="+", default=ALL_MMLU_PRO_DOMAINS,
                        choices=sorted(set(ALL_MMLU_PRO_DOMAINS) | set(LEGACY_MMLU_PRO_DOMAINS)),
                        help="MMLU-Pro categories to search. Default = ALL native categories. "
                             "Pass PAPER_MMLU_PRO_DOMAINS's subset (excludes 'computer science') "
                             "to reproduce a paper-table-comparable run.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--n-per-domain", type=int, default=None,
                        help="pilot subset PER requested domain; default all")
    parser.add_argument("--budget", type=int, default=100, help="MCTS simulations per input")
    parser.add_argument("--c", type=float, default=1.414, help="UCB exploration constant")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Progressive-widening scale: a node with `visits` visits may hold "
                             "up to ceil(alpha * visits**beta) children before selection is "
                             "forced to descend via UCB instead of expanding a new one "
                             "(re_polar/mcts/search.py's _widening_limit). Mirrors "
                             "run_search_dart_math.py's own --alpha.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Progressive-widening exponent (see --alpha). 0.5 (O(sqrt(visits)) "
                             "growth) is the literature-standard choice (Couetoux et al. 2011).")
    parser.add_argument("--lam", type=float, default=5.0, help="length penalty weight. PRE "
                        "paper Appendix B (arXiv:2507.07996) states 5.0; pass --lam 0.5 to "
                        "reproduce an earlier default (0.5, 10x smaller).")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Probability of force-expanding a random untried edit instead of "
                             "UCB-selecting an existing child during descent. PRE paper Appendix "
                             "B: 'selects a random unexplored child node with probability 0.1 "
                             "instead of the one with the highest UCB score.' Default 0.0, "
                             "matching run_search_dart_math.py's own default (isolates "
                             "widening+global-V from epsilon as a separate axis). Pass "
                             "--epsilon 0.1 to reproduce the paper's stated value.")
    parser.add_argument("--global-selection", action="store_true",
                        help="Opt IN to global_selection: flat tree-wide UCB argmax + no "
                             "progressive widening + no transposition/DAG collapsing, one literal "
                             "reading of CoLa/PoLar's own Algorithm 1. OFF by default "
                             "(hierarchical+widening+DAG), matching run_search_dart_math.py's "
                             "own default.")
    parser.add_argument("--ucb-parent-v", action="store_true",
                        help="Use the selected node's own parent's visit count in UCB's explore "
                             "term (standard textbook UCT/UCB1) instead of the paper's literal "
                             "global V (total simulations across the whole tree so far), for the "
                             "hierarchical (non---global-selection) mode's descent. Global V is "
                             "the default (staying close to the papers' literal formula). Only "
                             "affects hierarchical mode -- --global-selection already always "
                             "uses global V.")
    parser.add_argument("--max-repeat-times", type=int, default=5,
                        help="Max execution count for a REPEAT segment. Paper Appendix B.2 "
                             "bounds r<=4 (a segment executed up to 5x); default here is the "
                             "FAITHFUL value (unlike run_search_dart_math.py's r=2 default, kept "
                             "there only for backward compatibility with an earlier, under-"
                             "faithful search).")
    parser.add_argument("--loglik-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-probs", action="store_true",
                        help="Also log the FULL softmax probability distribution over answer "
                             "choices (not just the argmax) per (query_id, program) to "
                             "<cache-dir>/probs_<model>_mmlu.jsonl. OFF by default -- purely "
                             "additive side log, does not change search behavior or the main "
                             "output file. Requires --cache-dir.")
    parser.add_argument("--masked-batch", action="store_true",
                        help="Opt-in cross-program masked/gathered batching for LogLikReward "
                             "(re_polar/mcts/rewards.py::LogLikReward.masked_batch_call) -- one "
                             "shared forward pass per round across ALL distinct pending programs "
                             "instead of one program at a time. Unlike GenerationReward's "
                             "version, LogLikReward's serial baseline is ALREADY "
                             "use_cache=False (single forward pass, no generation loop) -- so "
                             "this carries none of the cache-vs-no-cache fidelity tradeoff the "
                             "DART-Math version has; expected closer to risk-free. OFF by "
                             "default -- zero behavior change when unset.")
    parser.add_argument("--masked-batch-max-group-size", type=int, default=None,
                        help="Caps peak per-forward-call memory in --masked-batch's greedy "
                             "largest-group-first scheduling (re_polar/mcts/rewards.py) -- added "
                             "after an OOM on a memory-constrained GPU (a single group grew "
                             "large enough to exceed budget). Splits an oversized group into "
                             "sequential sub-chunks of at most this many rows instead of one "
                             "huge call -- bounds memory WITHOUT reducing --n-per-domain. "
                             "Default None = uncapped (original behavior). No effect without "
                             "--masked-batch.")
    parser.add_argument("--masked-batch-buckets", type=int, default=8,
                        help="Length-bucketing for --masked-batch (re_polar/mcts/rewards.py) -- see "
                             "the DART-Math runner's flag of the same name for the full "
                             "rationale. Default 8. 1 = no bucketing (old behavior). No effect "
                             "without --masked-batch.")
    args = parser.parse_args()
    if args.log_probs and not args.cache_dir:
        raise SystemExit("--log-probs requires --cache-dir (that's where the probs log goes)")
    if args.ucb_parent_v and args.global_selection:
        raise SystemExit("--ucb-parent-v only affects the hierarchical mode's descent -- "
                         "--global-selection already always uses global V via its own "
                         "_global_ucb, so this flag would silently have no effect. Drop "
                         "--global-selection to use --ucb-parent-v, or drop --ucb-parent-v.")

    inputs = _load_inputs(Path(args.data_dir), args.domains, args.n_per_domain,
                          args.seed, split=args.split)

    cfg = MODEL_REGISTRY[args.model]

    from re_polar.core import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import LogLikReward

    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    num_layers = engine.num_layers
    probs_log = (str(Path(args.cache_dir) / f"probs_{args.model}_mmlu.jsonl")
                if args.log_probs else None)
    reward_fn = LogLikReward(executor, batch_size=args.loglik_batch_size,
                             no_think=cfg.get("no_think", True), probs_log_path=probs_log,
                             masked_batch_max_group_size=args.masked_batch_max_group_size,
                             masked_batch_buckets=args.masked_batch_buckets)
    masked_batch_reward_fn = reward_fn if args.masked_batch else None

    cache_path = Path(args.cache_dir) / f"{args.model}_mmlu.jsonl" if args.cache_dir else None
    runner = MCTSRunner(inputs, num_layers=num_layers, reward_fn=reward_fn,
                        budget=args.budget, c=args.c, lam=args.lam, seed=args.seed,
                        cache=EvalCache(cache_path), max_repeat_times=args.max_repeat_times,
                        epsilon=args.epsilon, masked_batch_reward_fn=masked_batch_reward_fn,
                        global_selection=args.global_selection,
                        ucb_global_v=not args.ucb_parent_v,
                        alpha=args.alpha, beta=args.beta)

    t0 = time.time()
    trees = runner.run()
    elapsed = time.time() - t0

    samples, solved, shorter = [], 0, 0
    per_domain = {d: {"n": 0, "solved": 0, "shorter": 0} for d in args.domains}
    for inp in inputs:
        tree = trees[inp["query_id"]]
        valid = tree.valid_paths()
        samples.append(sample_record(inp, valid, tree.invalid_paths(),
                                     runner.initial_metric[inp["query_id"]]))
        dstats = per_domain[inp["category"]]
        dstats["n"] += 1
        if valid:
            solved += 1
            dstats["solved"] += 1
            is_shorter = any(len(p) < num_layers for p in valid)
            shorter += is_shorter
            dstats["shorter"] += is_shorter

    write_merged_samples(Path(args.output_root), cfg["model_id"], "mmlu-pro-domains", samples)
    rate = shorter / solved if solved else 0.0
    per_domain_rate = {d: round(s["shorter"] / s["solved"], 4) if s["solved"] else 0.0
                       for d, s in per_domain.items()}
    stats = {"n_inputs": len(inputs), "solved": solved, "shorter_valid_rate": round(rate, 4),
             "identity_solve_rate": round(sum(runner.initial_metric.values()) / len(inputs), 4),
             "elapsed_s": round(elapsed, 1), "per_domain_shorter_valid_rate": per_domain_rate,
             "max_repeat_times": args.max_repeat_times}
    print(json.dumps(stats))
    print(f"PILOT GATE (>=0.60 shorter-valid among solved, same threshold as the DART-Math "
          f"runner): {rate:.3f} -> {'PASS' if rate >= 0.60 else 'FAIL'}")
    if probs_log:
        print(f"Choice-probability log written to {probs_log}")


if __name__ == "__main__":
    main()
