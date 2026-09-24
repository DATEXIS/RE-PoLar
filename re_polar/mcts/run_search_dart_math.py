"""MCTS program discovery over one DART-Math difficulty split.

Pilot run (a short budget/subset to sanity-check the search):
  python -m re_polar.mcts.run_search_dart_math --difficulty 1 --n-inputs 50 \
      --budget 100 --data-dir ./data/dart_math \
      --output-root ./data/mcts --cache-dir ./data/mcts_cache

Full run: omit --n-inputs (all 1250 train inputs), shard by --difficulty.
Output uses PoLar's own merged_mcts_samples.json schema
(re_polar/datasets/schemas.py), so it feeds the router trainer directly.
"""

import argparse
import json
import time
from pathlib import Path

# NOTE: keep this module's top level TORCH-FREE. The grader's spawn workers
# re-import __main__ (this file); the heavy imports (torch via re_polar.core/
# re_polar.core) live inside main() so those workers stay lightweight. re_polar.models
# is torch-free.
from re_polar.models import MODEL_REGISTRY


def main():
    # torch-free imports (these are all --offline-replay needs; the model-side
    # imports live in the non-offline branch below so a replay loads no torch).
    from re_polar.datasets.schemas import sample_record, write_merged_samples
    from re_polar.mcts.scheduler import EvalCache, MCTSRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_8b", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--difficulty", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "trainval", "traintest", "all"],
        help="traintest = train then test, one ordered file -- full search "
        "runs use train+test, NOT val (val is held out for the router/"
        "other downstream uses, not spent as extra MCTS search budget). "
        "trainval = train then val (PoLar: trainer slices [0:1250]=train, "
        "[1250:1500]=val). all = train then val then test, one ordered file "
        "covering the COMPLETE difficulty tier -- for comparing discovered "
        "program lengths against PoLar's own reported numbers, which are "
        "computed over their full search population, not one held-out "
        "slice.",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--n-inputs", type=int, default=None, help="pilot subset; default all")
    parser.add_argument("--budget", type=int, default=100, help="MCTS simulations per input")
    parser.add_argument("--c", type=float, default=1.414, help="UCB exploration constant")
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Progressive-widening scale: a node with `visits` visits may hold "
        "up to ceil(alpha * visits**beta) children before selection is "
        "forced to descend via UCB instead of expanding a new one "
        "(re_polar/mcts/search.py's _widening_limit). Not mentioned in either "
        "paper; necessary regardless because the action space (~690 actions "
        "at D=36) dwarfs any realistic budget. No single literature-"
        "canonical value for this scale constant -- domain-tuned everywhere "
        "it's used. Passing a very large value (e.g. far beyond the actual "
        "per-node action count) makes the cap unreachable within any "
        "realistic budget, i.e. effectively disables widening without a "
        "separate flag.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Progressive-widening exponent (see --alpha). 0.5 (O(sqrt(visits)) "
        "growth) is the literature-standard choice (Couetoux et al. 2011, "
        "'Continuous Upper Confidence Trees', and corroborating summaries) "
        "-- this IS the literature default, unlike --alpha's scale "
        "constant.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=5.0,
        help="length penalty weight. PRE "
        "paper Appendix B (arXiv:2507.07996) states 5.0; pass --lam 0.5 to "
        "reproduce an earlier default (0.5, 10x smaller).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Probability of force-expanding a random untried edit instead of "
        "UCB-selecting an existing child during descent. PRE paper Appendix "
        "B: 'selects a random unexplored child node with probability 0.1 "
        "instead of the one with the highest UCB score' (note: our force-"
        "expand mechanism isn't a literal rendering of that -- see "
        "propose()'s legacy-branch comment in re_polar/mcts/search.py). Default "
        "0.0: isolates widening+global-V from epsilon as a separate axis; "
        "pass --epsilon 0.1 to match the paper's stated value.",
    )
    parser.add_argument(
        "--max-repeat-times",
        type=int,
        default=2,
        help="Max execution count for a REPEAT segment. Paper Appendix B.2 bounds "
        "r<=4; our reconstruction defaulted to 2 (a single 2x loop). `times` "
        "here counts TOTAL executions, so r<=4 means pass 5 (not 4) to match "
        "the paper's search space -- see re_polar/mcts/search.py's "
        "DEFAULT_MAX_REPEAT_TIMES comment. >2 needs a FRESH cache (new paths) "
        "and is NOT valid with --offline-replay.",
    )
    parser.add_argument(
        "--log-answers",
        action="store_true",
        help="Persist every GENERATED ANSWER plus the batch composition/order "
        "it was generated in, to <cache-dir>/answers_<model>_diff<N>.jsonl "
        "(+ a .questions.jsonl sidecar mapping question-hash -> question "
        "text). See re_polar/mcts/textlog.py: merged_mcts_samples.json records "
        "only the pass/fail OUTCOME, and re-deriving a label by regenerating "
        "mismatches it 13.3%% of the time in isolation / 5.0%% at matched "
        "batch composition (bf16 batch-shape non-associativity) -- so "
        "without this a label cannot be audited after the fact. Purely "
        "additive side log: does not change search behavior, rewards, or "
        "the main output file. Requires --cache-dir. OFF by default.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)  # paper protocol (Appendix D.4)
    parser.add_argument(
        "--prompt-style",
        default="paper_minimal_fewshot",
        help="GenerationReward prompt style (re_polar/mcts/rewards.py PROMPT_STYLES). "
        "Default 'paper_minimal_fewshot' = one trivial demo "
        "(1+1->\\boxed{2}) added ahead of the real question, same 50-token "
        "budget, raw completion (no chat template) -- the winner of a "
        "prompt-format investigation into the paper's own D.4 wording (see "
        "re_polar/mcts/rewards.py's PROMPT_STYLES docstring for the full "
        "sweep). 'raw' = the paper's exact Appendix D.4 prompt, no fewshot. "
        "Any --offline-replay of a cache built under a DIFFERENT prompt "
        "style will hit cache misses the replay reward_fn treats as a hard "
        "error -- the prompt is baked into what got cached, not a "
        "replay-time parameter.",
    )
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offline-replay",
        action="store_true",
        help="Reconstruct supervision at --budget from an EXISTING cache with "
        "no model/GPU. Budget only gates termination, so the search "
        "trajectory for rounds 1..budget is identical to a longer/running "
        "run's; the reward fn asserts every (query_id, path) is already "
        "cached (holds for any budget <= the cache's search depth). Lets "
        "you snapshot a shorter-budget result from a running search.",
    )
    parser.add_argument(
        "--replicas",
        default="1",
        help="Reward-fn REPLICA POOL size for cross-sample MCTS parallelism "
        "(fills the GPU when program batches are small: late rounds / "
        "the per-tree-seed default). 'auto' = size to the GPU memory budget "
        "(re_polar.mcts.replica_pool, capped by --max-replicas); an integer N "
        "= exactly N model copies on one GPU. Default 1 = single model, "
        "bit-identical to the pre-pool scheduler. Result-preserving (each "
        "(program,input) scored by one equivalent replica). Not valid with "
        "--offline-replay (no model).",
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=4,
        help="Upper bound for --replicas auto (GPU memory-budget cap).",
    )
    parser.add_argument(
        "--replica-seq-len",
        type=int,
        default=1024,
        help="Assumed prompt+generation length for --replicas auto activation "
        "budgeting; larger => fewer, safer replicas.",
    )
    parser.add_argument(
        "--masked-batch",
        action="store_true",
        help="Opt-in cross-program masked/gathered batching (re_polar/mcts/"
        "scheduler.py module docstring 'MASKED-BATCH SCHEDULER'): a round's "
        "distinct pending programs share ONE forward pass instead of one "
        "reward-fn call each. Off by default. Measured tradeoff, NOT "
        "result-preserving -- ~8.6-8.8%% verdict-flip-rate cost vs. the "
        "cached serial path on real search-derived programs. Incompatible "
        "with --replicas != 1 (masked-batch takes priority in "
        "_evaluate_jobs whenever both would apply; pass --replicas 1 "
        "explicitly to avoid confusion).",
    )
    parser.add_argument(
        "--masked-batch-max-group-size",
        type=int,
        default=None,
        help="Caps peak per-forward-call memory in --masked-batch's greedy "
        "largest-group-first scheduling (re_polar/mcts/rewards.py) -- added "
        "after an OOM on a memory-constrained GPU for the mmlu_pro variant "
        "(a single group grew large enough to exceed budget). Splits an "
        "oversized group into sequential sub-chunks of at most this many "
        "rows instead of one huge call -- bounds memory WITHOUT reducing "
        "--n-inputs. Default None = uncapped (original behavior). No effect "
        "without --masked-batch.",
    )
    parser.add_argument(
        "--masked-batch-buckets",
        type=int,
        default=8,
        help="Length-bucketing for --masked-batch (re_polar/mcts/rewards.py): splits "
        "a round into this many length-sorted buckets instead of padding "
        "every row to the round's single longest prompt. Fixes the real "
        "driver of the longest-difficulty tier's OOMs (worse than the group "
        "cap ever was): one long outlier forces ALL rows to its length, and "
        "use_cache=False recomputes that padding waste at every decode step. "
        "Default 8: the length distribution is heavy-tailed, so 8 roughly-"
        "equal-count buckets isolate the outliers from the bulk without "
        "excessive per-bucket setup overhead multiplication. 1 = no "
        "bucketing (old behavior). No effect without --masked-batch.",
    )
    parser.add_argument(
        "--global-selection",
        action="store_true",
        help="Opt IN to global_selection (re_polar/mcts/search.py's 'GLOBAL "
        "SELECTION MODE'): flat tree-wide UCB argmax + no progressive "
        "widening + no transposition/DAG collapsing, one literal reading of "
        "CoLa/PoLar's own Algorithm 1. Piloted as the default, then reverted "
        "back to the hierarchical+widening+DAG search (this flag OFF) after "
        "real-data evidence showed this mode's median tree-evaluation depth "
        "is 10 edits, prompt-independent -- not the 'modest programs' "
        "PoLar's own ICML Finding 4 describes. Any --offline-replay of a "
        "cache built under this mode needs this flag.",
    )
    parser.add_argument(
        "--ucb-parent-v",
        action="store_true",
        help="Use the selected node's own parent's visit count in UCB's explore "
        "term (standard textbook UCT/UCB1, Kocsis & Szepesvari 2006) instead "
        "of the paper's literal global V (total simulations across the WHOLE "
        "tree so far), for the hierarchical (non---global-selection) mode's "
        "descent. Both PRE (arXiv:2507.07996) and ICML Appendix B.3 state 'V "
        "is the total number of simulations' verbatim -- global V is the "
        "default (staying close to the papers' literal formula), superseding "
        "an earlier decision to keep local parent-visit-count. Pass this "
        "flag for that textbook-UCT reading instead. Only affects the "
        "hierarchical mode: --global-selection already always uses global V "
        "via its own _global_ucb.",
    )
    parser.add_argument(
        "--shared-seed",
        action="store_true",
        help="Share --seed across all trees (an earlier default) instead of "
        "giving every tree its own deterministic seed derived from "
        "query_id. Confirmed confound: a shared-seed pilot's top program "
        "covered 51%% of diff1 inputs vs. per-tree-seed's 4%% -- a "
        "manufactured menu, not genuine per-input diversity. Default is "
        "now per-tree-seed (this flag OFF); pass --shared-seed to opt "
        "back into the old, more-batchable-but-confounded mode. "
        "Per-tree-seed costs substantially more reward-fn calls than "
        "shared-seed for the same budget (program-major batching collapses "
        "by design, scheduler.py module docstring) -- the reward-fn replica "
        "pool (--replicas) does not fully recoup this, so budget for it.",
    )
    args = parser.parse_args()
    if args.log_answers and not args.cache_dir:
        raise SystemExit("--log-answers requires --cache-dir (that's where the answer log goes)")
    per_tree_seed = not args.shared_seed

    ddir = Path(args.data_dir) / f"diff{args.difficulty}"
    if args.split == "trainval":
        inputs = json.load(open(ddir / "train.json")) + json.load(open(ddir / "val.json"))
    elif args.split == "traintest":
        inputs = json.load(open(ddir / "train.json")) + json.load(open(ddir / "test.json"))
    elif args.split == "all":
        inputs = (
            json.load(open(ddir / "train.json"))
            + json.load(open(ddir / "val.json"))
            + json.load(open(ddir / "test.json"))
        )
    else:
        inputs = json.load(open(ddir / f"{args.split}.json"))
    if args.n_inputs:
        inputs = inputs[: args.n_inputs]

    cfg = MODEL_REGISTRY[args.model]

    cache_path = (
        Path(args.cache_dir) / f"{args.model}_diff{args.difficulty}.jsonl"
        if args.cache_dir
        else None
    )
    reward_fns = None  # replica pool (set in the model branch when --replicas != 1)
    if args.offline_replay:
        # No model, no GPU: rebuild the tree state at --budget purely from the
        # already-evaluated (query_id, path) cache. reward_fn must never fire, a
        # miss means the cache doesn't cover this budget, so raise loudly instead
        # of silently generating a different (contaminated) result.
        if not (cache_path and cache_path.exists()):
            raise SystemExit(f"--offline-replay needs an existing cache at {cache_path}")
        num_layers = cfg["num_layers"]

        def reward_fn(program, questions, gt_answers):
            raise RuntimeError(
                f"cache MISS in --offline-replay (budget={args.budget}) on path "
                f"{program.to_layer_path()} for {len(questions)} input(s): the cache does "
                f"not cover this budget. Lower --budget or let the search run further."
            )

    else:
        from re_polar.core import LayerEngine
        from re_polar.core import ProgramExecutor
        from re_polar.mcts.rewards import GenerationReward

        # log grader blow-ups (pathological answers that would OOM the job) next to
        # the cache so they survive the run and can be inspected afterward. Shared
        # across replicas: _log_fail appends one small JSON line (atomic under
        # PIPE_BUF with O_APPEND), and it is diagnostics only, never a reward.
        fail_log = (
            str(Path(args.cache_dir) / f"grader_fails_{args.model}_diff{args.difficulty}.jsonl")
            if args.cache_dir
            else None
        )
        # generated-answer + batch-composition log (re_polar/mcts/textlog.py). Shared
        # across replicas like fail_log: each replica gets its own TextLog on the
        # same path, but rows are written one atomic O_APPEND line at a time and
        # carry their own random batch id, so the file stays well-formed.
        answer_log = (
            str(Path(args.cache_dir) / f"answers_{args.model}_diff{args.difficulty}.jsonl")
            if args.log_answers
            else None
        )
        reward_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
            difficulty=args.difficulty,
            fail_log_path=fail_log,
            masked_batch_max_group_size=args.masked_batch_max_group_size,
            masked_batch_buckets=args.masked_batch_buckets,
            text_log_path=answer_log,
            prompt_style=args.prompt_style,
        )
        replicas_arg = str(args.replicas).strip().lower()
        if replicas_arg == "1":
            engine = LayerEngine(
                cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True)
            )
            reward_fn = GenerationReward(ProgramExecutor(engine), **reward_kwargs)
            num_layers = engine.num_layers
        else:
            from re_polar.mcts.replica_pool import (
                build_budgeted_replicas,
                build_generation_replicas,
            )

            trc = cfg.get("trust_remote_code", True)
            if replicas_arg == "auto":
                reward_fns, n, diag = build_budgeted_replicas(
                    cfg["model_id"],
                    max_replicas=args.max_replicas,
                    batch_size=args.gen_batch_size,
                    seq_len=args.replica_seq_len,
                    trust_remote_code=trc,
                    reward_kwargs=reward_kwargs,
                )
                print(f"replica pool: auto-sized to {n} replica(s), {diag}")
            else:
                reward_fns = build_generation_replicas(
                    cfg["model_id"],
                    int(replicas_arg),
                    trust_remote_code=trc,
                    reward_kwargs=reward_kwargs,
                )
                print(f"replica pool: {len(reward_fns)} explicit replica(s)")
            reward_fn = reward_fns[0]
            num_layers = reward_fns[0].executor.engine.num_layers

    masked_batch_reward_fn = reward_fn if (not args.offline_replay and args.masked_batch) else None
    if args.masked_batch and args.offline_replay:
        raise SystemExit("--masked-batch has no model in --offline-replay, nothing to batch")

    if args.offline_replay and str(args.replicas).strip().lower() != "1":
        raise SystemExit(
            "--offline-replay has no model, so --replicas must be 1 (the replica "
            "pool only parallelizes real GPU generation)."
        )
    if args.offline_replay and args.max_repeat_times != 2:
        raise SystemExit(
            "--offline-replay requires --max-repeat-times 2 (the cache only "
            "covers the paper's 2x loops; deeper loops need a fresh search)."
        )
    if args.offline_replay and per_tree_seed:
        raise SystemExit(
            "--offline-replay replays an EXISTING shared-seed trajectory; the "
            "default (per-tree-seed) proposes a different one per tree, so it "
            "will hit cache misses the offline-replay reward_fn treats as a hard "
            "error. Pass --shared-seed alongside --offline-replay to replay a "
            "shared-seed cache, or run per-tree-seed against a fresh cache (no "
            "--offline-replay) instead."
        )
    if args.offline_replay and args.epsilon != 0.0:
        raise SystemExit(
            "--offline-replay replays the EXISTING epsilon=0.0 trajectory; "
            "--epsilon > 0 proposes a different one, so it will hit cache misses "
            "the offline-replay reward_fn treats as a hard error. Run --epsilon "
            "against a fresh cache (no --offline-replay) instead."
        )
    if args.offline_replay and args.global_selection:
        raise SystemExit(
            "--offline-replay only supports the hierarchical mode's trajectory -- "
            "--global-selection proposes an entirely different one (flat tree-wide "
            "argmax, no widening), so it will hit cache misses the offline-replay "
            "reward_fn treats as a hard error. Run --global-selection against a "
            "fresh cache (no --offline-replay) instead."
        )
    if args.ucb_parent_v and args.global_selection:
        raise SystemExit(
            "--ucb-parent-v only affects the hierarchical mode's descent "
            "(re_polar/mcts/search.py's _ucb) -- --global-selection already always "
            "uses global V via its own _global_ucb, so this flag would silently "
            "have no effect. Drop --global-selection to use --ucb-parent-v, or "
            "drop --ucb-parent-v."
        )

    runner = MCTSRunner(
        inputs,
        num_layers=num_layers,
        reward_fn=reward_fn,
        budget=args.budget,
        c=args.c,
        lam=args.lam,
        seed=args.seed,
        cache=EvalCache(cache_path),
        max_repeat_times=args.max_repeat_times,
        per_tree_seed=per_tree_seed,
        reward_fns=reward_fns,
        epsilon=args.epsilon,
        masked_batch_reward_fn=masked_batch_reward_fn,
        global_selection=args.global_selection,
        ucb_global_v=not args.ucb_parent_v,
        alpha=args.alpha,
        beta=args.beta,
    )

    t0 = time.time()
    trees = runner.run()
    elapsed = time.time() - t0

    samples, solved, shorter = [], 0, 0
    for inp in inputs:
        tree = trees[inp["query_id"]]
        valid = tree.valid_paths()
        samples.append(
            sample_record(
                inp,
                valid,
                tree.invalid_paths(),
                runner.initial_metric[inp["query_id"]],
                trajectory=tree.trajectory,
            )
        )
        if valid:
            solved += 1
            shorter += any(len(p) < num_layers for p in valid)

    write_merged_samples(
        Path(args.output_root), cfg["model_id"], f"dart-math-diff-{args.difficulty}", samples
    )
    rate = shorter / solved if solved else 0.0
    stats = {
        "n_inputs": len(inputs),
        "solved": solved,
        "shorter_valid_rate": round(rate, 4),
        "identity_solve_rate": round(sum(runner.initial_metric.values()) / len(inputs), 4),
        "elapsed_s": round(elapsed, 1),
        "per_tree_seed": per_tree_seed,
        "global_selection": args.global_selection,
        "ucb_global_v": not args.ucb_parent_v,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "beta": args.beta,
        "prompt_style": args.prompt_style,
    }
    print(json.dumps(stats))
    print(
        f"PILOT GATE (>=0.60 shorter-valid among solved): "
        f"{rate:.3f} -> {'PASS' if rate >= 0.60 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
