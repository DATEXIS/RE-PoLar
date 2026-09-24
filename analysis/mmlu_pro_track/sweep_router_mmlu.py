"""Router hyperparameter SWEEP for mmlu_pro_domains -- LogLikReward analog
of `re_polar/router/train.py`'s DART-Math sweep script.

Differs from the DART-Math sweep in three ways, all forced by mmlu's shape
(see `re_polar/mcts/rewards.py`'s LogLikReward docstring; `re_polar/router/mmlu_bridge.py`):
  - reward = LogLikReward (one log-likelihood forward pass) instead of
    GenerationReward -- no generation loop, no sympy grading subprocess pool.
  - questions are `{"question","options","category"}` dicts + an INT
    answer_index, not a plain string + string gt_ans. `re_polar/router/train.py`'s
    examples/encoder need a hashable STRING, so MCTS supervision is loaded via
    `re_polar.router.mmlu_bridge.load_mmlu_supervision` (flattens the dict into the
    router's encoder text; keeps the original dict at `question_struct` for
    grading) -- `build_examples`/`encode_examples`/`train` themselves are
    reused UNCHANGED, same as `re_polar.router.infer`'s `predict_programs_topk`/
    `grade_router_topk` (both reward-fn-agnostic; imported, not reimplemented).
  - grouping is `--group-by-domain` (mmlu category), not `--per-difficulty`:
    domains are topic tiers, not difficulty tiers, so the natural default is
    ONE POOLED router over every domain in `--samples`.

TWO DISTINCT holdouts are in play, matching `re_polar/datasets/mmlu_pro_domains.py`'s
`--val-frac` split design:
  1. `--holdout-frac` slices EACH `--samples` file's OWN rows (generation-free,
     same `split_samples_train_val` mechanic as the DART sweep) -- drives the
     (lr, batch, epochs) GRID SELECTION via val_loss, cheaply, with no LLM calls.
  2. `--val-data` is `val.json` (or `test.json`, see below) from the
     mmlu_pro_domains `--val-frac` split -- rows the MCTS search NEVER touched
     (disjoint from every `--samples` file's source pool). Used for the
     ONE-TIME winner measurement (online, via LogLikReward) after grid
     selection -- an honest held-out read, not a just-happened-to-be-excluded-
     from-training slice of the same search pool.

Real bug found and fixed during this project's own use of this script,
worth keeping documented since it changes which output is trustworthy: the
MCTS search that produces `--samples` was run with all splits unioned
(train+val+test), so the supervision pool silently contained real program
labels for `mmlu-test-*` query_ids too -- training on it and then
"held-out" evaluating on test.json (`infer_router_mmlu.py`) was actually
evaluating on (mostly) the router's own training data. The router@1==
identity@1 collapse still held even so (arguably a STRONGER result --
collapsed even when trained directly on the eval questions), but any
router@5-clears-identity number measured before this fix is not trustworthy
as a generalization claim. Fix: `drop_test_split()` below, called
unconditionally right after loading `--samples` supervision, so training
supervision is genuinely train+val only and test.json stays honestly held
out for `infer_router_mmlu.py`'s separate eval.

The frozen encoder + target LLM are loaded ONCE and shared across groups/
configs. Top level is stdlib-only so any future spawn-based grading stays
torch-free (mirrors the DART-Math sweep script).

    python -m analysis.mmlu_pro_track.sweep_router_mmlu \\
        --samples merged_mcts_samples.json \\
        --val-data mmlu_pro_domains/test.json \\
        --model qwen3_8b --out router_qwen3_8b_mmlu.pt \\
        --select-metric val_loss --strict-repeat-2x
"""

import argparse
import json
import random
from pathlib import Path


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def drop_test_split(supervision):
    """Filter out mmlu-test-* rows from a load_mmlu_supervision() list (pure
    stdlib, no torch -- see main()'s call site and the module docstring for
    why this exists)."""
    return [s for s in supervision if not s["question_struct"]["query_id"].startswith("mmlu-test-")]


def main(argv=None):
    # heavy imports INSIDE main -> mirrors the DART-Math sweep script's torch-free top level
    import torch
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import LogLikReward
    from re_polar.core import Program
    from re_polar.router.model import PolarRouter
    from re_polar.router.train import (
        build_examples,
        encode_examples,
        save_checkpoint,
        split_samples_train_val,
        train,
        _resolve_device,
    )
    from re_polar.router.infer import grade_router_topk, predict_programs_topk
    from re_polar.router.mmlu_bridge import format_mmlu_question, load_mmlu_supervision
    from transformers import AutoModel

    p = argparse.ArgumentParser(description="mmlu_pro_domains router sweep (val-loss selection).")
    p.add_argument(
        "--samples",
        required=True,
        nargs="+",
        help="merged_mcts_samples.json from the mmlu MCTS run",
    )
    p.add_argument(
        "--val-data",
        required=True,
        help="path to a re_polar.datasets.mmlu_pro_domains split file genuinely disjoint from "
        "training supervision -- used ONLY for the online winner measurement, "
        "never for grid selection. MUST be test.json, not val.json: --samples "
        "drops mmlu-test-* rows unconditionally (see drop_test_split above), so "
        "val.json rows are folded INTO training (train+val) and are no longer "
        "held out -- pointing --val-data at val.json here would silently measure "
        "the router on its own training data again.",
    )
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument("--out", required=True, help="checkpoint path (per-domain appends _<domain>)")
    p.add_argument(
        "--group-by-domain",
        action="store_true",
        help="train ONE router PER mmlu domain (category); default pooled (1 router "
        "over every domain present in --samples)",
    )
    p.add_argument(
        "--select-metric",
        choices=["val_loss", "val_cache_acc_at1"],
        default="val_loss",
        help="config/checkpoint selection metric (mirrors the DART sweep's flag). "
        "val_loss (default) is generation-free but minimised by collapsing to "
        "predict the majority op everywhere. val_cache_acc_at1 is ALSO "
        "generation-free (a lookup: does the router's decoded top-1 program land "
        "in this question's MCTS-VERIFIED final_valid_transitions set?) but does "
        "not reward collapse -- recommended.",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.1,
        help="fraction of EACH --samples file's rows held out for generation-free "
        "val-loss GRID SELECTION (split_samples_train_val)",
    )
    p.add_argument(
        "--lrs", default="1e-4,3e-4,5e-4,8e-4,1e-3,3e-3", help="comma-sep learning rates"
    )
    p.add_argument("--epochs-list", default="3,10", help="comma-sep epoch counts")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--batch-sizes",
        default=None,
        help="comma-sep batch sizes to sweep; overrides --batch-size when set",
    )
    p.add_argument("--val-topk", type=int, default=5, help="k for the winner's pass@k MEASUREMENT")
    p.add_argument(
        "--val-eval-limit",
        type=int,
        default=250,
        help="cap on val.json rows graded per group for the WINNER measurement "
        "(applied AFTER shuffling, per-domain when --group-by-domain)",
    )
    p.add_argument("--max-paths-per-sample", type=int, default=50)
    p.add_argument(
        "--label-mode",
        choices=["multi", "shortest"],
        default="multi",
        help="multi (default): one example per valid path (PoLar labelling, "
        "original_path_weight/drop_original_path apply). shortest: collapse each "
        "sample to its single SHORTEST valid path -- an ablation shown to collapse "
        "the router to identity before multi-path labelling was adopted; see "
        "re_polar/router/train.py's build_examples docstring. original-path-weight/"
        "drop-original-path are ignored in this mode.",
    )
    p.add_argument("--original-path-weight", type=float, default=0.30)
    p.add_argument(
        "--drop-original-path",
        action="store_true",
        help="DROP the identity path from targets (disables reweight). Off-recipe test lever.",
    )
    p.add_argument(
        "--segment-cap",
        choices=["all", "edit-only"],
        default="all",
        help="all (default, unchanged): MAX_SEGMENT_LEN caps every segment, including KEEP "
        "runs. edit-only (ablation): only SKIP/REPEAT stay capped; each maximal KEEP "
        "run merges into ONE segment however long. See the DART-Math sweep script's "
        "--segment-cap help / re_polar/router/train.py program_from_layer_path(cap_keep=...) "
        "docstring.",
    )
    p.add_argument(
        "--strict-repeat-2x",
        action="store_true",
        help="Reject (drop, whole-path) any valid path with a REPEAT times != 2, matching "
        "PoLar's own parser exactly. See the DART-Math sweep script's --strict-repeat-2x help.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="DR.LLM-style focal loss gamma for the op-head CE (0.0 = plain CE, off by "
        "default). Mirrors the DART-Math sweep's --focal-gamma exactly -- same "
        "DR.LLM class-balanced 'effective number of samples' alpha (beta=0.999) "
        "computed from this group's own TRAIN op-label counts.",
    )
    p.add_argument(
        "--lenpref-betas",
        default="0.0",
        help="comma-sep beta values to sweep for PoLar's OWN 'polar_lenpref' policy_mode "
        "(exp(-beta*path_len) length-preference reweighting), added as an ADDITIONAL "
        "axis to the existing lr x batch x epochs grid. Mirrors the DART-Math sweep's "
        "--lenpref-betas exactly. beta=0.0 runs under policy_mode='polar' (unchanged "
        "baseline); beta>0.0 runs under policy_mode='polar_lenpref'.",
    )
    p.add_argument(
        "--per-sample-weight-normalize",
        type=lambda s: s.lower() not in ("0", "false", "no"),
        default=True,
        help="See the DART-Math sweep script's --per-sample-weight-normalize help. "
        "Default True, ablatable with --per-sample-weight-normalize=false.",
    )
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--loglik-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    epochs_list = [int(x) for x in args.epochs_list.split(",") if x.strip()]
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",") if x.strip()]
        if args.batch_sizes
        else [args.batch_size]
    )
    lenpref_betas = [float(x) for x in args.lenpref_betas.split(",") if x.strip()]
    device = _resolve_device(None)
    cfg = MODEL_REGISTRY[args.model]
    D = cfg["num_layers"]
    no_think = cfg.get("no_think", True)

    build_kwargs = dict(
        max_paths_per_sample=args.max_paths_per_sample,
        per_sample_weight_normalize=args.per_sample_weight_normalize,
        reweight_original_path=not args.drop_original_path,
        original_path_weight=args.original_path_weight,
        drop_original_path=args.drop_original_path,
        shortest_only=(args.label_mode == "shortest"),
        cap_keep=(args.segment_cap == "all"),
        strict_repeat_2x=args.strict_repeat_2x,
        seed=args.seed,
    )

    # ---- true held-out val.json (never MCTS-searched) for the winner measurement ----
    all_val_rows = json.loads(Path(args.val_data).read_text())
    rng = random.Random(args.seed)
    rng.shuffle(all_val_rows)

    def val_slice(domain=None):
        rows = (
            all_val_rows if domain is None else [r for r in all_val_rows if r["category"] == domain]
        )
        return rows[: args.val_eval_limit] if args.val_eval_limit else rows

    # ---- load + flatten MCTS supervision (question dict -> router-encodable string) ----
    supervision = load_mmlu_supervision(args.samples, no_think=no_think)
    print(f"Loaded {len(supervision)} mmlu supervision samples from {len(args.samples)} file(s).")

    # See module docstring: --samples may contain mmlu-test-* rows if the MCTS
    # search unioned all splits -- drop them unconditionally so training (and
    # the grid-selection holdout, drawn from this same filtered pool) only
    # ever sees train+val query_ids, keeping test.json genuinely held out for
    # infer_router_mmlu.py's separate eval.
    n_before = len(supervision)
    supervision = drop_test_split(supervision)
    print(
        f"Dropped {n_before - len(supervision)} mmlu-test-* rows from training supervision "
        f"({len(supervision)} train+val rows remain) -- test.json stays genuinely held out."
    )

    if args.group_by_domain:
        by_domain: dict = {}
        for s in supervision:
            by_domain.setdefault(s["question_struct"]["category"], []).append(s)
        groups = sorted(by_domain.items())
    else:
        groups = [("all", supervision)]
    print(
        f"grouping: {'per-domain (' + str(len(groups)) + ' routers)' if args.group_by_domain else 'pooled (1 router)'}",
        flush=True,
    )
    print(f"label mode: {args.label_mode}  |  segment cap: {args.segment_cap}", flush=True)
    print(
        f"original-path handling: {'DROP' if args.drop_original_path else 'reweight-' + str(args.original_path_weight)}"
        f"{' (IGNORED -- label-mode=shortest)' if args.label_mode == 'shortest' else ''}",
        flush=True,
    )

    # ---- load frozen encoder + target LLM ONCE (shared across groups/configs) ----
    print("Loading frozen encoder...", flush=True)
    encoder = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to(device)

    def fresh_router():
        torch.manual_seed(
            args.seed
        )  # same init across configs -> isolates the hyperparameter effect
        return PolarRouter(num_layers=D, encoder=encoder).to(device)

    print(f"Loading {cfg['model_id']} for online winner grading (LogLikReward)...", flush=True)
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    reward_fn = LogLikReward(
        ProgramExecutor(engine), batch_size=args.loglik_batch_size, no_think=no_think
    )

    group_summaries = []
    for gname, samples in groups:
        print("\n" + "=" * 72, flush=True)
        print(f"GROUP {gname}  ({len(samples)} supervision sample(s))", flush=True)

        train_samples, holdout_samples = split_samples_train_val(samples, args.holdout_frac)
        examples = build_examples(train_samples, D, **build_kwargs)
        val_examples = build_examples(holdout_samples, D, **build_kwargs)
        if not examples or not val_examples:
            print(
                f"  SKIP {gname}: no examples (train {len(examples)}, holdout {len(val_examples)})",
                flush=True,
            )
            continue
        encode_examples(fresh_router(), examples, batch_size=args.encode_batch_size)
        encode_examples(fresh_router(), val_examples, batch_size=args.encode_batch_size)

        # GENERATION-FREE lookup data for val_cache_acc_at1 (re_polar.router.train.
        # evaluate_val_programs) -- built from `holdout_samples` (this group's OWN
        # grid-selection holdout, MCTS-searched, has final_valid_transitions), NOT
        # from val.json (the true held-out set MCTS never touched, used only for
        # the online winner measurement below and has no valid-path cache at all).
        val_program_data = None
        if args.select_metric == "val_cache_acc_at1":
            valid_sets = {}
            for s in holdout_samples:
                q = s.get("question")
                vs = {
                    tuple(int(x) for x in p) for p in (s.get("final_valid_transitions") or []) if p
                }
                if q is not None and vs:
                    valid_sets[q] = vs
            token_hiddens = {}
            for e in val_examples:
                if e.token_hidden is not None and e.question not in token_hiddens:
                    token_hiddens[e.question] = e.token_hidden
            vq_cache = [q for q in token_hiddens if q in valid_sets]
            if vq_cache:
                val_program_data = {
                    "questions": vq_cache,
                    "token_hiddens": token_hiddens,
                    "valid_sets": valid_sets,
                }
            else:
                print(
                    f"  WARNING {gname}: no holdout question has a non-empty "
                    f"final_valid_transitions -- val_cache_acc_at1 will fall back to val_loss "
                    f"for this group.",
                    flush=True,
                )

        # this group's slice of the TRUE held-out val.json (never searched by MCTS)
        group_val_rows = val_slice(gname if args.group_by_domain else None)
        val_question_dicts = [
            {"question": r["question"], "options": r["options"], "category": r["category"]}
            for r in group_val_rows
        ]
        val_question_strs = [format_mmlu_question(q, no_think=no_think) for q in val_question_dicts]
        val_gt = [int(r["answer_index"]) for r in group_val_rows]
        print(
            f"  train examples {len(examples)} | grid-selection holdout {len(val_examples)} | "
            f"winner-eval (val.json) questions {len(val_question_strs)}",
            flush=True,
        )
        if not val_question_strs:
            print(f"  SKIP {gname}: no val.json rows for this group", flush=True)
            continue

        # DR.LLM-exact class-balanced focal weights, computed once per GROUP (data
        # doesn't change across the hparam grid) from this group's own TRAIN
        # op-label counts. Mirrors the DART-Math sweep's --focal-gamma exactly.
        op_class_weights = None
        if args.focal_gamma > 0:
            from collections import Counter

            op_counts = Counter()
            for e in examples:
                for v in e.op_labels[e.op_labels != -100].tolist():
                    op_counts[v] += 1
            beta_ce = 0.999
            counts = [op_counts.get(i, 1) for i in range(3)]
            eff_num = [(1 - beta_ce**c) / (1 - beta_ce) for c in counts]
            op_class_weights = torch.tensor([1.0 / e for e in eff_num], dtype=torch.float32)
            op_class_weights = (op_class_weights / op_class_weights.mean()).to(device)
            print(
                f"  focal_gamma={args.focal_gamma} op_counts={counts} "
                f"class_weights={op_class_weights.tolist()}",
                flush=True,
            )

        # ---- grid (now with an extra lenpref-beta axis, mirrors the DART sweep) ----
        print(
            f"  {'lr':>8} {'batch':>6} {'epochs':>7} {'beta':>6} {'val_loss':>9} {'sel_score':>10} {'best?':>6}",
            flush=True,
        )
        best = None
        # best (lr,batch,epochs) config PER BETA value -- val_loss selection structurally
        # favours beta=0 (lenpref reweighting costs raw val CE almost everywhere), so
        # without this the sweep would never show whether any beta>0 moves nonident at
        # all. See the DART-Math sweep's --lenpref-betas help for the full rationale.
        best_per_beta = {}
        grid = []
        for epochs in epochs_list:
            for batch in batch_sizes:
                for lr in lrs:
                    for beta in lenpref_betas:
                        policy_mode = "polar_lenpref" if beta > 0.0 else "polar"
                        router = fresh_router()
                        inner_select_by = (
                            "val_cache_acc_at1"
                            if (
                                args.select_metric == "val_cache_acc_at1"
                                and val_program_data is not None
                            )
                            else "val_loss"
                        )
                        res = train(
                            router,
                            examples,
                            val_examples=val_examples,
                            epochs=epochs,
                            lr=lr,
                            batch_size=batch,
                            lr_scheduler="cosine",
                            warmup_steps=args.warmup_steps,
                            device=device,
                            seed=args.seed,
                            move_encodings_to_device=True,
                            select_by=inner_select_by,
                            val_program_data=val_program_data,
                            val_topk=args.val_topk,
                            policy_mode=policy_mode,
                            lenpref_beta=beta,
                            op_class_weights=op_class_weights,
                            op_focal_gamma=args.focal_gamma,
                        )
                        cfg_vloss = min(res.val_losses) if res.val_losses else float("inf")
                        if inner_select_by == "val_cache_acc_at1":
                            score = (
                                res.best_metric if res.best_metric is not None else float("-inf")
                            )
                        else:
                            score = -cfg_vloss  # maximise score == minimise loss
                        is_best = (best is None) or (score > best["score"])
                        print(
                            f"  {lr:>8.0e} {batch:>6} {epochs:>7} {beta:>6.3g} {cfg_vloss:>9.4f} {score:>10.4f} {'  <=' if is_best else '':>6}",
                            flush=True,
                        )
                        grid.append(
                            {
                                "lr": lr,
                                "batch": batch,
                                "epochs": epochs,
                                "lenpref_beta": beta,
                                "policy_mode": policy_mode,
                                "val_loss": cfg_vloss,
                                "sel_score": score,
                            }
                        )
                        if is_best:
                            best = {
                                "score": score,
                                "lr": lr,
                                "batch": batch,
                                "epochs": epochs,
                                "lenpref_beta": beta,
                                "policy_mode": policy_mode,
                                "val_loss": cfg_vloss,
                                "router": router,
                            }
                        if beta not in best_per_beta or score > best_per_beta[beta]["score"]:
                            best_per_beta[beta] = {
                                "score": score,
                                "lr": lr,
                                "batch": batch,
                                "epochs": epochs,
                                "lenpref_beta": beta,
                                "policy_mode": policy_mode,
                                "val_loss": cfg_vloss,
                                "router": router,
                            }

        # ---- ONE online eval of the WINNER on the true held-out val.json ----
        winner = best["router"]
        winner.eval()
        base_p1 = _mean(reward_fn(Program.identity(D), val_question_dicts, val_gt))
        tk = predict_programs_topk(
            winner, val_question_strs, k=args.val_topk, batch_size=args.loglik_batch_size
        )
        r_at_k, r_at_1, _ = grade_router_topk(tk, val_question_dicts, val_gt, reward_fn)
        win_p1, win_pk = _mean(r_at_1), _mean(r_at_k)
        nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in tk])

        out_path = Path(args.out)
        if args.group_by_domain:
            out_path = out_path.with_name(
                out_path.stem + f"_{gname.replace(' ', '_')}" + out_path.suffix
            )
        save_checkpoint(
            winner,
            str(out_path),
            meta={
                "sweep": True,
                "select_metric": args.select_metric,
                "label_mode": args.label_mode,
                "segment_cap": args.segment_cap,
                "group": gname,
                "lr": best["lr"],
                "epochs": best["epochs"],
                "batch_size": best["batch"],
                "lenpref_beta": best["lenpref_beta"],
                "policy_mode": best["policy_mode"],
                "focal_gamma": args.focal_gamma,
                "val_loss": best["val_loss"],
                "base_val_p1": base_p1,
                "val_p1": win_p1,
                f"val_p{args.val_topk}": win_pk,
                "nonidentity_rate": nonident,
                "model": args.model,
                "val_data": str(args.val_data),
                "samples": [str(s) for s in args.samples],
            },
        )
        print(
            f"  WINNER {gname}: lr={best['lr']:.0e} batch={best['batch']} epochs={best['epochs']} "
            f"policy_mode={best['policy_mode']} lenpref_beta={best['lenpref_beta']:.3g} val_loss={best['val_loss']:.4f} "
            f"-> base_p1 {base_p1:.4f} | router_p1 {win_p1:.4f} "
            f"(Δ{win_p1 - base_p1:+.4f}) | router_p{args.val_topk} {win_pk:.4f} | "
            f"nonident {nonident:.1%}  -> {out_path}",
            flush=True,
        )
        group_summaries.append(
            {
                "group": gname,
                "lr": best["lr"],
                "batch": best["batch"],
                "epochs": best["epochs"],
                "lenpref_beta": best["lenpref_beta"],
                "policy_mode": best["policy_mode"],
                "val_loss": best["val_loss"],
                "base_val_p1": base_p1,
                "val_p1": win_p1,
                f"val_p{args.val_topk}": win_pk,
                "nonidentity_rate": nonident,
                "n_val_questions": len(val_question_strs),
                "grid": grid,
            }
        )

        # ---- ALSO eval the best config AT EACH BETA (not just the global winner) ----
        # See best_per_beta comment above -- one extra online-scoring pass per beta
        # value so nonident-vs-beta is visible even though val_loss selection
        # structurally favours beta=0.
        beta_curve = []
        for beta in lenpref_betas:
            cfg = best_per_beta[beta]
            if beta == best["lenpref_beta"]:
                b_p1, b_pk, b_nonident = win_p1, win_pk, nonident
            else:
                b_router = cfg["router"]
                b_router.to(device)
                b_router.eval()
                b_tk = predict_programs_topk(
                    b_router, val_question_strs, k=args.val_topk, batch_size=args.loglik_batch_size
                )
                b_r_at_k, b_r_at_1, _ = grade_router_topk(
                    b_tk, val_question_dicts, val_gt, reward_fn
                )
                b_p1, b_pk = _mean(b_r_at_1), _mean(b_r_at_k)
                b_nonident = _mean(
                    [0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in b_tk]
                )
                b_router.to("cpu")  # free GPU memory before the next beta's eval
            beta_curve.append(
                {
                    "lenpref_beta": beta,
                    "policy_mode": cfg["policy_mode"],
                    "lr": cfg["lr"],
                    "batch": cfg["batch"],
                    "epochs": cfg["epochs"],
                    "val_loss": cfg["val_loss"],
                    "val_p1": b_p1,
                    f"val_p{args.val_topk}": b_pk,
                    "nonidentity_rate": b_nonident,
                }
            )
        group_summaries[-1]["beta_curve"] = beta_curve
        print(f"  beta curve {gname} (best lr/bs/ep PER beta, base_p1 {base_p1:.4f}):", flush=True)
        print(
            f"    {'beta':>6} {'val_loss':>9} {'router_p1':>10} {'Δp1':>7} {'nonident':>9}",
            flush=True,
        )
        for c in beta_curve:
            print(
                f"    {c['lenpref_beta']:>6.3g} {c['val_loss']:>9.4f} {c['val_p1']:>10.4f} "
                f"{c['val_p1'] - base_p1:>+7.4f} {c['nonidentity_rate']:>8.1%}",
                flush=True,
            )

    # ---- combined summary ----
    print("\n" + "=" * 72, flush=True)
    print(
        f"{'group':>18} {'lr':>8} {'bs':>5} {'ep':>4} {'beta':>6} {'val_loss':>9} {'base_p1':>8} {'router_p1':>10} "
        f"{'Δp1':>7} {'router_pk':>10} {'nonident':>9}",
        flush=True,
    )
    for s in group_summaries:
        print(
            f"{s['group']:>18} {s['lr']:>8.0e} {s['batch']:>5} {s['epochs']:>4} {s['lenpref_beta']:>6.3g} {s['val_loss']:>9.4f} "
            f"{s['base_val_p1']:>8.4f} {s['val_p1']:>10.4f} {s['val_p1'] - s['base_val_p1']:>+7.4f} "
            f"{s[f'val_p{args.val_topk}']:>10.4f} {s['nonidentity_rate']:>8.1%}",
            flush=True,
        )

    out = {
        "model": args.model,
        "group_by_domain": args.group_by_domain,
        "select_metric": args.select_metric,
        "label_mode": args.label_mode,
        "segment_cap": args.segment_cap,
        "val_topk": args.val_topk,
        "val_data": str(args.val_data),
        "lenpref_betas": lenpref_betas,
        "focal_gamma": args.focal_gamma,
        "groups": group_summaries,
    }
    resj = Path(args.out).with_suffix(".sweep.json")
    resj.write_text(json.dumps(out, indent=2))
    print(f"\nwrote sweep table -> {resj}", flush=True)


if __name__ == "__main__":
    main()
