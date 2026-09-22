"""Router hyperparameter SWEEP, PAPER-FAITHFUL selection (PoLar train.py).

PoLar selects the checkpoint by MIN VALIDATION LOSS (polar/train.py: best_val /
`if vavg < best_val`), NOT by any accuracy/pass@k metric, and it trains ONE router
PER DIFFICULTY (`--target_diff`), not one pooled router over all 5. This script
mirrors both:

  --select-metric val_loss   (default)  select config + checkpoint by min val loss
                                          (generation-free; the paper's actual metric)
  --select-metric val_passk             legacy: select config by online val pass@k
  --per-difficulty                      run an INDEPENDENT sweep per --samples file
                                          (one router per difficulty), else pool all.

For each group (a difficulty, or the pool) and each (lr, epochs) config we train a
FRESH head to `epochs`, restoring its own MIN-val-loss epoch (train(select_by=
"val_loss")). The config with the lowest val loss wins. We then run ONE online eval
of the winner only (base identity vs winner pass@1/@k, nonidentity rate) so we can
still SEE whether the paper-faithful router beats identity, that is a MEASUREMENT,
not the selection criterion.

This is the DART-Math sweep; `analysis/mmlu_pro_track/sweep_router_mmlu.py` is its
LogLikReward analog for the MMLU-Pro track, and mirrors most of these flags
directly.

The frozen encoder + target LLM are loaded ONCE and shared across groups/configs.
Top level is stdlib-only so spawn grading workers stay torch-free.

    python -m analysis.router_recipe_sweep.sweep_router \\
        --samples merged_mcts_samples_diff1.json ... merged_mcts_samples_diff5.json \\
        --model qwen3_8b --out router_qwen3_8b.pt \\
        --select-metric val_loss --strict-repeat-2x --drop-original-path
"""
import argparse
import json
import random
from pathlib import Path


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main(argv=None):
    # heavy imports INSIDE main -> spawn grading workers stay torch-free
    import torch
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program
    from re_polar.router.model import PolarRouter
    from re_polar.router.train import (
        build_examples, encode_examples, load_supervision, save_checkpoint,
        split_samples_train_val, train, _resolve_device,
    )
    from re_polar.router.infer import grade_router_topk, predict_programs_topk
    from transformers import AutoModel

    p = argparse.ArgumentParser(description="Paper-faithful router sweep (val-loss selection, per-difficulty).")
    p.add_argument("--samples", required=True, nargs="+", help="merged_mcts_samples.json (one per diff)")
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument("--out", required=True, help="checkpoint path (per-difficulty appends _diffK)")
    p.add_argument("--select-metric", choices=["val_loss", "val_passk", "val_cache_acc_at1"],
                   default="val_loss",
                   help="config/checkpoint selection metric. val_loss = PoLar's actual metric "
                        "(default; generation-free, but minimised by collapsing to predict the "
                        "majority op everywhere). val_passk = online pass@k re-GENERATION per "
                        "config -- exactly the batch-composition/bf16-kernel noise this project "
                        "has documented elsewhere (a real, measured 13-17%% verdict flip rate "
                        "re-executing the SAME recorded-valid program in isolation), used here as "
                        "the selection signal itself. val_cache_acc_at1 = GENERATION-FREE lookup: "
                        "does the router's top-1 decoded program land in this question's "
                        "MCTS-VERIFIED final_valid_transitions set (re_polar.router.train."
                        "evaluate_val_programs)? No LLM forward pass, so it cannot be contaminated "
                        "by generation noise, and unlike val_loss it does not reward collapse -- "
                        "recommended over both of the others.")
    p.add_argument("--per-difficulty", action="store_true",
                   help="train ONE router per --samples file (PoLar --target_diff), else pool all.")
    p.add_argument("--lrs", default="3e-4,5e-4,1e-3,3e-3", help="comma-sep learning rates to sweep")
    p.add_argument("--epochs-list", default="3,10", help="comma-sep epoch counts to sweep")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-sizes", default=None,
                   help="comma-sep batch sizes to SWEEP as part of the grid (paper D.2: 32,128,256). "
                        "Overrides --batch-size when set.")
    p.add_argument("--val-topk", type=int, default=5, help="k for the winner's pass@k MEASUREMENT")
    p.add_argument("--train-per-diff", type=int, default=1250,
                   help="PoLar positional split: first N samples/diff = train, rest = val (250)")
    p.add_argument("--val-eval-limit", type=int, default=250,
                   help="online-grade at most N val questions for the WINNER measurement (None=all)")
    p.add_argument("--max-paths-per-sample", type=int, default=50)
    p.add_argument("--label-mode", choices=["multi", "shortest"], default="multi",
                    help="multi (default): one example per valid path (PoLar labelling, "
                         "original_path_weight/drop_original_path apply). shortest: collapse each "
                         "sample to its single SHORTEST valid path -- an ablation that collapsed the "
                         "router to identity before multi-path labelling was adopted; see "
                         "re_polar/router/train.py build_examples docstring. original-path-weight/"
                         "drop-original-path are ignored in this mode.")
    p.add_argument("--original-path-weight", type=float, default=0.30)
    p.add_argument("--original-path-weights", default=None,
                   help="comma-sep original-path-weight values to sweep as an ADDITIONAL axis, "
                        "following the identity-collapse finding (EVERY variant/model/difficulty "
                        "tested has router pass@1 == Base exactly). Default (None): don't sweep, "
                        "use the single --original-path-weight value as before (unchanged "
                        "behaviour). When set, OVERRIDES --original-path-weight and becomes the "
                        "OUTERMOST loop (build_examples/encode_examples re-run per value, since the "
                        "weight is baked into each example's .weight field at build time, not at "
                        "train() time) -- every value gets the full lr x batch x epochs x "
                        "lenpref-beta grid underneath it. Every grid config (not just the winner) "
                        "also gets a cheap, GENERATION-FREE nonidentity_rate: does the router's own "
                        "top-1 decode differ from the identity path? (predict_programs_topk is a "
                        "router-only forward pass -- no target-LLM call, so this costs nothing "
                        "beyond what training already does.) Answers two questions directly: (1) is "
                        "there a config elsewhere in the existing lr/batch/epoch/beta grid that "
                        "already escapes identity at the SAME original_path_weight, just not the "
                        "val-loss-selected winner -- val_loss selection stores none of that today, "
                        "this makes it visible; (2) does lowering the down-weight below the paper's "
                        "0.30 actually break the collapse. See the printed 'opw curve' (same shape "
                        "as the existing lenpref beta_curve) for the REAL online-graded pass@1/@k at "
                        "the best (by val_loss) config PER weight value.")
    p.add_argument("--drop-original-path", action="store_true",
                   help="DROP the identity path from targets (disables reweight). Off-recipe test lever.")
    p.add_argument("--segment-cap", choices=["all", "edit-only"], default="all",
                   help="all (default, unchanged): MAX_SEGMENT_LEN caps every segment, including KEEP "
                        "runs -- a long uninterrupted keep stretch is chunked into consecutive <=4-layer "
                        "pieces purely to fit the router's fixed-length segment representation (PoLar "
                        "Sec 3.1's literal text). edit-only (an ablation testing a specific hypothesis): "
                        "only SKIP/REPEAT segments stay capped; each maximal KEEP run merges into ONE "
                        "segment however long, so seg_flip boundaries only fire where the operation "
                        "actually changes. Execution-equivalent either way (KEEP takes no params) -- "
                        "this changes the router's training TARGET only, not the search space. See "
                        "re_polar/router/train.py program_from_layer_path(cap_keep=...) docstring.")
    p.add_argument("--strict-repeat-2x", action="store_true",
                   help="Reject (drop, whole-path) any valid path containing a REPEAT with times != 2, "
                        "matching PoLar's OWN released parser exactly (polar/data.py's DP only ever "
                        "tries chunk+chunk -- times 3/4/5 fail to parse and the path is silently dropped "
                        "from their training set; their decode is hardcoded to times==2 same as ours). "
                        "Off by default (lenient -- times>2 paths are kept, times info just isn't "
                        "supervised past the op-type label). Measured on diff1: about half of all valid "
                        "paths have >=1 repeat with times != 2 (up to 5) -- PoLar's pipeline would "
                        "reject over half our data. This flag matches their behaviour exactly, and is "
                        "the recipe the paper's own winning checkpoint (\"Drop+CE\") actually uses.")
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="DR.LLM-style focal loss gamma for the op-head CE (0.0 = plain CE, off by "
                        "default; DR.LLM's own value is 2.0). When >0, ALSO applies DR.LLM's exact "
                        "class-balanced 'effective number of samples' alpha (beta=0.999, computed from "
                        "this group's own TRAIN op-label counts) as the focal weight.")
    p.add_argument("--anti-original-lambdas", default="0.0",
                   help="comma-sep anti_original_lambda values to sweep as an ADDITIONAL grid axis, "
                        "following the opw-sweep finding that down-weighting the identity-path "
                        "EXAMPLE's own loss to 0.0 does not break the collapse. This is PoLar's "
                        "OTHER, separate anti-collapse lever (run_polar.py --anti_original_lambda, "
                        "default 0.0 -- unused in their own documented recipe too) with a different "
                        "mechanism: an explicit auxiliary loss term that directly penalizes the mean "
                        "predicted KEEP-probability across layers, applied only to samples whose "
                        "identity path is NOT EVEN VALID at all (re_polar/router/train.py compute_loss "
                        "anti_original_lambda>0 branch; Example.anti_original_active, set in "
                        "build_examples via identity not in valid). A structurally larger, more "
                        "directly-relevant subset than original_path_weight ever touched: on diff1, a "
                        "large minority of samples (roughly 45%%) have no valid identity path at all. "
                        "Unlike original_path_weight, this value does NOT change build_examples' output "
                        "(only WHETHER a sample is flagged anti-original, not the flag's strength -- that "
                        "lives in train()'s loss), so it is an INNER grid axis alongside lenpref_beta, "
                        "not an outer one requiring re-encode. Examples are built with anti_original=True "
                        "whenever any swept value is >0 (flagging happens once; the actual penalty "
                        "strength varies per grid cell via train(anti_original_lambda=...)). Each value "
                        "also gets its own 'anti-original-lambda curve' (best lr/batch/epochs/beta AT "
                        "that lambda) mirroring the existing beta_curve/opw_curve pattern.")
    p.add_argument("--lenpref-betas", default="0.0",
                   help="comma-sep beta values to sweep for PoLar's OWN 'polar_lenpref' policy_mode "
                        "(exp(-beta*path_len) length-preference reweighting of the per-example loss), "
                        "added as an ADDITIONAL axis to the existing lr x batch x epochs grid -- every "
                        "beta value gets the full lr/batch/epochs grid, and the group's single winner is "
                        "still whichever config (now including beta) has the lowest val_loss. beta=0.0 "
                        "runs under policy_mode='polar' (today's unchanged default/baseline, matching "
                        "PoLar's own README sample command); any beta>0.0 runs under "
                        "policy_mode='polar_lenpref' with that beta -- both exist verbatim in PoLar's own "
                        "run_polar.py (--policy_mode {polar,polar_lenpref}, --lenpref_beta default 0.05), "
                        "but their OWN documented README command never enables polar_lenpref -- this is an "
                        "unexplored-by-them extension of an already-faithful mechanism, not a fidelity "
                        "fix, motivated by re-reading the paper's own Finding-2 justification for the "
                        "down-weight. CAVEAT: val_loss under different beta values is a differently-"
                        "WEIGHTED average (not strictly apples-to-apples across beta), though weights are "
                        "renormalized to mean 1 each batch so the scale stays comparable -- treat "
                        "cross-beta val_loss comparison as an approximation, same as PoLar's own recipe "
                        "would if they swept it. ALSO: since val_loss selection structurally favours "
                        "beta=0 (perturbing training toward length-preference almost always costs some "
                        "raw val CE), the sweep additionally runs ONE generation-eval per beta value "
                        "(best lr/batch/epochs AT that beta, not just the global val_loss winner) so "
                        "nonident is actually visible as a function of beta -- see the per-group 'beta "
                        "curve' table and each group's 'beta_curve' field in the output .sweep.json.")
    p.add_argument("--per-sample-weight-normalize", type=lambda s: s.lower() not in ("0", "false", "no"),
                   default=True,
                   help="Normalize each SAMPLE's total example weight to ~1 before the down-weight "
                        "multiplier applies (prevents a question with many MCTS valid paths from "
                        "dominating the loss over one with few). Default True. PoLar's own CLI defaults "
                        "this OFF; their README's one documented training command turns it ON (alongside "
                        "the down-weight, alongside warmup=10) -- the paper's prose never mentions this "
                        "setting either way, so 'True' matches their documented example, not their bare "
                        "CLI default. Pass --per-sample-weight-normalize=false to ablate.")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--gen-batch-size", type=int, default=32)
    p.add_argument("--prompt-style", default="paper_minimal_fewshot",
                   help="passed through to GenerationReward -- default matches this project's own "
                        "canonical MCTS-search prompt choice. Passing this through explicitly matters: "
                        "GenerationReward's OWN class default is 'raw' (no fewshot demo), so silently "
                        "omitting this flag would diverge from whatever prompt style actually generated "
                        "--samples.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    epochs_list = [int(x) for x in args.epochs_list.split(",") if x.strip()]
    batch_sizes = ([int(x) for x in args.batch_sizes.split(",") if x.strip()]
                   if args.batch_sizes else [args.batch_size])
    lenpref_betas = [float(x) for x in args.lenpref_betas.split(",") if x.strip()]
    anti_original_lambdas = [float(x) for x in args.anti_original_lambdas.split(",") if x.strip()]
    original_path_weights = ([float(x) for x in args.original_path_weights.split(",") if x.strip()]
                             if args.original_path_weights else [args.original_path_weight])
    device = _resolve_device(None)
    cfg = MODEL_REGISTRY[args.model]
    D = MODEL_REGISTRY[args.model]["num_layers"]

    build_kwargs = dict(max_paths_per_sample=args.max_paths_per_sample,
                        per_sample_weight_normalize=args.per_sample_weight_normalize,
                        reweight_original_path=not args.drop_original_path,
                        drop_original_path=args.drop_original_path,
                        shortest_only=(args.label_mode == "shortest"),
                        cap_keep=(args.segment_cap == "all"),
                        strict_repeat_2x=args.strict_repeat_2x, seed=args.seed,
                        anti_original=any(al > 0.0 for al in anti_original_lambdas))

    # groups: per-difficulty => one group per file; else one pooled group
    if args.per_difficulty:
        groups = [(f"diff{i + 1}", [path]) for i, path in enumerate(args.samples)]
    else:
        groups = [("all", list(args.samples))]
    print(f"selection metric: {args.select_metric}  |  grouping: "
          f"{'per-difficulty (' + str(len(groups)) + ' routers)' if args.per_difficulty else 'pooled (1 router)'}",
          flush=True)
    print(f"label mode: {args.label_mode}  |  segment cap: {args.segment_cap}", flush=True)
    print(f"original-path handling: {'DROP' if args.drop_original_path else 'reweight-0.30'}"
          f"{' (IGNORED -- label-mode=shortest)' if args.label_mode == 'shortest' else ''}", flush=True)
    if args.original_path_weights:
        print(f"original-path-weight SWEEP: {original_path_weights} (outermost axis, "
              f"examples rebuilt+re-encoded per value)", flush=True)
    if any(al > 0.0 for al in anti_original_lambdas):
        print(f"anti-original-lambda SWEEP: {anti_original_lambdas} (inner axis alongside "
              f"lenpref_beta, no re-encode needed)", flush=True)

    # ---- load frozen encoder + target LLM ONCE (shared across groups/configs) ----
    print("Loading frozen encoder...", flush=True)
    encoder = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to(device)

    def fresh_router():
        torch.manual_seed(args.seed)  # same init across configs -> isolates the hyperparameter effect
        return PolarRouter(num_layers=D, encoder=encoder).to(device)

    print(f"Loading {cfg['model_id']} for online winner grading...", flush=True)
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    reward_fn = GenerationReward(ProgramExecutor(engine), batch_size=args.gen_batch_size,
                                 prompt_style=args.prompt_style)

    group_summaries = []
    for gname, files in groups:
        print("\n" + "=" * 72, flush=True)
        print(f"GROUP {gname}  ({len(files)} file(s))", flush=True)

        # split train/val POSITIONALLY like PoLar (trainer slices [0:1250]=train,
        # [1250:]=val on the trainval-ordered merged file), NOT a random 90/10 holdout.
        train_samples, val_samples = [], []
        for path in files:
            s = load_supervision(path)
            train_samples.extend(s[: args.train_per_diff])
            val_samples.extend(s[args.train_per_diff:])

        # val questions for the WINNER measurement (subsample for cost) -- fixed
        # across every original_path_weight value below (only the TRAINING
        # weight on the identity-path example changes per value, not which val
        # questions get graded), so base_p1 only needs computing once per group.
        rng = random.Random(args.seed)
        vq = [(s["question"], s["gt_ans"]) for s in val_samples if s.get("gt_ans")]
        rng.shuffle(vq)
        if args.val_eval_limit:
            vq = vq[: args.val_eval_limit]
        val_questions = [q for q, _ in vq]
        val_gt = [g for _, g in vq]
        base_p1 = _mean(reward_fn(Program.identity(D), val_questions, val_gt))

        best = None                 # global winner across EVERY opw x beta x lr x batch x epochs
        # best (lr,batch,epochs) config PER BETA value (across every opw seen), by
        # the same sel_score used for the global winner -- lets us see nonident AT
        # EACH beta, not just at whichever beta happens to win the global val_loss
        # race (which is beta=0 essentially by construction: lenpref reweighting
        # perturbs training away from the pure-CE optimum, so it costs raw val CE
        # almost everywhere -- see --lenpref-betas help).
        best_per_beta = {}
        # best config PER original_path_weight VALUE, across its own beta/lr/batch/
        # epochs grid -- same idea as best_per_beta, one level up (--original-path-
        # weights help).
        best_per_opw = {}
        # best config PER anti_original_lambda VALUE, across its own opw/beta/lr/
        # batch/epochs grid -- same idea, one axis over (--anti-original-lambdas).
        best_per_antilambda = {}
        grid = []

        for opw in original_path_weights:
            this_build_kwargs = dict(build_kwargs, original_path_weight=opw)
            examples = build_examples(train_samples, D, **this_build_kwargs)
            val_examples = build_examples(val_samples, D, **this_build_kwargs)
            if not examples or not val_examples:
                print(f"  SKIP {gname} opw={opw:.3g}: no examples (train {len(examples)}, val {len(val_examples)})", flush=True)
                continue
            encode_examples(fresh_router(), examples, batch_size=args.encode_batch_size)
            encode_examples(fresh_router(), val_examples, batch_size=args.encode_batch_size)

            # GENERATION-FREE lookup data for val_cache_acc_at1 (re_polar.router.train.
            # evaluate_val_programs): per val question, its set of MCTS-VERIFIED valid
            # executed paths (final_valid_transitions, already on val_samples -- these
            # rows were part of the trainval-ordered MCTS run, just held out
            # positionally) + its shared encoded token_hidden. None of this touches the
            # target LLM, so it cannot inherit the batch-composition/bf16 noise that
            # online re-generation (val_passk) would.
            val_program_data = None
            if args.select_metric == "val_cache_acc_at1":
                valid_sets = {}
                for s in val_samples:
                    q = s.get("question")
                    vs = {tuple(int(x) for x in p) for p in (s.get("final_valid_transitions") or []) if p}
                    if q is not None and vs:
                        valid_sets[q] = vs
                token_hiddens = {}
                for e in val_examples:
                    if e.token_hidden is not None and e.question not in token_hiddens:
                        token_hiddens[e.question] = e.token_hidden
                vq_cache = [q for q in token_hiddens if q in valid_sets]
                if vq_cache:
                    val_program_data = {"questions": vq_cache, "token_hiddens": token_hiddens,
                                        "valid_sets": valid_sets}
                else:
                    print(f"  WARNING {gname} opw={opw:.3g}: no val question has a non-empty "
                          f"final_valid_transitions -- val_cache_acc_at1 will fall back to val_loss "
                          f"for this group.", flush=True)

            print(f"  train examples {len(examples)} | val examples {len(val_examples)} | "
                  f"winner-eval questions {len(val_questions)} | original_path_weight={opw:.3g}", flush=True)

            # DR.LLM-exact class-balanced focal weights, computed once per (GROUP, opw)
            # (data doesn't change across the lr/batch/epochs/beta grid) from this
            # group's own TRAIN op-label counts. See --focal-gamma help.
            op_class_weights = None
            if args.focal_gamma > 0:
                from collections import Counter
                op_counts = Counter()
                for e in examples:
                    for v in e.op_labels[e.op_labels != -100].tolist():
                        op_counts[v] += 1
                cb_beta = 0.999
                counts = [op_counts.get(i, 1) for i in range(3)]
                eff_num = [(1 - cb_beta ** c) / (1 - cb_beta) for c in counts]
                op_class_weights = torch.tensor([1.0 / e for e in eff_num], dtype=torch.float32)
                op_class_weights = (op_class_weights / op_class_weights.mean()).to(device)
                print(f"  focal_gamma={args.focal_gamma} op_counts={counts} "
                      f"class_weights={op_class_weights.tolist()}", flush=True)

            # ---- grid ----
            print(f"  {'opw':>6} {'lr':>8} {'batch':>6} {'epochs':>7} {'beta':>6} {'anti_lam':>8} {'val_loss':>9} {'sel_score':>10} {'nonident':>9} {'best?':>6}", flush=True)
            for epochs in epochs_list:
                for batch in batch_sizes:
                    for lr in lrs:
                        for beta in lenpref_betas:
                            for anti_lambda in anti_original_lambdas:
                                policy_mode = "polar_lenpref" if beta > 0.0 else "polar"
                                router = fresh_router()
                                inner_select_by = ("val_cache_acc_at1" if (args.select_metric == "val_cache_acc_at1"
                                                                            and val_program_data is not None)
                                                   else "val_loss")
                                res = train(router, examples, val_examples=val_examples, epochs=epochs, lr=lr,
                                            batch_size=batch, lr_scheduler="cosine", warmup_steps=args.warmup_steps,
                                            device=device, seed=args.seed, move_encodings_to_device=True,
                                            select_by=inner_select_by, val_program_data=val_program_data,
                                            val_topk=args.val_topk, policy_mode=policy_mode, lenpref_beta=beta,
                                            anti_original_lambda=anti_lambda,
                                            op_class_weights=op_class_weights, op_focal_gamma=args.focal_gamma)
                                # train() restored the epoch selected by `inner_select_by` into `router`.
                                cfg_vloss = min(res.val_losses) if res.val_losses else float("inf")
                                if inner_select_by == "val_cache_acc_at1":
                                    # generation-free: does the router's decoded top-1 land in a
                                    # question's MCTS-verified valid-path set? (res.best_metric is that
                                    # rate at the epoch train() already restored.)
                                    score = res.best_metric if res.best_metric is not None else float("-inf")
                                elif args.select_metric == "val_loss":
                                    score = -cfg_vloss  # maximise score == minimise loss
                                else:  # val_passk: online-grade (with --val-topk 1 this selects by pass@1 accuracy)
                                    router.eval()
                                    tk = predict_programs_topk(router, val_questions, k=args.val_topk, batch_size=args.gen_batch_size)
                                    score = _mean(grade_router_topk(tk, val_questions, val_gt, reward_fn)[0])
                                # GENERATION-FREE nonidentity check for THIS config (router-only
                                # forward pass, no target-LLM call). Doing this for EVERY grid
                                # config (not just the eventual val-loss winner) is what answers
                                # "is there a config elsewhere in the grid that already escapes
                                # identity" -- without it val_loss selection alone would never
                                # surface such a config even if one exists.
                                router.eval()
                                cfg_tk1 = predict_programs_topk(router, val_questions, k=1, batch_size=args.gen_batch_size)
                                cfg_nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in cfg_tk1])
                                is_best = (best is None) or (score > best["score"])
                                print(f"  {opw:>6.3g} {lr:>8.0e} {batch:>6} {epochs:>7} {beta:>6.3g} {anti_lambda:>8.3g} {cfg_vloss:>9.4f} {score:>10.4f} {cfg_nonident:>8.1%} {'  <=' if is_best else '':>6}",
                                      flush=True)
                                grid.append({"original_path_weight": opw, "lr": lr, "batch": batch, "epochs": epochs,
                                             "lenpref_beta": beta, "anti_original_lambda": anti_lambda, "policy_mode": policy_mode,
                                             "val_loss": cfg_vloss, "sel_score": score, "nonidentity_rate_generation_free": cfg_nonident})
                                cfg_record = {"score": score, "original_path_weight": opw, "lr": lr, "batch": batch,
                                              "epochs": epochs, "lenpref_beta": beta, "anti_original_lambda": anti_lambda,
                                              "policy_mode": policy_mode, "val_loss": cfg_vloss, "router": router}
                                if is_best:
                                    best = cfg_record
                                if beta not in best_per_beta or score > best_per_beta[beta]["score"]:
                                    best_per_beta[beta] = cfg_record
                                if opw not in best_per_opw or score > best_per_opw[opw]["score"]:
                                    best_per_opw[opw] = cfg_record
                                if anti_lambda not in best_per_antilambda or score > best_per_antilambda[anti_lambda]["score"]:
                                    best_per_antilambda[anti_lambda] = cfg_record

        if best is None:
            print(f"  SKIP {gname}: no valid config across any original_path_weight", flush=True)
            continue

        # ---- ONE online eval of the WINNER (measurement, not selection) ----
        winner = best["router"]
        winner.eval()
        tk = predict_programs_topk(winner, val_questions, k=args.val_topk, batch_size=args.gen_batch_size)
        r_at_k, r_at_1, _ = grade_router_topk(tk, val_questions, val_gt, reward_fn)
        win_p1, win_pk = _mean(r_at_1), _mean(r_at_k)
        nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in tk])

        out_path = Path(args.out)
        if args.per_difficulty:
            out_path = out_path.with_name(out_path.stem + f"_{gname}" + out_path.suffix)
        save_checkpoint(winner, str(out_path), meta={
            "sweep": True, "select_metric": args.select_metric, "label_mode": args.label_mode, "segment_cap": args.segment_cap, "group": gname,
            "original_path_weight": best["original_path_weight"], "lr": best["lr"], "epochs": best["epochs"], "batch_size": best["batch"],
            "lenpref_beta": best["lenpref_beta"], "anti_original_lambda": best["anti_original_lambda"], "policy_mode": best["policy_mode"],
            "val_loss": best["val_loss"], "base_val_p1": base_p1, "val_p1": win_p1,
            f"val_p{args.val_topk}": win_pk, "nonidentity_rate": nonident, "model": args.model})
        print(f"  WINNER {gname}: original_path_weight={best['original_path_weight']:.3g} lr={best['lr']:.0e} batch={best['batch']} epochs={best['epochs']} "
              f"policy_mode={best['policy_mode']} lenpref_beta={best['lenpref_beta']:.3g} anti_original_lambda={best['anti_original_lambda']:.3g} "
              f"val_loss={best['val_loss']:.4f} "
              f"-> base_p1 {base_p1:.4f} | router_p1 {win_p1:.4f} (Δ{win_p1 - base_p1:+.4f}) "
              f"| router_p{args.val_topk} {win_pk:.4f} | nonident {nonident:.1%}  -> {out_path}", flush=True)
        group_summaries.append({"group": gname, "original_path_weight": best["original_path_weight"],
                                "lr": best["lr"], "batch": best["batch"], "epochs": best["epochs"],
                                "lenpref_beta": best["lenpref_beta"], "anti_original_lambda": best["anti_original_lambda"],
                                "policy_mode": best["policy_mode"],
                                "val_loss": best["val_loss"], "base_val_p1": base_p1, "val_p1": win_p1,
                                f"val_p{args.val_topk}": win_pk, "nonidentity_rate": nonident, "grid": grid})

        # ---- ALSO eval the best config AT EACH BETA (not just the global winner) ----
        # val_loss selection structurally favours beta=0 (see best_per_beta comment
        # above), so without this the sweep can never show whether ANY beta>0 actually
        # moves nonident off 0% -- it would always silently pick beta=0 and we'd never
        # know. One extra generation-eval pass per beta value (cheap relative to the
        # lr x batch x epochs grid itself, which needs no generation at all).
        beta_curve = []
        for beta in lenpref_betas:
            cfg = best_per_beta[beta]
            if cfg is best:
                # global winner already evaluated above -- reuse, don't re-run generation.
                b_p1, b_pk, b_nonident = win_p1, win_pk, nonident
            else:
                b_router = cfg["router"]
                b_router.to(device)
                b_router.eval()
                b_tk = predict_programs_topk(b_router, val_questions, k=args.val_topk, batch_size=args.gen_batch_size)
                b_r_at_k, b_r_at_1, _ = grade_router_topk(b_tk, val_questions, val_gt, reward_fn)
                b_p1, b_pk = _mean(b_r_at_1), _mean(b_r_at_k)
                b_nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in b_tk])
                b_router.to("cpu")  # free GPU memory before the next beta's eval
            beta_curve.append({"lenpref_beta": beta, "policy_mode": cfg["policy_mode"],
                                "original_path_weight": cfg["original_path_weight"],
                                "lr": cfg["lr"], "batch": cfg["batch"], "epochs": cfg["epochs"],
                                "val_loss": cfg["val_loss"], "val_p1": b_p1,
                                f"val_p{args.val_topk}": b_pk, "nonidentity_rate": b_nonident})
        group_summaries[-1]["beta_curve"] = beta_curve
        print(f"  beta curve {gname} (best lr/bs/ep PER beta, base_p1 {base_p1:.4f}):", flush=True)
        print(f"    {'beta':>6} {'val_loss':>9} {'router_p1':>10} {'Δp1':>7} {'nonident':>9}", flush=True)
        for c in beta_curve:
            print(f"    {c['lenpref_beta']:>6.3g} {c['val_loss']:>9.4f} {c['val_p1']:>10.4f} "
                  f"{c['val_p1'] - base_p1:>+7.4f} {c['nonidentity_rate']:>8.1%}", flush=True)

        # ---- ALSO eval the best config AT EACH original_path_weight (not just the
        # global winner) -- same rationale as the beta curve above, one axis over.
        # Answers directly: does lowering the down-weight below the paper's 0.30
        # default actually reduce nonidentity_rate, and at what real accuracy cost
        # (if any)?
        opw_curve = []
        for opw in original_path_weights:
            if opw not in best_per_opw:
                continue  # this value had no valid examples for this group, skipped above
            cfg = best_per_opw[opw]
            if cfg is best:
                o_p1, o_pk, o_nonident = win_p1, win_pk, nonident
            else:
                o_router = cfg["router"]
                o_router.to(device)
                o_router.eval()
                o_tk = predict_programs_topk(o_router, val_questions, k=args.val_topk, batch_size=args.gen_batch_size)
                o_r_at_k, o_r_at_1, _ = grade_router_topk(o_tk, val_questions, val_gt, reward_fn)
                o_p1, o_pk = _mean(o_r_at_1), _mean(o_r_at_k)
                o_nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in o_tk])
                o_router.to("cpu")
            opw_curve.append({"original_path_weight": opw, "lenpref_beta": cfg["lenpref_beta"],
                              "anti_original_lambda": cfg["anti_original_lambda"],
                              "policy_mode": cfg["policy_mode"], "lr": cfg["lr"], "batch": cfg["batch"],
                              "epochs": cfg["epochs"], "val_loss": cfg["val_loss"], "val_p1": o_p1,
                              f"val_p{args.val_topk}": o_pk, "nonidentity_rate": o_nonident})
        group_summaries[-1]["opw_curve"] = opw_curve
        print(f"  original-path-weight curve {gname} (best lr/bs/ep/beta PER weight, base_p1 {base_p1:.4f}):", flush=True)
        print(f"    {'opw':>6} {'val_loss':>9} {'router_p1':>10} {'Δp1':>7} {'nonident':>9}", flush=True)
        for c in opw_curve:
            print(f"    {c['original_path_weight']:>6.3g} {c['val_loss']:>9.4f} {c['val_p1']:>10.4f} "
                  f"{c['val_p1'] - base_p1:>+7.4f} {c['nonidentity_rate']:>8.1%}", flush=True)

        # ---- ALSO eval the best config AT EACH anti_original_lambda (not just the
        # global winner) -- same rationale as beta_curve/opw_curve, one axis over.
        # Answers: does PoLar's OTHER (unused-by-default) anti-collapse lever --
        # penalizing mean KEEP-probability on samples whose identity path isn't even
        # valid -- move nonident where original_path_weight provably did not?
        antilambda_curve = []
        for anti_lambda in anti_original_lambdas:
            if anti_lambda not in best_per_antilambda:
                continue
            cfg = best_per_antilambda[anti_lambda]
            if cfg is best:
                a_p1, a_pk, a_nonident = win_p1, win_pk, nonident
            else:
                a_router = cfg["router"]
                a_router.to(device)
                a_router.eval()
                a_tk = predict_programs_topk(a_router, val_questions, k=args.val_topk, batch_size=args.gen_batch_size)
                a_r_at_k, a_r_at_1, _ = grade_router_topk(a_tk, val_questions, val_gt, reward_fn)
                a_p1, a_pk = _mean(a_r_at_1), _mean(a_r_at_k)
                a_nonident = _mean([0.0 if c[0].to_layer_path() == list(range(D)) else 1.0 for c in a_tk])
                a_router.to("cpu")
            antilambda_curve.append({"anti_original_lambda": anti_lambda, "lenpref_beta": cfg["lenpref_beta"],
                                     "original_path_weight": cfg["original_path_weight"],
                                     "policy_mode": cfg["policy_mode"], "lr": cfg["lr"], "batch": cfg["batch"],
                                     "epochs": cfg["epochs"], "val_loss": cfg["val_loss"], "val_p1": a_p1,
                                     f"val_p{args.val_topk}": a_pk, "nonidentity_rate": a_nonident})
        group_summaries[-1]["anti_original_lambda_curve"] = antilambda_curve
        print(f"  anti-original-lambda curve {gname} (best lr/bs/ep/beta/opw PER lambda, base_p1 {base_p1:.4f}):", flush=True)
        print(f"    {'anti_lam':>8} {'val_loss':>9} {'router_p1':>10} {'Δp1':>7} {'nonident':>9}", flush=True)
        for c in antilambda_curve:
            print(f"    {c['anti_original_lambda']:>8.3g} {c['val_loss']:>9.4f} {c['val_p1']:>10.4f} "
                  f"{c['val_p1'] - base_p1:>+7.4f} {c['nonidentity_rate']:>8.1%}", flush=True)

    # ---- combined summary ----
    print("\n" + "=" * 72, flush=True)
    print(f"{'group':>7} {'opw':>6} {'lr':>8} {'bs':>5} {'ep':>4} {'beta':>6} {'anti_lam':>8} {'val_loss':>9} {'base_p1':>8} {'router_p1':>10} "
          f"{'Δp1':>7} {'router_pk':>10} {'nonident':>9}", flush=True)
    for s in group_summaries:
        print(f"{s['group']:>7} {s['original_path_weight']:>6.3g} {s['lr']:>8.0e} {s['batch']:>5} {s['epochs']:>4} {s['lenpref_beta']:>6.3g} {s['anti_original_lambda']:>8.3g} {s['val_loss']:>9.4f} {s['base_val_p1']:>8.4f} "
              f"{s['val_p1']:>10.4f} {s['val_p1'] - s['base_val_p1']:>+7.4f} {s[f'val_p{args.val_topk}']:>10.4f} "
              f"{s['nonidentity_rate']:>8.1%}", flush=True)

    out = {"model": args.model, "select_metric": args.select_metric, "label_mode": args.label_mode, "segment_cap": args.segment_cap,
           "per_difficulty": args.per_difficulty, "val_topk": args.val_topk, "prompt_style": args.prompt_style,
           "lenpref_betas": lenpref_betas, "original_path_weights": original_path_weights,
           "anti_original_lambdas": anti_original_lambdas, "groups": group_summaries}
    resj = Path(args.out).with_suffix(".sweep.json")
    resj.write_text(json.dumps(out, indent=2))
    print(f"\nwrote sweep table -> {resj}", flush=True)


if __name__ == "__main__":
    main()
