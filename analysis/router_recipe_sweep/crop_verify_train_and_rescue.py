"""The third repeat-handling data treatment, "Crop-and-Re-verify": for every
TRAIN path with a REPEAT segment of times>2, crop it to times=2 and
RE-EXECUTE it online against the real question/gt -- MCTS only ever verified
the original times>=3 path, never its 2x-truncated form, so "keep it as
REPEAT supervision" (Crop, see --no-strict-repeat-2x in re_polar/router/train.py)
and "drop the path" (Drop, --strict-repeat-2x) are both unverified guesses
about whether the cropped program is still correct. This script actually
checks, and only keeps the cropped label when it verifiably is.

Pipeline (one job, one model load):
  1. VERIFY: for each TRAIN [0:1250] sample with >=1 times!=2 path, build the
     cropped (times=2) program, execute it online (GenerationReward, the
     standard cached serial path -- NOT masked_batch_call, which has a
     measured 8.6-8.8% verdict-flip cost, not wanted for a trustworthy
     verdict here). Correct -> keep the cropped path (now a genuinely
     times=2-verified non-identity program). Incorrect -> drop that path
     from training entirely.
  2. WRITE a new merged_mcts_samples.json variant: TRAIN indices get the
     crop-verified `final_valid_transitions`; VAL/TEST indices are untouched
     (RESCUE measurement must stay against the REAL MCTS cache).
  3. TRAIN one router on the new data (single config: lr=3e-4, batch=32,
     epochs=10, cosine+warmup=10 -- the config the paper's own Drop/Crop
     sweeps converged to as their winner, not cherry-picked here).
  4. RESCUE: same real-protocol (online) check as the paper's rescue-vs-random
     comparison, on the identity-wrong subset of the [1500:1750] TEST slice.

Usage:
    python -m analysis.router_recipe_sweep.crop_verify_train_and_rescue \\
        --samples /path/to/Qwen/Qwen3-8B/dart-math-diff-1/merged_mcts_samples.json \\
        --out-data /path/to/merged_mcts_samples_cropverified.json \\
        --out-ckpt /path/to/router_qwen3_8b_diff1_cropverified.pt \\
        --model qwen3_8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--samples", required=True)
    ap.add_argument(
        "--out-data", required=True, help="where to write the crop-verified data variant"
    )
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--train-per-diff", type=int, default=1250)
    ap.add_argument("--test-start", type=int, default=1500)
    ap.add_argument("--test-end", type=int, default=1750)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument(
        "--prompt-style",
        default="paper_minimal_fewshot",
        help="passed through to GenerationReward -- must match whatever prompt style "
        "originally decided the (uncropped) paths were valid during MCTS search, "
        "since this script's whole job is a like-for-like online re-verification "
        "of the cropped form under that same protocol.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # heavy imports inside main, matching this package's other scripts (sweep_router.py)
    import torch
    from transformers import AutoModel
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor, Program, Segment, Op
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.router.model import PolarRouter
    from re_polar.router.infer import grade_router_topk, predict_programs_topk
    from re_polar.router.train import (
        program_from_layer_path,
        build_examples,
        encode_examples,
        train,
        save_checkpoint,
        _resolve_device,
    )

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    cfg = MODEL_REGISTRY[args.model]
    D = cfg["num_layers"]
    device = _resolve_device(None)

    data = json.loads(Path(args.samples).read_text())
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    train_samples = samples[: args.train_per_diff]
    print(
        f"Loading {cfg['model_id']} + executor for crop-verification and training "
        f"(prompt_style={args.prompt_style}) ...",
        flush=True,
    )
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    reward_fn = GenerationReward(
        ProgramExecutor(engine), batch_size=args.gen_batch_size, prompt_style=args.prompt_style
    )

    # ---- Phase 1: crop + verify -------------------------------------------------
    print(f"\n=== Phase 1: crop-verify {len(train_samples)} TRAIN samples ===", flush=True)
    n_pairs = n_kept = n_dropped = 0
    for i, s in enumerate(train_samples):
        valid = [p for p in (s.get("final_valid_transitions") or []) if p]
        question = s.get("question")
        gt = s.get("gt_ans")
        new_valid = []
        for path in valid:
            prog = program_from_layer_path(list(path), D, strict_repeat_2x=False)
            offending = [seg for seg in prog.segments if seg.op is Op.REPEAT and seg.times != 2]
            if not offending:
                new_valid.append(list(path))  # already times==2 or no repeat -- unchanged
                continue
            n_pairs += 1
            cropped_segs = [
                Segment(seg.start, seg.end, seg.op, {"times": 2}) if seg in offending else seg
                for seg in prog.segments
            ]
            cropped_program = Program(num_layers=D, segments=cropped_segs)
            cropped_path = cropped_program.to_layer_path()
            if not gt:
                n_dropped += 1
                continue
            reward = reward_fn(cropped_program, [question], [gt])[0]
            if reward >= 1.0:
                n_kept += 1
                new_valid.append(cropped_path)
            else:
                n_dropped += 1
        s["final_valid_transitions"] = new_valid
        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{len(train_samples)} samples processed | "
                f"pairs so far: {n_pairs} (kept {n_kept}, dropped {n_dropped})",
                flush=True,
            )

    print(
        f"\nCrop-verification done: {n_pairs} (sample,path) pairs with times!=2 checked", flush=True
    )
    print(
        f"  kept (cropped-to-2x version verified CORRECT): {n_kept}/{n_pairs} "
        f"= {n_kept/max(1,n_pairs):.4f}",
        flush=True,
    )
    print(
        f"  dropped (cropped-to-2x version WRONG or no gt_ans): {n_dropped}/{n_pairs} "
        f"= {n_dropped/max(1,n_pairs):.4f}",
        flush=True,
    )

    out_data = {"samples": samples} if isinstance(data, dict) and "samples" in data else samples
    Path(args.out_data).write_text(json.dumps(out_data))
    print(f"Wrote crop-verified data variant: {args.out_data}", flush=True)

    # ---- Phase 2: train one router on the new data ------------------------------
    print(f"\n=== Phase 2: train router on crop-verified data ===", flush=True)
    val_samples = samples[args.train_per_diff :]
    build_kwargs = dict(
        max_paths_per_sample=50,
        reweight_original_path=True,
        original_path_weight=0.30,
        cap_keep=True,
        strict_repeat_2x=False,  # moot now -- no times!=2 paths remain
        seed=args.seed,
    )
    examples = build_examples(train_samples, D, **build_kwargs)
    val_examples = build_examples(val_samples, D, **build_kwargs)
    print(f"train examples {len(examples)} | val examples {len(val_examples)}", flush=True)

    encoder = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to(device)

    def fresh_router():
        torch.manual_seed(args.seed)
        return PolarRouter(num_layers=D, encoder=encoder).to(device)

    encode_examples(fresh_router(), examples, batch_size=64)
    encode_examples(fresh_router(), val_examples, batch_size=64)

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
    val_program_data = (
        {"questions": vq_cache, "token_hiddens": token_hiddens, "valid_sets": valid_sets}
        if vq_cache
        else None
    )
    select_by = "val_cache_acc_at1" if val_program_data is not None else "val_loss"

    router = fresh_router()
    res = train(
        router,
        examples,
        val_examples=val_examples,
        epochs=10,
        lr=3e-4,
        batch_size=32,
        lr_scheduler="cosine",
        warmup_steps=10,
        device=device,
        seed=args.seed,
        move_encodings_to_device=True,
        select_by=select_by,
        val_program_data=val_program_data,
        val_topk=5,
    )
    print(f"Training done. select_by={select_by} best_metric={res.best_metric}", flush=True)
    save_checkpoint(router, args.out_ckpt, meta={"variant": "cropverified"})
    print(f"Saved checkpoint: {args.out_ckpt}", flush=True)

    # ---- Phase 3: real-protocol RESCUE on identity-wrong TEST subset ------------
    print(f"\n=== Phase 3: real-protocol RESCUE on identity-wrong TEST subset ===", flush=True)
    test = samples[args.test_start : args.test_end]
    identity = tuple(range(D))
    identity_wrong = [
        s
        for s in test
        if s.get("gt_ans")
        and identity not in {tuple(p) for p in (s.get("final_valid_transitions") or []) if p}
    ]
    questions = [s["question"] for s in identity_wrong]
    gt = [s["gt_ans"] for s in identity_wrong]
    n = len(questions)
    print(f"Identity-wrong TEST subset: {n}", flush=True)

    router.eval()
    tk = predict_programs_topk(router, questions, k=5, batch_size=args.gen_batch_size)
    r_at_k, r_at_1, _ = grade_router_topk(tk, questions, gt, reward_fn)
    rescue_at1 = _mean(r_at_1)
    rescue_at5 = _mean(r_at_k)

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"Crop-verification: {n_kept}/{n_pairs} = {n_kept/max(1,n_pairs):.4f} of times!=2 paths "
        f"verified correct after cropping to 2x",
        flush=True,
    )
    print(f"RESCUE_cropverified@1: {sum(r_at_1):.0f}/{n} = {rescue_at1:.4f}", flush=True)
    print(f"RESCUE_cropverified@5: {sum(r_at_k):.0f}/{n} = {rescue_at5:.4f}", flush=True)


if __name__ == "__main__":
    main()
