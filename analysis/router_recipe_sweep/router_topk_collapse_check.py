"""Does the router's pass@k candidate set COLLAPSE to a fixed small menu,
independent of the question -- as opposed to pass@1's already-documented
collapse to identity?

Decode-only, GENERATION-FREE (same spirit as sweep_router.py's cheap
nonidentity_rate signal, extended to ranks 2..k): loads a trained
PolarRouter checkpoint and runs `predict_programs_topk` over a data split,
then measures -- for every k in 1..K -- how many DISTINCT programs appear
anywhere across all questions' top-k lists ("menu size at k"), the identity
share, and how often a question's entire top-k SET matches the single most
common set. No LLM generation, no execution grading -- this is strictly
about what the router PREDICTS, not whether predictions are correct (real
pass@k accuracy is a separate, already-computed val_p5 field in the paired
.sweep.json from `sweep_router.py` / a separate, much more expensive ask on
test).

Deliberately does NOT build a LayerEngine/GenerationReward/target-LLM at
all -- `router.num_layers` is already in the checkpoint (see
re_polar/router/train.py::load_checkpoint), so nothing but the frozen
Qwen3-Embedding-0.6B encoder + small heads ever loads.

TIMING NOTE: on GPU the actual predict_programs_topk compute is a couple
seconds -- the large majority of wall-clock time is one-time encoder+
tokenizer load (`AutoModel.from_pretrained`/`AutoTokenizer.from_pretrained`),
NOT per-checkpoint work. `--recipe-dir` batch mode below exists BECAUSE of
this: it loads the encoder+tokenizer ONCE and shares them
(`load_checkpoint(..., encoder=...)`) across every checkpoint/split/
difficulty in one process, so a run over many checkpoints costs roughly one
encoder load, not one per checkpoint.

Usage (single checkpoint):
  python -m analysis.router_recipe_sweep.router_topk_collapse_check \
      --checkpoint router_qwen3_8b_diff1_strict_ce.pt \
      --data-dir dart_math_v2 --difficulty 1 \
      --split test --k 5 \
      --output topk_collapse_qwen3_8b_diff1_test.json

Usage (batch, the real run -- all models/diffs found under --recipe-dir,
sharing one encoder load):
  python -m analysis.router_recipe_sweep.router_topk_collapse_check \
      --recipe-dir winning_recipe_strict_ce \
      --data-dir dart_math_v2 \
      --splits test,val,train --k 5 \
      --output topk_collapse_winning_recipe_strict_ce_all.json
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DIFF_RE = re.compile(r"_diff(\d)_")


def load_split(data_dir, difficulty: int, split: str, limit=None) -> List[dict]:
    path = Path(data_dir) / f"diff{difficulty}" / f"{split}.json"
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of questions, got {type(data)}")
    if limit is not None:
        data = data[:limit]
    return data


def resolve_difficulties(raw) -> List[int]:
    if not raw:
        return [1, 2, 3, 4, 5]
    if any(str(x).strip().lower() == "all" for x in raw):
        return [1, 2, 3, 4, 5]
    return sorted({int(x) for x in raw})


def collapse_diagnostics(topk_programs: Sequence[Sequence[object]], num_layers: int, k_max: int) -> dict:
    """Per-k (1..k_max) menu-collapse diagnostics over one (model, diff, split).

    topk_programs[i] = question i's decoded candidates, best-first (already
    per-question deduped by decode_topk).
    """
    identity = tuple(range(num_layers))
    n = len(topk_programs)
    per_k = {}
    for k in range(1, k_max + 1):
        truncated = [tuple(tuple(p.to_layer_path()) for p in cands[:k]) for cands in topk_programs]
        # menu size at k: distinct programs anywhere across all questions' top-k lists
        pool = set()
        for cands in truncated:
            pool.update(cands)
        # rank-wise frequency (rank r's program, counted over questions that have >= r+1 candidates)
        rank_counts = [dict() for _ in range(k)]
        for cands in truncated:
            for r, p in enumerate(cands):
                rank_counts[r][p] = rank_counts[r].get(p, 0) + 1
        rank_top = []
        for r in range(k):
            total_r = sum(rank_counts[r].values())
            if total_r == 0:
                rank_top.append({"program": None, "share": 0.0, "is_identity": None})
                continue
            best_p, best_c = max(rank_counts[r].items(), key=lambda kv: kv[1])
            rank_top.append({
                "program": list(best_p), "share": round(best_c / total_r, 4),
                "is_identity": best_p == identity,
            })
        # exact top-k SET collapse: how often does a question's whole candidate
        # set (order-independent) match the single most common set?
        set_counts = {}
        for cands in truncated:
            key = frozenset(cands)
            set_counts[key] = set_counts.get(key, 0) + 1
        most_common_set_count = max(set_counts.values()) if set_counts else 0
        identity_at_1 = sum(1 for cands in truncated if cands and cands[0] == identity)
        per_k[str(k)] = {
            "menu_size": len(pool),
            "menu_size_over_n": round(len(pool) / n, 4) if n else 0.0,
            "n_distinct_sets": len(set_counts),
            "most_common_set_share": round(most_common_set_count / n, 4) if n else 0.0,
            "identity_at_rank1_share": round(identity_at_1 / n, 4) if n else 0.0,
            "rank_top": rank_top,
        }
    return {"n_questions": n, "num_layers": num_layers, "per_k": per_k}


def _resolve_device(name):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_checkpoints(recipe_dir, models: Optional[Sequence[str]] = None) -> List[Tuple[str, int, Path]]:
    """(model, diff, path) for every ``*_diffN_*.pt`` under one subdir per model.

    Filenames may vary slightly by model (some checkpoint-naming conventions
    carry extra infixes), but always contain ``_diff{N}_`` -- so we glob
    broadly and parse the diff out with `DIFF_RE` rather than assume one
    fixed filename template.
    """
    recipe_dir = Path(recipe_dir)
    found = []
    for model_dir in sorted(p for p in recipe_dir.iterdir() if p.is_dir()):
        if models and model_dir.name not in models:
            continue
        for ckpt in sorted(model_dir.glob("*.pt")):
            m = DIFF_RE.search(ckpt.name)
            if not m:
                print(f"WARNING: {ckpt} has no _diffN_ in its name, skipping", flush=True)
                continue
            found.append((model_dir.name, int(m.group(1)), ckpt))
    if not found:
        raise SystemExit(f"no checkpoints found under {recipe_dir}")
    return found


def resolve_splits(raw) -> List[str]:
    splits: List[str] = []
    for item in raw or ["test"]:
        splits.extend(s.strip() for s in str(item).split(",") if s.strip())
    valid = {"train", "val", "test"}
    bad = [s for s in splits if s not in valid]
    if bad:
        raise SystemExit(f"unknown split(s) {bad}; must be in {sorted(valid)}")
    return splits


def _run_router_over_splits(router, model, diff, data_dir, splits, k, batch_size, limit):
    """One checkpoint x its own diff, over every requested split -> list of diag dicts."""
    from re_polar.router.infer import predict_programs_topk

    rows = []
    for split in splits:
        t1 = time.time()
        data = load_split(data_dir, diff, split, limit)
        questions = [d["question"] for d in data]
        topk = predict_programs_topk(router, questions, k=k, batch_size=batch_size)
        diag = collapse_diagnostics(topk, router.num_layers, k)
        diag.update(model=model, difficulty=diff, split=split, seconds=round(time.time() - t1, 2))
        rows.append(diag)
        top_k_diag = diag["per_k"][str(k)]
        print(f"[{model} diff{diff}, {split}, n={diag['n_questions']}] "
              f"menu_size@1={diag['per_k']['1']['menu_size']} "
              f"menu_size@{k}={top_k_diag['menu_size']} "
              f"({top_k_diag['menu_size_over_n']:.1%} of n) "
              f"most_common_set_share@{k}={top_k_diag['most_common_set_share']:.1%} "
              f"identity@1={diag['per_k']['1']['identity_at_rank1_share']:.1%} "
              f"({diag['seconds']}s)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Router pass@k menu-collapse check (decode-only, no generation).")
    ap.add_argument("--checkpoint", help="single-checkpoint mode: one .pt file")
    ap.add_argument("--difficulty", action="append", default=None,
                     help="single-checkpoint mode only: int (repeatable) or 'all'")
    ap.add_argument("--recipe-dir",
                     help="batch mode: dir with one subdir per model, each holding *_diffN_*.pt "
                          "checkpoints -- loads the frozen encoder ONCE and shares it across "
                          "every checkpoint found")
    ap.add_argument("--models", action="append", default=None,
                     help="batch mode only: restrict to these model subdir names (repeatable)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                     help="single-checkpoint mode: one split (legacy flag, still supported)")
    ap.add_argument("--splits", action="append", default=None,
                     help="batch mode: comma-separated and/or repeatable, e.g. test,val,train "
                          "(default: test only)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N questions per split (smoke)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    if bool(args.checkpoint) == bool(args.recipe_dir):
        raise SystemExit("pass exactly one of --checkpoint (single) or --recipe-dir (batch)")

    from re_polar.router.train import load_checkpoint

    t0 = time.time()
    device = _resolve_device(args.device)
    results = []

    if args.checkpoint:
        # --- single-checkpoint mode (unchanged behaviour, used by the initial probes) ---
        splits = [args.split] if args.split else resolve_splits(args.splits)
        difficulties = resolve_difficulties(args.difficulty)
        router = load_checkpoint(args.checkpoint)
        router.encode_questions(["warmup"])  # materialize the lazy frozen encoder
        router.to(device)
        router.eval()
        load_s = time.time() - t0
        for diff in difficulties:
            results.extend(_run_router_over_splits(
                router, "checkpoint", diff, args.data_dir, splits, args.k, args.batch_size, args.limit))
    else:
        # --- batch mode: one encoder+tokenizer load shared across every checkpoint ---
        splits = resolve_splits(args.splits)
        jobs = discover_checkpoints(args.recipe_dir, args.models)
        print(f"Found {len(jobs)} checkpoints under {args.recipe_dir} "
              f"({len(set(m for m, _, _ in jobs))} model(s)); splits={splits}", flush=True)
        shared_encoder = None
        shared_tokenizer = None
        load_s = None
        for model, diff, ckpt_path in jobs:
            t1 = time.time()
            if shared_encoder is None:
                router = load_checkpoint(ckpt_path)
                router.encode_questions(["warmup"])  # materializes + attaches the default encoder
                shared_encoder = router.encoder
                shared_tokenizer = router._tokenizer
                load_s = time.time() - t1  # one-time cost, reported separately
            else:
                router = load_checkpoint(ckpt_path, encoder=shared_encoder)
                router._tokenizer = shared_tokenizer  # skip a redundant AutoTokenizer.from_pretrained
            router.to(device)
            router.eval()
            results.extend(_run_router_over_splits(
                router, model, diff, args.data_dir, splits, args.k, args.batch_size, args.limit))

    out = {
        "mode": "batch" if args.recipe_dir else "single",
        "source": str(args.recipe_dir or args.checkpoint), "k": args.k,
        "device": str(device), "first_load_seconds": round(load_s, 2) if load_s is not None else None,
        "total_seconds": round(time.time() - t0, 2),
        "rows": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output} -- {len(results)} (checkpoint x split) rows, "
          f"total wall-clock {out['total_seconds']}s (first load {out['first_load_seconds']}s)", flush=True)
    return Path(args.output)


if __name__ == "__main__":
    main()
