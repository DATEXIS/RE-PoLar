"""IDENTITY BASELINE on the OOD eval sets (ASDiv/MAWPS), no router, no checkpoint.

Grades the identity program (full model, no layer edits) with
GenerationReward (boxed gate + strict extract) -- the accuracy any router
built on these sets would need to beat. Split out into its own script
because ASDiv/MAWPS (`re_polar/datasets/{asdiv,mawps}.py`) are FLAT single-file eval
sets (no diff{N}/ per-difficulty directories, no train/val), the
difficulty-binned DART-Math loading logic doesn't apply here.

Default `--prompt-style paper_minimal_fewshot` matches this project's own
default MCTS-search prompt, so this baseline is directly comparable to the
DART-Math numbers on record for the same model; PoLar's Appendix D.4
"Direct Prompting" template itself (`--prompt-style raw`, no few-shot) is
also available for a more literal paper-protocol run, the underlying
instruction+format string is identical either way (`re_polar/mcts/rewards.py`
`PAPER_INSTRUCTION`), few-shot styles only prepend worked examples.

  python -m analysis.baseline_reproduction.baseline_identity_ood \
      --model qwen3_8b --dataset asdiv \
      --output baseline_identity_ood_asdiv.json

Top level is stdlib-only so spawn grading workers re-importing this module
stay torch-free.
"""

import argparse
import json
from pathlib import Path

# dataset name -> path relative to --data-root, matching re_polar/datasets/{asdiv,mawps}.py's
# own output layout
DATASETS = {
    "asdiv": "asdiv/test.json",
    "mawps": "mawps/test.json",
}


def load_dataset(data_root, name, limit=None):
    path = Path(data_root) / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- expected re_polar/datasets/{name}.py's output "
            f"(test.json) under --data-root"
        )
    data = json.loads(path.read_text())
    return data[:limit] if limit else data


def main(argv=None):
    # heavy imports INSIDE main() -> spawn grading workers stay light
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument(
        "--dataset", required=True, choices=sorted(DATASETS), help="which OOD eval set to grade"
    )
    p.add_argument(
        "--data-root",
        default="datasets",
        help="dir containing <dataset>/test.json (re_polar/datasets/{asdiv,mawps}.py's "
        "own output layout)",
    )
    p.add_argument("--output", required=True, help="path to write the results JSON")
    p.add_argument(
        "--prompt-style",
        default="paper_minimal_fewshot",
        help="GenerationReward prompt style (re_polar/mcts/rewards.py "
        "PROMPT_STYLES) -- default matches this project's own MCTS-search "
        "prompt, for direct comparability with existing DART-Math numbers "
        "on the same model. Pass 'raw' for PoLar's literal Appendix D.4 "
        "zero-shot protocol instead.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="first N questions (smoke tests)")
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}; known: {sorted(MODEL_REGISTRY)}")

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    reward_fn = GenerationReward(
        executor, batch_size=args.batch_size, prompt_style=args.prompt_style
    )  # boxed gate + strict grade
    identity = Program.identity(D)

    data = load_dataset(args.data_root, args.dataset, args.limit)
    questions = [d["question"] for d in data]
    gt = [d["gt_ans"] for d in data]
    print(
        f"[{args.dataset}] grading identity on {len(questions)} questions "
        f"(prompt_style={args.prompt_style})...",
        flush=True,
    )
    rewards = reward_fn(identity, questions, gt)
    acc = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[{args.dataset}] identity_acc = {acc:.4f}  (n={len(rewards)})", flush=True)

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "prompt_style": args.prompt_style,
        "n": len(rewards),
        "identity_acc": round(acc, 4),
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}", flush=True)
    return outp


if __name__ == "__main__":
    main()
