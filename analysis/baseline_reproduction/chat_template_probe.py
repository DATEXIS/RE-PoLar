"""Chat-template probe: does wrapping the prompt in the model's own chat
template change DART-Math baseline accuracy, holding prompt CONTENT fixed at
`paper_minimal_fewshot`?

Motivation: this project's default (`raw`, no chat template) was set from
ONE finding on Qwen3-8B specifically (`re_polar/mcts/rewards.py`'s
`default_prompt` docstring: chat template induces CoT narration that eats
the 50-token budget before `\\boxed{}`) and extrapolated to every other
model without re-testing. PoLar's own released code applies a chat template
(hardcoded to Qwen3), and the paper text never says either way. This checks
whether "raw beats chat" is a Qwen3-8B-specific artifact or holds
project-wide.

Cheap diagnostic, NOT a replacement for the main baseline table: ONE random,
seed-42, POOLED 200-question subsample across all 5 DART-Math difficulties
(same 200 questions for every model, for a clean cross-model comparison), full
6-row protocol (Base(tau=0) + Base(sampling) p@1..5, same best-across-temperature
method as baseline_passk.py) under BOTH `paper_minimal_fewshot` (raw) and
`paper_minimal_fewshot_chat` (same content, chat-template-wrapped) -- one model
load, both conditions, so grading conditions are identical between them.

  python -m analysis.baseline_reproduction.chat_template_probe \
      --model qwen3_8b --data-dir dart_math_v2 \
      --output chat_template_probe_qwen3_8b.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict


def load_pooled_sample(data_dir, n, seed):
    """Pools all 5 difficulties' TEST rows together, then draws ONE random,
    seed-fixed n-row sample -- same 200 questions for every model (data pool is
    model-agnostic), so cross-model comparisons of the chat-template delta aren't
    confounded by different questions per model."""
    pool = []
    for diff in range(1, 6):
        rows = json.loads((Path(data_dir) / f"diff{diff}" / "test.json").read_text())
        for r in rows:
            pool.append({**r, "difficulty": diff})
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def main(argv=None):
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperatures", default="0.3,0.7,1.0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--chat-prompt-style",
        default="paper_minimal_fewshot_chat",
        help="which PROMPT_STYLES entry to use for the 'chat' condition -- "
        "default is the no-system-message variant (faithful to PoLar's "
        "own Qwen3 branch); pass 'paper_minimal_fewshot_chat_sys' for "
        "the system-message variant (faithful to PoLar's Qwen2.5-Instruct/ "
        "Qwen1.5-MoE-Chat branches -- see that prompt's docstring)",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    cfg = MODEL_REGISTRY[args.model]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]

    sample = load_pooled_sample(args.data_dir, args.n, args.seed)
    qs = [d["question"] for d in sample]
    gt = [d["gt_ans"] for d in sample]
    print(
        f"[data] pooled sample n={len(sample)} (seed={args.seed}), "
        f"difficulty mix={sorted(set(d['difficulty'] for d in sample))}",
        flush=True,
    )

    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    identity = Program.identity(D)

    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu_name = "unknown"
    run_env = {
        "gpu_name": gpu_name,
        "batch_size": args.batch_size,
        "n": len(sample),
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "num_layers": D,
        "chat_prompt_style": args.chat_prompt_style,
    }
    print(f"[env] {run_env}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    conditions = {}
    for label, style in [("raw", "paper_minimal_fewshot"), ("chat", args.chat_prompt_style)]:
        print(f"\n=== condition: {label} (prompt_style={style}) ===", flush=True)
        reward = GenerationReward(executor, batch_size=args.batch_size, prompt_style=style)

        greedy1 = mean(reward(identity, qs, gt))
        per_temp_curve: Dict[float, Dict[int, float]] = {}
        for t in temps:
            curve = reward.passk_curve(identity, qs, gt, k_max=args.k, temperature=t)
            per_temp_curve[t] = {k: mean(v) for k, v in curve.items()}
        best_per_k = {k: max(per_temp_curve[t][k] for t in temps) for k in range(1, args.k + 1)}

        conditions[label] = {
            "prompt_style": style,
            "base_greedy_tau0": round(greedy1, 4),
            "base_sampling_pk_best_per_k": {str(k): round(a, 4) for k, a in best_per_k.items()},
        }
        print(
            f"  tau=0={greedy1:.4f}  "
            + " ".join(f"p@{k}={best_per_k[k]:.4f}" for k in range(1, args.k + 1)),
            flush=True,
        )

    delta = {
        "tau0": round(
            conditions["chat"]["base_greedy_tau0"] - conditions["raw"]["base_greedy_tau0"], 4
        ),
        **{
            f"p{k}": round(
                conditions["chat"]["base_sampling_pk_best_per_k"][str(k)]
                - conditions["raw"]["base_sampling_pk_best_per_k"][str(k)],
                4,
            )
            for k in range(1, args.k + 1)
        },
    }

    out = {
        "model": args.model,
        "env": run_env,
        "conditions": conditions,
        "delta_chat_minus_raw": delta,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}", flush=True)

    print(f"\n{'':>6} {'tau=0':>7} {'p@1':>7} {'p@2':>7} {'p@3':>7} {'p@4':>7} {'p@5':>7}")
    for label in ("raw", "chat"):
        c = conditions[label]
        bk = c["base_sampling_pk_best_per_k"]
        print(
            f"{label:>6} {c['base_greedy_tau0']:>7.3f} "
            + " ".join(f"{bk[str(k)]:>7.3f}" for k in range(1, args.k + 1))
        )
    print(
        f"{'delta':>6} {delta['tau0']:>7.3f} "
        + " ".join(f"{delta[f'p{k}']:>7.3f}" for k in range(1, args.k + 1))
    )
    return outp


if __name__ == "__main__":
    main()
