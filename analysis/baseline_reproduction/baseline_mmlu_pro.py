"""Base(tau=0) MMLU-Pro baseline -- the paper's OOD protocol (Tables 3/8/9/10):
greedy (deterministic, single-forward-pass log-likelihood scoring over answer-
choice logits) accuracy per domain, per model. No sampling variant -- the
paper's own text: "Unless otherwise stated, OOD results are reported using
pass@1," and for MMLU-Pro specifically the tables carry exactly ONE row,
`Base (tau=0)`, no p@k breakdown at all -- there is no natural stochastic-
sampling analogue for a multiple-choice argmax-over-logits task the way
there is for DART-Math's free-generation answers.

Runs against the FULL official TIGER-Lab/MMLU-Pro test split (all 14
categories, no per-domain cap) pre-staged by `re_polar.datasets.mmlu_pro_official_test`
-- NOT the repo's own 500/domain `mmlu_pro_domains` train/val/test split
(that split is for MCTS/router supervision, a different track entirely --
see `analysis/mmlu_pro_track/`).

Uses `re_polar/core/mmlu_pro_domain_eval.py`'s `run_mmlu_pro_domains` directly
rather than going through `LogLikReward.__call__` -- that function already
returns a native per-domain breakdown (`result["per_domain"]`), so there's
no need to re-derive it by grouping per-sample scores ourselves.

  python -m analysis.baseline_reproduction.baseline_mmlu_pro \
      --model qwen3_8b --data-dir mmlu_pro_official \
      --output baseline_mmlu_pro_qwen3_8b.json
"""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.core import Program
    from re_polar.core.mmlu_pro_scoring import MMLUProSample
    from re_polar.core.mmlu_pro_domain_eval import run_mmlu_pro_domains

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument(
        "--data-dir", required=True, help="dir with test.json from mmlu_pro_official_test"
    )
    p.add_argument(
        "--domains",
        default=None,
        help="comma-separated category filter; default = all categories present in the data",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="logits_to_keep=1 for this scorer -> memory dominated by weights, "
        "not logits, so a larger batch than a generation task is fine",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    cfg = MODEL_REGISTRY[args.model]
    no_think = cfg.get("no_think", False)

    records = json.loads((Path(args.data_dir) / "test.json").read_text())
    if args.domains:
        wanted = {d.strip() for d in args.domains.split(",")}
        records = [r for r in records if r["category"] in wanted]
    samples = [
        MMLUProSample(
            id=i,
            question=str(r["question"]),
            options=tuple(r["options"]),
            answer_index=int(r["answer_index"]),
            category=str(r["category"]),
        )
        for i, r in enumerate(records)
    ]
    print(
        f"[data] {len(samples)} questions, {len(set(s.category for s in samples))} categories, "
        f"batch_size={args.batch_size}, no_think={no_think}",
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
        "no_think": no_think,
        "data_dir": str(args.data_dir),
        "num_layers": D,
        "n": len(samples),
    }
    print(f"[env] {run_env}", flush=True)

    with executor.apply(identity) as model:
        result = run_mmlu_pro_domains(
            model,
            engine.tokenizer,
            samples=samples,
            batch_size=args.batch_size,
            no_think=no_think,
            return_details=False,
        )

    out = {
        "model": args.model,
        "env": run_env,
        "base_greedy_average": round(result["average"], 4),
        "per_domain": {
            d: {"average": round(v["average"], 4), "n": v["n"]}
            for d, v in sorted(result["per_domain"].items())
        },
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}", flush=True)

    print(f"\n{'domain':>20} {'n':>6} {'Base(tau=0)':>12}")
    for d, v in out["per_domain"].items():
        print(f"{d:>20} {v['n']:>6} {v['average']:>12.4f}")
    print(f"{'ALL':>20} {result['n']:>6} {out['base_greedy_average']:>12.4f}")
    return outp


if __name__ == "__main__":
    main()
