"""
Entry point for the layer-duplication / skip sweep behind the paper's
motivation-section figure on repeat/skip accuracy by layer block.

Drives re_polar.core.layer_engine.LayerEngine over the mmlu_pro_domains benchmark
(the paper's own 6-domain MMLU-Pro subset) -- see prestudy/sweep.py for the
sweep mechanics.

Usage examples:
  # Qwen3-8B full dup+skip sweep (the paper's own figure)
  python -m prestudy.run --model qwen3_8b --output-dir ./prestudy/results

  # Duplication only
  python -m prestudy.run --model qwen3_8b --mode dup

  # Quick local smoke (2 random configs)
  python -m prestudy.run --model qwen3_8b --quick-test 2 --output-dir ./scratch/smoke

  # Subsample a big sweep (every 4th config) and/or shard across workers
  python -m prestudy.run --model qwen3_32b --stride 4
  python -m prestudy.run --model qwen3_8b --num-shards 4 --shard-index 0

  # Pre-supply the baseline score (skip re-measurement)
  python -m prestudy.run --model qwen3_8b --baseline 0.71
"""

import argparse
from typing import List, Optional, Tuple

from re_polar.core.layer_engine import LayerEngine
from re_polar.models import MODEL_REGISTRY
from prestudy.sweep import LayerDuplicationSweep, VALID_MODES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer duplication / skip sweep: baseline vs rerouted (i, j) configs."
    )
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), help="Use a registered model config"
    )
    model_group.add_argument("--model-id", help="Arbitrary HuggingFace model id")
    p.add_argument("--output-dir", default="./prestudy/results")
    p.add_argument(
        "--mode",
        choices=["dup", "skip", "both"],
        default="both",
        help="Which rerouting configs to sweep (default: both)",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Pre-computed mmlu_pro_domains baseline score, e.g. 0.71",
    )
    p.add_argument(
        "--quick-test", type=int, default=0, help="Run N random configs instead of the full sweep"
    )
    p.add_argument(
        "--configs-file",
        default=None,
        help="JSON file with a list of [mode, i, j] configs to run, bypassing the "
        'full sweep. Accepts a bare list or {"configs": [...]}. '
        "Honors --num-shards/--shard-index.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every STRIDE-th config per mode (subsample a big sweep)",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--no-pretokenize", action="store_true", help="Disable prompt pre-tokenization")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="mmlu_pro_domains forward-pass batch size (default: 8)",
    )
    p.add_argument(
        "--n-per-domain",
        type=int,
        default=200,
        help="mmlu_pro_domains samples per domain (paper default: 200; -1=all available)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.model:
        cfg = MODEL_REGISTRY[args.model]
        model_id = cfg["model_id"]
        trust_remote_code = cfg.get("trust_remote_code", True)
        no_think = cfg.get("no_think", True)
    else:
        model_id = args.model_id
        trust_remote_code = True
        no_think = True

    batch_size = args.batch_size or 8
    modes = VALID_MODES if args.mode == "both" else (args.mode,)

    explicit_configs: Optional[List[Tuple[str, int, int]]] = None
    if args.configs_file:
        import json
        from pathlib import Path

        raw = json.loads(Path(args.configs_file).read_text(encoding="utf-8"))
        items = raw["configs"] if isinstance(raw, dict) else raw
        explicit_configs = [(str(c[0]), int(c[1]), int(c[2])) for c in items]
        print(f"Loaded {len(explicit_configs)} explicit configs from {args.configs_file}")

    engine = LayerEngine(model_id, trust_remote_code=trust_remote_code)
    sweep = LayerDuplicationSweep(
        engine=engine,
        output_dir=args.output_dir,
        modes=modes,
        n_per_domain=args.n_per_domain,
        batch_size=batch_size,
        no_think=no_think,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        stride=args.stride,
        pretokenize=not args.no_pretokenize,
        explicit_configs=explicit_configs,
    )
    baseline = {"mmlu_pro_domains": args.baseline} if args.baseline is not None else None
    sweep.run(baseline_scores=baseline, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
