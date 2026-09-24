"""FAIR pass@k baselines for the router comparison.

Three numbers per difficulty on the DART-Math TEST splits, all with the same
grader/prompt as MCTS + infer:

  * Base greedy p@1       , identity, greedy (== baseline_identity; reference).
  * Base (sampling) p@k   , identity, k stochastic samples at each temperature in
                             --temperatures, pass@k per temperature AND best across
                             temperatures (the paper's "Base (sampling)" baseline).
  * Menu p@k              , the router's near-fixed 5-program menu run on EVERY
                             question (greedy), pass@k = any solves. If this ~=
                             router pass@k, the router adds nothing over a hardcoded
                             menu. (D>=32; built from the observed Qwen menu.)

The point: router pass@5 must beat BOTH Base(sampling) p@5 and Menu p@5 to be a
real, adaptive result rather than a best-of-k / fixed-ensemble artifact.

Also used, with different `--prompt-style` flags, for the PoLar-literal
reproduction track (each model's PoLar-exact chat-template mechanism, see
`aggregate_baseline_tables.py`'s `build_literal_repro`) -- same script, no
separate driver needed.

  python -m analysis.baseline_reproduction.baseline_passk \
      --model qwen3_8b --data-dir dart_math_v2 \
      --difficulty all --k 5 --temperatures 0.3,0.7,1.0 \
      --output baseline_passk_qwen3_8b.json

Top level is stdlib-only so the spawn grading workers stay torch-free.
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_test_split(data_dir, diff, limit=None):
    data = json.loads((Path(data_dir) / f"diff{diff}" / "test.json").read_text())
    return data[:limit] if limit else data


def resolve_difficulties(vals):
    if not vals or "all" in vals:
        return [1, 2, 3, 4, 5]
    return sorted(int(v) for v in vals)


def menu_paths(D):
    """The observed near-universal router menu as executed layer-paths (D>=32)."""
    return {
        "identity": list(range(D)),
        "skip_tail4": list(range(D - 4)),
        "repeat_8_12": list(range(12)) + [8, 9, 10, 11] + list(range(12, D)),
        "repeat_12_16": list(range(16)) + [12, 13, 14, 15] + list(range(16, D)),
        "repeat_tail": list(range(D - 4)) + [D - 8, D - 7, D - 6, D - 5] + list(range(D - 4, D)),
    }


def main(argv=None):
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program
    from re_polar.router.train import program_from_layer_path

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--difficulty", action="append", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k", type=int, default=5, help="samples per temperature / menu size cap")
    p.add_argument(
        "--temperatures",
        default="0.3,0.7,1.0",
        help="comma-separated sampling temperatures for Base(sampling)",
    )
    p.add_argument(
        "--prompt-style",
        default="paper_minimal_fewshot",
        help="passed through to GenerationReward -- default matches this "
        "project's own canonical MCTS-search prompt choice; the paper's "
        "literal Appendix D.4 zero-shot prompt is 'raw' (pass "
        "--prompt-style raw / paper_chat_sys for the PoLar-literal "
        "reproduction track, see module docstring)",
    )
    p.add_argument("--no-menu", action="store_true", help="skip the hardcoded-menu control")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}")
    diffs = resolve_difficulties(args.difficulty)
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    reward = GenerationReward(executor, batch_size=args.batch_size, prompt_style=args.prompt_style)
    identity = Program.identity(D)

    # Environment metadata -- so every result JSON is self-describing (which GPU
    # this ran on, batch size, prompt style) without re-deriving it from a
    # filename convention.
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu_name = "unknown"
    run_env = {
        "gpu_name": gpu_name,
        "batch_size": args.batch_size,
        "prompt_style": args.prompt_style,
        "data_dir": str(args.data_dir),
        "num_layers": D,
    }
    print(f"[env] {run_env}", flush=True)
    menu = (
        None
        if args.no_menu
        else {name: program_from_layer_path(path, D) for name, path in menu_paths(D).items()}
    )

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    results = []
    for diff in diffs:
        data = load_test_split(args.data_dir, diff, args.limit)
        qs = [d["question"] for d in data]
        gt = [d["gt_ans"] for d in data]
        n = len(qs)
        print(f"[diff {diff}] {n} questions...", flush=True)

        greedy1 = mean(reward(identity, qs, gt))  # Base (τ=0) p@1

        # FULL pass@k curve (k=1..args.k) per temperature, from ONE k=args.k
        # generation batch per temp -- passk_curve's own docstring covers why a
        # single-k call is not enough: the sampling p@1 row is a DIFFERENT number
        # from Base(τ=0) p@1, not a duplicate (paper Table 7 confirms, e.g. DM-1:
        # greedy p@1=41.6 vs sampling p@1=37.0).
        per_temp_curve: Dict[float, Dict[int, float]] = {}
        for t in temps:
            curve = reward.passk_curve(identity, qs, gt, k_max=args.k, temperature=t)
            per_temp_curve[t] = {k: mean(v) for k, v in curve.items()}
            print(
                f"  base sampling p@1..{args.k} T={t}: "
                + " ".join(f"p@{k}={per_temp_curve[t][k]:.4f}" for k in range(1, args.k + 1)),
                flush=True,
            )
        # best-across-temperature AT EACH k independently (matches the paper's
        # monotonic-in-k table -- taking one fixed "best" temperature for the whole
        # row would NOT generally be monotonic in k across temps).
        best_per_k = {k: max(per_temp_curve[t][k] for t in temps) for k in range(1, args.k + 1)}

        menu_passk = None
        if menu is not None:
            any_solved = [0.0] * n
            for name, prog in menu.items():
                r = reward(prog, qs, gt)
                any_solved = [max(a, b) for a, b in zip(any_solved, r)]
            menu_passk = mean(any_solved)
            print(f"  menu p@{len(menu)}: {menu_passk:.4f}", flush=True)

        row = {
            "difficulty": diff,
            "n": n,
            "base_greedy_p1": round(greedy1, 4),
            "base_sampling_curve_per_temp": {
                str(t): {str(k): round(a, 4) for k, a in curve.items()}
                for t, curve in per_temp_curve.items()
            },
            "base_sampling_pk_best_per_k": {str(k): round(a, 4) for k, a in best_per_k.items()},
            "base_sampling_p1_best": round(best_per_k[1], 4),
            "base_sampling_pk_best": round(best_per_k[args.k], 4),  # back-compat name, == p@k_max
            "menu_pk": None if menu_passk is None else round(menu_passk, 4),
        }
        results.append(row)
        print(
            f"[diff {diff}] greedy@1 {greedy1:.4f} | base_sampling@1(best) {best_per_k[1]:.4f} | "
            f"base_sampling@{args.k}(best) {best_per_k[args.k]:.4f} | menu@{args.k} "
            f"{'n/a' if menu_passk is None else f'{menu_passk:.4f}'}",
            flush=True,
        )

    out = {
        "model": args.model,
        "k": args.k,
        "temperatures": temps,
        "env": run_env,
        "per_difficulty": results,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}", flush=True)

    print(f"\n{'diff':>4} {'greedy@1':>9} {'baseSamp@'+str(args.k):>11} {'menu@'+str(args.k):>9}")
    for r in results:
        menu_str = "n/a" if r["menu_pk"] is None else f"{r['menu_pk']:.4f}"
        print(
            f"{r['difficulty']:>4} {r['base_greedy_p1']:>9.4f} "
            f"{r['base_sampling_pk_best']:>11.4f} {menu_str:>9}"
        )
    return outp


if __name__ == "__main__":
    main()
