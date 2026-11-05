"""Held-out check of menu coverage (paper Finding 5): in-sample against held-out menus.

Reads the grids of analysis.mcts_design.menu_coverage for every model and level:
  in-sample  data/menu_crossexec/<model>/diff<L>.json.gz          ranked and scored on all
             2,000 questions (--rank-pool all --eval-pool all; paper Fig. 4a and Fig. 9)
  held-out   data/menu_crossexec_heldout/<model>/diff<L>.json.gz  ranked on the 1,500
             train+val questions, scored on the 500 test questions
             (--rank-pool trainval --eval-pool test)
and prints, per model and level and as the mean over all of them, identity accuracy, pass@K
(share of the scored questions that at least one of the first K programs solves) and the
ratio of mean pass@100 to mean identity accuracy. CPU only, a few seconds.

  python -m analysis.mcts_design.menu_heldout --data-dir ./data
"""

import argparse
import gzip
import json
from pathlib import Path

from re_polar.models import MODEL_REGISTRY

MODELS = ("qwen15_moe_a27b", "qwen25_3b", "qwen25_7b", "qwen3_8b", "qwen3_32b", "llama32_3b")
KS = (1, 5, 10, 25, 100)
GRIDS = {"in-sample": "menu_crossexec", "held-out": "menu_crossexec_heldout"}


def menu_stats(grid: dict, num_layers: int, ks=KS) -> dict:
    """Identity accuracy and pass@K of one grid, programs taken in menu order."""
    (pd,) = grid["per_difficulty"]
    progs = pd["per_program"]
    (ident,) = [p for p in progs if p["path"] == list(range(num_layers))]
    n = len(pd["qids"])
    solved = [0] * n
    curve = {}
    for k, p in enumerate(progs, 1):
        solved = [a or b for a, b in zip(solved, p["rewards"])]
        if k in ks:
            curve[k] = sum(solved) / n
    return {"identity": sum(ident["rewards"]) / n, "pass_at": curve, "n": n}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--output", default=None, help="optional JSON with every number printed")
    args = p.parse_args(argv)
    models = args.models.split(",")

    res = {}
    for setting, sub in GRIDS.items():
        for i, m in enumerate(models, 1):
            for lvl in range(1, 6):
                path = Path(args.data_dir) / sub / m / f"diff{lvl}.json.gz"
                with gzip.open(path, "rt") as f:
                    s = menu_stats(json.load(f), MODEL_REGISTRY[m]["num_layers"])
                res[(setting, m, lvl)] = s
                print(
                    f"[{setting} {i}/{len(models)}] {m} DM-{lvl} ({s['n']} questions): identity "
                    f"{100 * s['identity']:.1f}, pass@100 {100 * s['pass_at'][100]:.1f}",
                    flush=True,
                )

    print(f"\nSUMMARY, mean over {len(models)} models x 5 levels (%):")
    print(f"  {'':10s} {'identity':>8s} " + " ".join(f"{'@' + str(k):>6s}" for k in KS) + "  ratio@100")
    summary = {}
    for setting in GRIDS:
        cells = [v for (s, _, _), v in res.items() if s == setting]
        ident = 100 * sum(c["identity"] for c in cells) / len(cells)
        at = {k: 100 * sum(c["pass_at"][k] for c in cells) / len(cells) for k in KS}
        summary[setting] = {"identity": ident, "pass_at": at, "ratio_at_100": at[100] / ident}
        print(
            f"  {setting:10s} {ident:8.1f} "
            + " ".join(f"{at[k]:6.1f}" for k in KS)
            + f"  {at[100] / ident:.2f}x"
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "summary": summary,
                    "per_cell": {f"{s}|{m}|{lvl}": v for (s, m, lvl), v in res.items()},
                },
                indent=1,
            )
        )
        print(f"written to {args.output}")


if __name__ == "__main__":
    main()
