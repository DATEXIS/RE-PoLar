"""Compute two summary tables for the MMLU-Pro track directly from the
underlying result JSON files, instead of transcribing numbers by hand --
transcribing by hand is where errors creep in.

Reshuffle strata: Original vs Shuffled accuracy for the 4 populations
described in `letter_bias_shuffle_robustness.py`'s module docstring
(identity control; any-program + identity-correct; any-program + RESCUE;
RESCUE-only dedicated selection).

Router vs random baseline: pass@5 comparing identity, the 5-seed
random-valid-program baseline (mean +/- std), and the winning router
recipe's 5-seed replication (mean +/- std, expected ~0 std -- see
`infer_router_mmlu.py`'s docstring for why).

Inputs are small JSON files -- see --data-dir and each loader's docstring
for the exact expected filename.

    python -m analysis.mmlu_pro_track.summarize_mmlu_reshuffle_and_router_baseline \\
        --data-dir mmlu_router_figures
"""

import argparse
import json
import statistics
from pathlib import Path


def load_reshuffle_strata(data_dir: Path):
    """4 rows for the reshuffle-strata table, from 3 summary.json files:
    identity_control.json (letter_bias_shuffle_identity_cross_model.py's output
    for Qwen3-8B), rescue_only.json (letter_bias_shuffle_robustness.py's output,
    RESCUE-only run), broad.json (letter_bias_shuffle_robustness.py's output,
    --include-identity-correct run)."""
    identity = json.loads((data_dir / "identity_control.json").read_text())["overall"]
    rescue = json.loads((data_dir / "rescue_only.json").read_text())["overall"]
    broad = json.loads((data_dir / "broad.json").read_text())["by_identity_orig_correct"]

    return [
        (
            "Identity control (whole dataset)",
            identity["n"],
            identity["original_acc"],
            identity["shuffled_acc"],
        ),
        (
            "Any program, identity correct",
            broad["identity_orig_correct"]["n"],
            broad["identity_orig_correct"]["program_orig_acc"],
            broad["identity_orig_correct"]["program_shuffled_acc"],
        ),
        (
            "Any program, identity wrong (RESCUE)",
            broad["identity_orig_wrong"]["n"],
            broad["identity_orig_wrong"]["program_orig_acc"],
            broad["identity_orig_wrong"]["program_shuffled_acc"],
        ),
        (
            "RESCUE only (dedicated)",
            rescue["n"],
            rescue["program_orig_acc"],
            rescue["program_shuffled_acc"],
        ),
    ]


def load_router_vs_random(data_dir: Path):
    """pass@5 identity / random-baseline (mean+/-std) / winning-recipe router
    (mean+/-std, computed here from the 5 seed test-eval JSONs rather than
    asserted, so this would visibly show if the 5 seeds ever stopped
    agreeing) -- router_random_baseline.json and winning_recipe_seed{0..4}.json,
    all `infer_router_mmlu.py --random-baseline-seeds ... --top-k-paths 5` output."""
    rb = json.loads((data_dir / "router_random_baseline.json").read_text())
    identity_acc = rb["overall"]["identity_acc"]
    random_mean_std = rb["random_baseline"]["mean_std"]["5"]

    seed_accs = []
    for i in range(5):
        d = json.loads((data_dir / f"winning_recipe_seed{i}.json").read_text())
        seed_accs.append(d["overall"]["router_acc_at_k"])
    router_mean = statistics.mean(seed_accs)
    router_std = statistics.stdev(seed_accs) if len(set(seed_accs)) > 1 else 0.0

    return {
        "identity": identity_acc,
        "random_mean": random_mean_std["mean"],
        "random_std": random_mean_std["std"],
        "router_mean": router_mean,
        "router_std": router_std,
        "router_seed_accs": seed_accs,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True, help="dir with the input result JSON files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    rows = load_reshuffle_strata(data_dir)
    print("Reshuffle strata:")
    for label, n, o, s in rows:
        print(f"  {label}: n={n} orig={o:.4f} shuffled={s:.4f}")

    rv = load_router_vs_random(data_dir)
    print("\nRouter vs random baseline:")
    print(f"  identity={rv['identity']:.4f}")
    print(f"  random baseline: mean={rv['random_mean']:.4f} std={rv['random_std']:.4f}")
    print(f"  router (winning recipe), 5 seeds: {rv['router_seed_accs']}")
    print(f"  router mean={rv['router_mean']:.4f} std={rv['router_std']:.4f}")


if __name__ == "__main__":
    main()
