"""Identity-only option-shuffle check, over the WHOLE evaluation set (not
just RESCUE queries): as a control for `letter_bias_shuffle_robustness.py`'s
RESCUE-population reshuffle result, does reshuffling option order alone (no
program rerouting) move the base model's accuracy, and if not, does it still
move WHICH queries it solves? Written to run uniformly for any --model-id,
including re-deriving a model's own identity-wrong set fresh (rather than
reusing a precomputed flag) -- one shared method across models.

Steps:
  1. Runs identity-only forced-choice MC scoring (`re_polar/core/mmlu_pro_domain_eval.py`,
     the paper's own scoring method) on the full mmlu_pro_domains train+val
     (+test) pool, ORIGINAL option order, to find this model's own
     identity-WRONG set.
  2. Reshuffles the WHOLE population's options (same deterministic
     derangement as letter_bias_shuffle_robustness.py's shuffle_options, same
     seed convention) and reruns identity-only scoring on the shuffled
     prompts.
  3. Reports accuracy before/after AND the right->wrong / wrong->right flip
     rates: the interesting question is not just whether overall accuracy
     moves, but whether it stays flat while the SOLVED SET underneath it
     changes -- that flat-accuracy-but-churning-set pattern is the
     letter-bias signature, distinct from ordinary accuracy drift.

No per-query program rerouting here (this script is the identity-only
control; the program side is `letter_bias_shuffle_robustness.py`) -- both
passes are fully batched, so this is cheap (a single identity pass over the
whole pool, twice).

Pure-stdlib logic (reused from letter_bias_shuffle_robustness.py:
shuffle_options, build_query_id_index) needs no torch; this script's own new
logic (select_identity_wrong, aggregate) is unit-testable independently.

    python -m analysis.mmlu_pro_track.letter_bias_shuffle_identity_cross_model \\
        --model-id Qwen/Qwen3-8B \\
        --mmlu-data-dir mmlu_pro_domains \\
        --seed 0 \\
        --output letter_bias_shuffle_identity_qwen3_8b.jsonl
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

_SHUFFLE_MODULE_PATH = Path(__file__).resolve().parent / "letter_bias_shuffle_robustness.py"


def _load_shuffle_module():
    spec = importlib.util.spec_from_file_location(
        "letter_bias_shuffle_robustness", _SHUFFLE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_identity_wrong(query_ids: List[str], correct: List[bool]) -> List[str]:
    """[qid, ...] for every query where identity (original option order) scored wrong."""
    return [qid for qid, c in zip(query_ids, correct) if not c]


def _symmetric_stats(rows: List[dict]) -> dict:
    """Full before/after picture over the WHOLE population: does overall
    accuracy stay flat while the solved SET changes (the letter-bias
    signature) or does it move in a clear direction? Reports both directions
    of flip, not just wrong->right."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "original_acc": 0.0,
            "shuffled_acc": 0.0,
            "net_delta": 0.0,
            "n_originally_right": 0,
            "n_originally_wrong": 0,
            "n_right_to_wrong": 0,
            "n_wrong_to_right": 0,
            "right_to_wrong_rate": 0.0,
            "wrong_to_right_rate": 0.0,
        }
    orig_right = [r for r in rows if r["orig_correct"]]
    orig_wrong = [r for r in rows if not r["orig_correct"]]
    n_r2w = sum(1 for r in orig_right if not r["shuffled_correct"])
    n_w2r = sum(1 for r in orig_wrong if r["shuffled_correct"])
    original_acc = len(orig_right) / n
    shuffled_acc = sum(r["shuffled_correct"] for r in rows) / n
    return {
        "n": n,
        "original_acc": original_acc,
        "shuffled_acc": shuffled_acc,
        "net_delta": shuffled_acc - original_acc,
        "n_originally_right": len(orig_right),
        "n_originally_wrong": len(orig_wrong),
        "n_right_to_wrong": n_r2w,
        "n_wrong_to_right": n_w2r,
        "right_to_wrong_rate": n_r2w / len(orig_right) if orig_right else 0.0,
        "wrong_to_right_rate": n_w2r / len(orig_wrong) if orig_wrong else 0.0,
    }


def aggregate(records: List[dict]) -> dict:
    """records: [{"query_id", "num_options", "orig_correct", "shuffled_correct"}, ...]
    over the WHOLE population (both originally-right and originally-wrong)."""
    overall = _symmetric_stats(records)
    by_n: Dict[int, dict] = {}
    for n_opts in sorted({r["num_options"] for r in records}):
        by_n[n_opts] = _symmetric_stats([r for r in records if r["num_options"] == n_opts])
    return {"overall": overall, "by_num_options": by_n}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--mmlu-data-dir", default="mmlu_pro_domains")
    parser.add_argument(
        "--official-path",
        default="mmlu_pro_official/test.json",
        help="mmlu_pro_official test.json, used only if --mmlu-data-dir needs "
        "rebuilding (see below)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    shuf = _load_shuffle_module()

    data_dir = Path(args.mmlu_data_dir)
    if not (data_dir / "train.json").exists():
        print(
            f"{data_dir}/train.json not found -- rebuilding mmlu_pro_domains locally "
            f"from {args.official_path} (seed=42, val_frac=0.15, n_per_domain=500)...",
            flush=True,
        )
        from re_polar.datasets.mmlu_pro_domains import build_train_split

        build_train_split(
            data_dir,
            official_path=args.official_path,
            val_frac=0.15,
            n_per_domain=500,
            test_n_per_domain=200,
        )

    rows_by_split = {
        "train": json.loads((data_dir / "train.json").read_text()),
        "val": json.loads((data_dir / "val.json").read_text()),
    }
    if (data_dir / "test.json").exists():
        # Same as letter_bias_shuffle_robustness.py: include test.json if present so
        # "the whole dataset" really means all of it.
        rows_by_split["test"] = json.loads((data_dir / "test.json").read_text())
    index = shuf.build_query_id_index(rows_by_split)
    print(
        f"Loaded {len(index)} query_ids ("
        + " + ".join(f"{len(rows)} {s}" for s, rows in rows_by_split.items())
        + ")",
        flush=True,
    )

    # heavy imports inside main() (mirrors select_and_generate.py / run_mcts.py)
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core.mmlu_pro_domain_eval import (
        MMLUProSample,
        prepare_mmlu_pro_domain_inputs,
        run_mmlu_pro_domains,
    )

    engine = LayerEngine(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = engine.tokenizer
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    device = str(next(engine.model.parameters()).device)

    qids = sorted(index)  # deterministic order
    print(f"\n=== IDENTITY pass, ORIGINAL order, all {len(qids)} queries ===", flush=True)
    orig_samples = [
        MMLUProSample(
            id=qid,
            question=index[qid]["question"],
            options=tuple(index[qid]["options"]),
            answer_index=index[qid]["answer_index"],
            category=index[qid]["category"],
        )
        for qid in qids
    ]
    orig_inputs = prepare_mmlu_pro_domain_inputs(
        tokenizer, orig_samples, no_think=True, device=device
    )
    orig_result = run_mmlu_pro_domains(
        engine.model, tokenizer, prepared_inputs=orig_inputs, batch_size=args.batch_size
    )
    orig_correct = [bool(r["score"]) for r in orig_result["per_prompt"]]
    print(
        f"Identity ORIGINAL accuracy: {sum(orig_correct) / len(orig_correct):.1%} "
        f"({sum(orig_correct)}/{len(orig_correct)})",
        flush=True,
    )

    wrong_qids = select_identity_wrong(qids, orig_correct)
    print(
        f"Identity-WRONG population: {len(wrong_qids)}/{len(qids)} "
        f"({len(wrong_qids) / len(qids):.1%})",
        flush=True,
    )

    # Shuffle the WHOLE population, not just the wrong subset: if accuracy stays
    # ~flat but the SET of solved queries changes, that's a letter bias --
    # answerable only by seeing right->wrong flips (shuffling breaks previously-
    # correct queries) alongside wrong->right flips, not either alone.
    print(f"\n=== IDENTITY pass, SHUFFLED order, ALL {len(qids)} queries ===", flush=True)
    shuf_samples = []
    for qid in qids:
        info = index[qid]
        shuf_opts, shuf_ans = shuf.shuffle_options(
            info["options"], info["answer_index"], seed_key=f"{args.seed}:{qid}"
        )
        shuf_samples.append(
            MMLUProSample(
                id=qid,
                question=info["question"],
                options=tuple(shuf_opts),
                answer_index=shuf_ans,
                category=info["category"],
            )
        )
    shuf_inputs = prepare_mmlu_pro_domain_inputs(
        tokenizer, shuf_samples, no_think=True, device=device
    )
    shuf_result = run_mmlu_pro_domains(
        engine.model, tokenizer, prepared_inputs=shuf_inputs, batch_size=args.batch_size
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    with open(out_path, "w") as f:
        for qid, orig_ok, row in zip(qids, orig_correct, shuf_result["per_prompt"]):
            rec = {
                "query_id": qid,
                "num_options": len(index[qid]["options"]),
                "orig_correct": bool(orig_ok),
                "shuffled_correct": bool(row["score"]),
            }
            records.append(rec)
            f.write(json.dumps(rec) + "\n")

    summary = aggregate(records)
    summary["model_id"] = args.model_id
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        "\n=== SUMMARY (does accuracy stay flat while the SOLVED SET changes -- "
        "the letter-bias signature -- or does it move?) ===",
        flush=True,
    )
    print(f"model={args.model_id}", flush=True)
    o = summary["overall"]
    print(
        f"n={o['n']}  original_acc={o['original_acc']:.1%}  shuffled_acc={o['shuffled_acc']:.1%}  "
        f"net_delta={o['net_delta']*100:+.1f}pp",
        flush=True,
    )
    print(
        f"right->wrong (shuffle BROKE it): {o['n_right_to_wrong']}/{o['n_originally_right']} "
        f"({o['right_to_wrong_rate']:.1%})",
        flush=True,
    )
    print(
        f"wrong->right (shuffle FIXED it): {o['n_wrong_to_right']}/{o['n_originally_wrong']} "
        f"({o['wrong_to_right_rate']:.1%})",
        flush=True,
    )
    print("\nBy num_options:", flush=True)
    for n_opts, s in sorted(summary["by_num_options"].items()):
        print(
            f"  n_opts={n_opts:2d}  n={s['n']:4d}  orig={s['original_acc']:.1%}  "
            f"shuf={s['shuffled_acc']:.1%}  net={s['net_delta']*100:+.1f}pp  "
            f"r->w={s['right_to_wrong_rate']:.1%}  w->r={s['wrong_to_right_rate']:.1%}",
            flush=True,
        )
    print(f"\nWrote {len(records)} per-query records -> {out_path}", flush=True)
    print(f"Wrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
