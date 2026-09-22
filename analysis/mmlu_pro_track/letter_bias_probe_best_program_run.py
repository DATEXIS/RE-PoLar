"""Best-program-vs-identity letter-bias comparison on MMLU-Pro RESCUE samples.

For a subsample of RESCUE queries (identity/root program wrong, >=1 other
logged program correct) from the full mmlu_pro MCTS search
(`probs_qwen3_8b_mmlu.jsonl`, the `--log-probs` sidecar), apply each sample's
shortest correct program to Qwen3-8B and run the sampled-permutation
letter-bias probe (`letter_bias_probe_sampled.py`) under it, then compare to
an identity-program baseline probe run at the same `num_options`
(num_options is read directly off the log's own `probs` vector length --
verified to equal the source MMLU-Pro question's actual option count, no
separate dataset join needed).

This answers: does using the SPECIFIC program that rescues a sample change
the model's overall letter/position preference on an unrelated,
content-free prompt ("what's your favorite letter"), relative to running
identity?

NOTE: this main() is an exploratory diagnostic, not one of this track's
paper-cited results -- the paper's actual option-order finding is the
reshuffle-robustness check (`letter_bias_shuffle_robustness.py`), a
fundamentally different (and better-grounded) test: it re-scores the
program on the REAL question under a shuffled option order, rather than
probing an unrelated content-free prompt. Kept here (unmodified logic)
because `select_rescue_best_programs` below -- the RESCUE-selection
function -- is reused by both `select_rescue_population.py` and this
script; running this file's own main() is optional, not part of
reproducing any number in the paper.

Top level is stdlib-only (RESCUE selection / subsampling) so it's importable
and unit-testable without torch; heavy imports live inside main() (mirrors
`analysis/error_analysis/select_and_generate.py`).

    python -m analysis.mmlu_pro_track.letter_bias_probe_best_program_run \
        --probs-jsonl probs_qwen3_8b_mmlu.jsonl \
        --n-samples 200 --seed 0 --k-orderings 24 \
        --output letter_bias_best_program_run.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def is_identity_path(path: List[int]) -> bool:
    return list(path) == list(range(len(path)))


def select_rescue_best_programs(by_query: Dict[str, List[dict]],
                                 require_identity_wrong: bool = True) -> Dict[str, dict]:
    """{query_id: {"path": [...], "num_options": int, "identity_correct": bool}} for
    every query with identity present, at least one non-identity program logged
    correct (score==1.0); "best" = shortest such program (ties broken by whichever
    the log lists first, matching the repo's existing
    `min(final_valid_transitions, key=len)` convention).

    require_identity_wrong=True (default, unchanged): the RESCUE definition --
    identity present AND wrong (score==0.0). require_identity_wrong=False:
    drops that gate, broadening the population to also include samples
    identity already answers correctly but for which MCTS ALSO found a
    distinct valid program -- tests whether reshuffle-fragility is specific
    to the identity-wrong/rescue mechanism or general to any MCTS-discovered
    non-identity program (see `letter_bias_shuffle_robustness.py`'s
    `by_identity_orig_correct` stratum, the actual paper-cited use of this
    broadened population)."""
    result = {}
    for qid, records in by_query.items():
        identity = next((r for r in records if is_identity_path(r["path"])), None)
        if identity is None:
            continue
        if require_identity_wrong and identity["score"] != 0.0:
            continue
        correct = [r for r in records if r["score"] == 1.0 and not is_identity_path(r["path"])]
        if not correct:
            continue
        best = min(correct, key=lambda r: len(r["path"]))
        result[qid] = {"path": best["path"], "num_options": len(identity["probs"]),
                        "identity_correct": identity["score"] == 1.0}
    return result


def subsample(rescue: Dict[str, dict], n: int, seed: int) -> Dict[str, dict]:
    rng = random.Random(seed)
    keys = sorted(rescue)  # sort first for determinism regardless of dict/file iteration order
    rng.shuffle(keys)
    return {k: rescue[k] for k in keys[:n]}


def l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    assert set(a) == set(b)
    return sum(abs(a[k] - b[k]) for k in a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs-jsonl", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-orderings", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    by_query: Dict[str, List[dict]] = defaultdict(list)
    with open(args.probs_jsonl) as f:
        for line in f:
            r = json.loads(line)
            by_query[r["query_id"]].append(r)
    print(f"Loaded {sum(len(v) for v in by_query.values())} records across "
          f"{len(by_query)} distinct query_ids", flush=True)

    rescue = select_rescue_best_programs(by_query)
    print(f"RESCUE queries (identity wrong, >=1 program correct): {len(rescue)}", flush=True)

    chosen = subsample(rescue, args.n_samples, args.seed)
    print(f"Subsampled {len(chosen)} RESCUE queries (seed={args.seed})", flush=True)

    num_options_needed = sorted({v["num_options"] for v in chosen.values()})
    print(f"Distinct num_options in subsample: {num_options_needed}", flush=True)

    # heavy imports inside main() (mirrors select_and_generate.py / run_mcts.py)
    from re_polar.core.layer_engine import LayerEngine
    import importlib.util
    _probe_path = Path(__file__).resolve().parent / "letter_bias_probe_sampled.py"
    _spec = importlib.util.spec_from_file_location("letter_bias_probe_sampled", _probe_path)
    probe = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(probe)

    engine = LayerEngine(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = engine.tokenizer
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    identity_path = list(range(engine.num_layers))

    print("Computing identity baselines...", flush=True)
    identity_baseline: Dict[int, Dict[str, float]] = {}
    for num_options in num_options_needed:
        result = probe.run_letter_bias_probe(
            engine.model, tokenizer, num_options,
            k_orderings=args.k_orderings, seed=args.seed)
        identity_baseline[num_options] = result
        print(f"  identity num_options={num_options}: {result}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    l1_shifts: List[float] = []
    with open(out_path, "w") as f:
        for i, (qid, info) in enumerate(chosen.items()):
            engine.apply_layer_rerouting(info["path"])
            try:
                result = probe.run_letter_bias_probe(
                    engine.model, tokenizer, info["num_options"],
                    k_orderings=args.k_orderings, seed=args.seed)
            finally:
                engine.restore_original()

            baseline = identity_baseline[info["num_options"]]
            shift = l1_distance(result, baseline)
            l1_shifts.append(shift)
            rec = {
                "query_id": qid,
                "path": info["path"],
                "num_options": info["num_options"],
                "program_len": len(info["path"]),
                "program_dist": result,
                "identity_dist": baseline,
                "l1_shift_vs_identity": shift,
            }
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 20 == 0 or (i + 1) == len(chosen):
                print(f"probed {i + 1}/{len(chosen)} samples, running mean L1 shift so far: "
                      f"{sum(l1_shifts) / len(l1_shifts):.4f}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"n={len(l1_shifts)} RESCUE samples probed", flush=True)
    print(f"mean L1 shift (program vs. matched-num_options identity baseline): "
          f"{sum(l1_shifts) / len(l1_shifts):.4f}", flush=True)
    print(f"max possible L1 for a fully disjoint distribution: 2.0", flush=True)
    print(f"identity baselines used: {identity_baseline}", flush=True)


if __name__ == "__main__":
    main()
