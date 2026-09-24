"""Letter-bias shuffle-robustness check on MMLU-Pro RESCUE queries: does
reshuffling a question's option ORDER (content unchanged, just permuting
which option sits behind which letter) change whether identity / the rescue
program get the query right? If baseline itself has a bias toward a certain
letter and the program merely fixes that instead of the actual question,
this should surface it directly: compare the set of solved samples between
identity and identity-shuffled, and same for program vs. program-shuffled.

Two comparisons, same RESCUE population (identity wrong, a program correct,
both under the ORIGINAL option order -- built by `select_rescue_population.py`):

  1. IDENTITY, original order (all wrong by construction of RESCUE) vs.
     IDENTITY, options reshuffled (deterministic per-query derangement,
     content unchanged, only which option sits behind which letter moves).
     If a meaningful fraction newly becomes correct just from reordering, the
     original "wrong" answer was at least partly a letter/position artifact,
     not a pure reasoning failure.
  2. The query's own best RESCUE PROGRAM (same skip/repeat layer edits),
     original order (~100% correct by construction) vs. reshuffled. If the
     program's fix survives reshuffling far more often than identity alone
     gains from shuffling, that's evidence the program does something beyond
     exploiting the original option arrangement.

Recomputes IDENTITY-original and PROGRAM-original fresh in this same process
(not reused from a separately-computed label) so all four numbers per query
come from the SAME hardware in the SAME run -- greedy/logit-readout scoring
is known to diverge in its last-bit numerics across GPU architectures, so
mixing a label computed on one machine with one computed on another would be
an apples-to-oranges comparison. A handful of freshly-computed original
scores may disagree with a separately-stored label on close ties; that's
expected and reported, not a bug.

Reuses the paper's own forced-choice logit-readout scoring (no generation)
verbatim: `re_polar/core/mmlu_pro_domain_eval.py`'s `MMLUProSample` /
`prepare_mmlu_pro_domain_inputs` / `run_mmlu_pro_domains` -- only the option ORDER passed into `MMLUProSample.options`
differs between the two prompt variants, the scoring method itself is
untouched.

Data: rebuilds the mmlu_pro_domains train+val(+test) split locally
(`re_polar.datasets.mmlu_pro_domains.build_train_split`, seed=42, val_frac=0.15,
n_per_domain=500, sourced from the local mmlu_pro_official pool) to join
RESCUE query_ids (`mmlu-{split}-{id}`) back to real question/options/
answer_index. No network needed once mmlu_pro_official is available locally.

Pure-stdlib functions (shuffle_options, joining, aggregation) are
unit-testable without torch; heavy imports live inside main() (mirrors
`analysis/error_analysis/select_and_generate.py`).

    python -m analysis.mmlu_pro_track.letter_bias_shuffle_robustness \\
        --rescue-jsonl rescue_population_qwen3_8b.jsonl \\
        --mmlu-data-dir mmlu_pro_domains \\
        --full --seed 0 \\
        --output letter_bias_shuffle_robustness.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LETTERS = "ABCDEFGHIJ"


def load_rescue(path: str) -> Dict[str, dict]:
    """{query_id: {"path": [...], "num_options": int}} straight off the
    already-computed RESCUE population file (see
    letter_bias_probe_best_program_run.py::select_rescue_best_programs for
    how it was produced, via select_rescue_population.py -- no need to redo
    that selection here)."""
    rescue: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rescue[r["query_id"]] = {"path": r["path"], "num_options": r["num_options"]}
    return rescue


def subsample(rescue: Dict[str, dict], n: int, seed: int) -> Dict[str, dict]:
    rng = random.Random(seed)
    keys = sorted(rescue)  # sort first for determinism regardless of dict/file iteration order
    rng.shuffle(keys)
    return {k: rescue[k] for k in keys[:n]}


def build_query_id_index(rows_by_split: Dict[str, List[dict]]) -> Dict[str, dict]:
    """{query_id: row} for {"train": [...], "val": [...], "test": [...]} split
    lists, using the exact `mmlu-{split}-{id}` convention the MCTS input
    loader uses."""
    index: Dict[str, dict] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            index[f"mmlu-{split}-{row['id']}"] = row
    return index


def join_rescue_with_mmlu(rescue: Dict[str, dict], index: Dict[str, dict]) -> Dict[str, dict]:
    """Merge each rescue entry with its real question/options/answer_index,
    dropping (and reporting via the caller) any query_id not found in the
    rebuilt split -- should be empty if the split was rebuilt with the exact
    params the released split used, but never silently assumed."""
    joined = {}
    for qid, info in rescue.items():
        row = index.get(qid)
        if row is None:
            continue
        if len(row["options"]) != info["num_options"]:
            raise AssertionError(
                f"{qid}: rescue file says num_options={info['num_options']} but "
                f"rebuilt split has {len(row['options'])} options -- split mismatch, "
                f"do not trust this join"
            )
        joined[qid] = {
            **info,
            "question": row["question"],
            "options": row["options"],
            "answer_index": row["answer_index"],
            "category": row["category"],
        }
    return joined


def shuffle_options(options: List[str], answer_index: int, seed_key: str) -> Tuple[List[str], int]:
    """Deterministic, per-query derangement of `options` (never returns the
    original order -- rejection-sampled, cheap since num_options <= 10).
    Returns (shuffled_options, new_answer_index) so the correct option's
    CONTENT is preserved, only its position/letter moves."""
    n = len(options)
    if n < 2:
        return list(options), answer_index
    rng = random.Random(seed_key)
    order = list(range(n))
    while True:
        rng.shuffle(order)
        if order != list(range(n)):
            break
    shuffled = [options[i] for i in order]
    new_answer_index = order.index(answer_index)
    return shuffled, new_answer_index


def aggregate(records: List[dict]) -> dict:
    """Summary stats over per-query result records (see main()'s `rec` shape).
    Solved-set sizes/rates for identity/program x original/shuffled, plus
    Jaccard overlap between the two SHUFFLED solved-sets, broken down overall
    and by num_options."""

    def rate(key: str, rows: List[dict]) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    def jaccard(rows: List[dict], key_a: str, key_b: str) -> Optional[float]:
        a = {r["query_id"] for r in rows if r[key_a]}
        b = {r["query_id"] for r in rows if r[key_b]}
        if not a and not b:
            return None
        return len(a & b) / len(a | b)

    overall = {
        "n": len(records),
        "identity_orig_acc": rate("identity_orig_correct", records),
        "identity_shuffled_acc": rate("identity_shuffled_correct", records),
        "program_orig_acc": rate("program_orig_correct", records),
        "program_shuffled_acc": rate("program_shuffled_correct", records),
        "jaccard_identity_shuffled_vs_program_shuffled": jaccard(
            records, "identity_shuffled_correct", "program_shuffled_correct"
        ),
    }

    by_n: Dict[int, dict] = {}
    num_options_vals = sorted({r["num_options"] for r in records})
    for n_opts in num_options_vals:
        rows = [r for r in records if r["num_options"] == n_opts]
        by_n[n_opts] = {
            "n": len(rows),
            "identity_orig_acc": rate("identity_orig_correct", rows),
            "identity_shuffled_acc": rate("identity_shuffled_correct", rows),
            "program_orig_acc": rate("program_orig_correct", rows),
            "program_shuffled_acc": rate("program_shuffled_correct", rows),
        }

    # Stratify by whether identity ALREADY got this query right (fresh, this run's
    # own identity pass -- not a precomputed flag, for the same same-hardware-
    # same-run consistency the module docstring already requires of
    # identity_orig/program_orig). When --rescue-jsonl was built with
    # --include-identity-correct (any valid non-identity program, not just
    # RESCUE), this answers whether reshuffle-fragility is specific to the
    # identity-wrong/rescue mechanism or general to any MCTS-discovered
    # program. On a pure-RESCUE population this collapses to one bucket
    # (identity_orig ~all False by construction, modulo the rare
    # same-hardware disagreement noted above).
    by_identity_orig: Dict[str, dict] = {}
    for flag, rows in (
        ("identity_orig_correct", [r for r in records if r["identity_orig_correct"]]),
        ("identity_orig_wrong", [r for r in records if not r["identity_orig_correct"]]),
    ):
        by_identity_orig[flag] = {
            "n": len(rows),
            "program_orig_acc": rate("program_orig_correct", rows),
            "program_shuffled_acc": rate("program_shuffled_correct", rows),
        }

    return {
        "overall": overall,
        "by_num_options": by_n,
        "by_identity_orig_correct": by_identity_orig,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rescue-jsonl", required=True)
    parser.add_argument("--mmlu-data-dir", default="mmlu_pro_domains")
    parser.add_argument(
        "--official-path",
        default="mmlu_pro_official/test.json",
        help="mmlu_pro_official test.json, used only if --mmlu-data-dir needs "
        "rebuilding (see below)",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument(
        "--full", action="store_true", help="use ALL RESCUE queries, ignoring --n-samples"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="identity batched pass only; "
        "program pass is necessarily per-query (each query reroutes the model)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rescue = load_rescue(args.rescue_jsonl)
    print(f"Loaded {len(rescue)} RESCUE queries from {args.rescue_jsonl}", flush=True)
    chosen = rescue if args.full else subsample(rescue, args.n_samples, args.seed)
    print(
        f"Using {len(chosen)} RESCUE queries "
        f"({'all' if args.full else f'subsampled, seed={args.seed}'})",
        flush=True,
    )

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
        # Needed for RESCUE query_ids drawn from a train+val+test-unioned MCTS run --
        # without this, every mmlu-test-* query_id silently fails the join below and
        # gets dropped from the population instead of scored.
        rows_by_split["test"] = json.loads((data_dir / "test.json").read_text())
    index = build_query_id_index(rows_by_split)
    print(
        f"Rebuilt split has {len(index)} query_ids ("
        + " + ".join(f"{len(rows)} {s}" for s, rows in rows_by_split.items())
        + ")",
        flush=True,
    )

    joined = join_rescue_with_mmlu(chosen, index)
    missing = set(chosen) - set(joined)
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(chosen)} RESCUE query_ids not found in the "
            f"rebuilt split -- dropped, not silently substituted",
            flush=True,
        )
    print(f"Joined {len(joined)} queries with real question/options/answer_index", flush=True)
    if not joined:
        raise SystemExit(
            "Nothing to do after joining -- check --mmlu-data-dir matches the "
            "split the rescue file's query_ids came from."
        )

    for qid, info in joined.items():
        shuf_options, shuf_answer = shuffle_options(
            info["options"], info["answer_index"], seed_key=f"{args.seed}:{qid}"
        )
        info["shuffled_options"] = shuf_options
        info["shuffled_answer_index"] = shuf_answer

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

    def make_sample(qid: str, variant: str) -> MMLUProSample:
        # composite id "{qid}::{variant}" (not a plain qid) so identity_by_qid_variant
        # below can key off run_mmlu_pro_domains's returned per_prompt rows directly, instead
        # of relying on prepared-input iteration order staying in lockstep with a
        # separately-built variant sequence -- robust to future reordering.
        info = joined[qid]
        composite_id = f"{qid}::{variant}"
        if variant == "orig":
            return MMLUProSample(
                id=composite_id,
                question=info["question"],
                options=tuple(info["options"]),
                answer_index=info["answer_index"],
                category=info["category"],
            )
        return MMLUProSample(
            id=composite_id,
            question=info["question"],
            options=tuple(info["shuffled_options"]),
            answer_index=info["shuffled_answer_index"],
            category=info["category"],
        )

    print(f"\n=== IDENTITY pass (batched, {len(joined)} queries x 2 variants) ===", flush=True)
    identity_samples = []
    for qid in joined:
        identity_samples.append(make_sample(qid, "orig"))
        identity_samples.append(make_sample(qid, "shuf"))
    identity_inputs = prepare_mmlu_pro_domain_inputs(
        tokenizer, identity_samples, no_think=True, device=device
    )
    identity_result = run_mmlu_pro_domains(
        engine.model, tokenizer, prepared_inputs=identity_inputs, batch_size=args.batch_size
    )
    identity_by_qid_variant: Dict[Tuple[str, str], dict] = {}
    for row in identity_result["per_prompt"]:
        qid, variant = row["id"].rsplit("::", 1)
        identity_by_qid_variant[(qid, variant)] = row
    print(
        f"Identity done: {identity_result['n']} scored, "
        f"overall avg={identity_result['average']:.3f} (orig+shuffled pooled)",
        flush=True,
    )

    print(
        f"\n=== PROGRAM pass (per-query reroute, {len(joined)} queries x 2 variants) ===",
        flush=True,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    with open(out_path, "w") as f:
        for i, qid in enumerate(joined):
            info = joined[qid]
            prog_samples = [make_sample(qid, "orig"), make_sample(qid, "shuf")]
            prog_inputs = prepare_mmlu_pro_domain_inputs(
                tokenizer, prog_samples, no_think=True, device=device
            )
            engine.apply_layer_rerouting(info["path"])
            try:
                prog_result = run_mmlu_pro_domains(
                    engine.model, tokenizer, prepared_inputs=prog_inputs, batch_size=2
                )
            finally:
                engine.restore_original()
            prog_orig, prog_shuf = prog_result["per_prompt"]

            id_orig = identity_by_qid_variant[(qid, "orig")]
            id_shuf = identity_by_qid_variant[(qid, "shuf")]
            rec = {
                "query_id": qid,
                "num_options": info["num_options"],
                "path": info["path"],
                "identity_orig_correct": bool(id_orig["score"]),
                "identity_shuffled_correct": bool(id_shuf["score"]),
                "program_orig_correct": bool(prog_orig["score"]),
                "program_shuffled_correct": bool(prog_shuf["score"]),
            }
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0 or (i + 1) == len(joined):
                agg_so_far = aggregate(records)["overall"]
                print(
                    f"  {i + 1}/{len(joined)}  running: identity_orig={agg_so_far['identity_orig_acc']:.1%} "
                    f"identity_shuf={agg_so_far['identity_shuffled_acc']:.1%} "
                    f"program_orig={agg_so_far['program_orig_acc']:.1%} "
                    f"program_shuf={agg_so_far['program_shuffled_acc']:.1%}",
                    flush=True,
                )

    summary = aggregate(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    o = summary["overall"]
    print(f"n={o['n']} RESCUE queries", flush=True)
    print(
        f"identity: original={o['identity_orig_acc']:.1%}  shuffled={o['identity_shuffled_acc']:.1%}  "
        f"(gain from shuffling alone, no program: {o['identity_shuffled_acc'] - o['identity_orig_acc']:+.1%})",
        flush=True,
    )
    print(
        f"program:  original={o['program_orig_acc']:.1%}  shuffled={o['program_shuffled_acc']:.1%}  "
        f"(loss from shuffling the SAME rescue program: "
        f"{o['program_shuffled_acc'] - o['program_orig_acc']:+.1%})",
        flush=True,
    )
    if o["jaccard_identity_shuffled_vs_program_shuffled"] is not None:
        print(
            f"Jaccard(identity_shuffled solved, program_shuffled solved) = "
            f"{o['jaccard_identity_shuffled_vs_program_shuffled']:.3f}",
            flush=True,
        )
    print("\nBy num_options:", flush=True)
    for n_opts, s in sorted(summary["by_num_options"].items()):
        print(
            f"  n_opts={n_opts:2d}  n={s['n']:4d}  identity orig/shuf="
            f"{s['identity_orig_acc']:.1%}/{s['identity_shuffled_acc']:.1%}  "
            f"program orig/shuf={s['program_orig_acc']:.1%}/{s['program_shuffled_acc']:.1%}",
            flush=True,
        )
    print(
        "\nBy identity's OWN original correctness (program orig/shuf, this run's "
        "fresh identity pass -- only interesting when --rescue-jsonl includes "
        "identity-already-correct queries, i.e. --include-identity-correct):",
        flush=True,
    )
    for flag, s in summary["by_identity_orig_correct"].items():
        print(
            f"  {flag:>22}  n={s['n']:4d}  program orig/shuf="
            f"{s['program_orig_acc']:.1%}/{s['program_shuffled_acc']:.1%}",
            flush=True,
        )
    print(f"\nWrote {len(records)} per-query records -> {out_path}", flush=True)
    print(f"Wrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
