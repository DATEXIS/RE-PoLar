"""RESCUE-pilot data prep for LLM-assisted error analysis.

Loads a `merged_mcts_samples.json` file (see `re_polar/datasets/schemas.py`), selects
RESCUE samples (base model wrong, search found >=1 working program), and
regenerates REAL text from the base model under both the identity program and
the shortest found valid program. Layer paths from the released data are flat
layer-index lists, not `Program` objects, so this goes through
`LayerEngine.apply_layer_rerouting()` directly (the same seam
`re_polar.core.ProgramExecutor` wraps) rather than forcing them through the
`Program`/segment IR.

Also re-grades both generations locally with the exact same strict grader
MCTS itself used, as a sanity check that regeneration reproduces the stored
pass/fail labels (a standing repo invariant) before anything goes to the LLM
tagging step in tag_with_llm.py -- the LLM only tags failure modes, it never
decides correctness.

**Superseded for full-corpus reproduction** by `build_error_analysis_from_
textlog.py`, which reads the exact text MCTS itself generated (via `re_polar/mcts/
textlog.py`'s `--log-answers` side-log) instead of regenerating it, this
script's own regeneration does NOT perfectly reproduce the original MCTS-time
verdict in every case (masked-batch-evaluated programs specifically; see
`re_polar/mcts/rewards.py::GenerationReward.masked_batch_call`'s docstring for the
~13% mismatch this stems from). Kept here for its `select_rescue`/
`select_base_wrong`/`pick_program` selection functions, which the textlog-based
pipeline reuses, and as the original, small-scale, no-cluster way to build a
pilot sample without needing the text log at all.

    python -m analysis.error_analysis.select_and_generate \
        --samples data/dart_math/diff1/merged_mcts_samples.json \
        --n-per-difficulty 10 --output rescue_pairs_pilot.jsonl

Top level is stdlib-only so this module can be imported (e.g. for tests of
`select_rescue`) without pulling in torch.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _round_robin_by_source(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for s in rows:
        by_source[s["sample_info"].get("source", "unknown")].append(s)
    for v in by_source.values():
        rng.shuffle(v)
    sources = list(by_source)
    selected: List[dict] = []
    i = 0
    while len(selected) < n and any(by_source.values()):
        src = sources[i % len(sources)]
        if by_source[src]:
            selected.append(by_source[src].pop())
        i += 1
    return selected


def select_rescue(samples: List[dict], n: int, rng: random.Random) -> List[dict]:
    """RESCUE = base wrong (`initial_transition_metric == 0.0`) with >=1 found
    program. Round-robins across `sample_info.source` (math/gsm8k) for balance."""
    rescue = [
        s for s in samples if s["initial_transition_metric"] == 0.0 and s["final_valid_transitions"]
    ]
    return _round_robin_by_source(rescue, n, rng)


def select_base_wrong(samples: List[dict], n: int, rng: random.Random) -> List[dict]:
    """ALL base-wrong samples (`initial_transition_metric == 0.0`), whether or
    not the search ever found a working fix -- a strict superset of
    select_rescue(). Samples where the search found nothing (UNRESCUABLE) get
    identity-alone error classification, no program comparison (see
    pick_program())."""
    wrong = [s for s in samples if s["initial_transition_metric"] == 0.0]
    return _round_robin_by_source(wrong, n, rng)


def pick_program(s: dict) -> Tuple[Optional[List[int]], str]:
    """Shortest valid program if the search found one (RESCUE) -- else None
    (UNRESCUABLE): deliberately NOT comparing against an attempted-but-failed
    program for these, since neither answer would be correct and there's no
    principled program to pick among the failed attempts. UNRESCUABLE samples
    only get identity-alone error classification (see tag_with_llm.py)."""
    if s["final_valid_transitions"]:
        return min(s["final_valid_transitions"], key=len), "RESCUE"
    return None, "UNRESCUABLE"


def generate_all(
    engine,
    tokenizer,
    device,
    prompt_fn,
    path: List[int],
    questions: List[str],
    max_new_tokens: int,
    batch_size: int,
    label: str = "",
) -> List[str]:
    import time
    import torch

    engine.apply_layer_rerouting(path)
    try:
        texts: List[str] = []
        n_batches = (len(questions) + batch_size - 1) // batch_size
        for bi, i in enumerate(range(0, len(questions), batch_size)):
            t0 = time.monotonic()
            chunk = questions[i : i + batch_size]
            prompts = [prompt_fn(tokenizer, q) for q in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = engine.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts.extend(
                tokenizer.batch_decode(
                    out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            )
            print(
                f"{label} batch {bi + 1}/{n_batches} "
                f"({len(texts)}/{len(questions)} samples, {time.monotonic() - t0:.1f}s/batch)",
                flush=True,
            )
        return texts
    finally:
        engine.restore_original()


def main(argv=None):
    # heavy imports INSIDE main() (mirrors the rest of this repo's CLIs)
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.datasets.schemas import load_samples
    from re_polar.mcts.rewards import PROMPT_STYLES, truncate_after_first_boxed
    from re_polar.core.grader import _safe_grade
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples",
        action="append",
        required=True,
        help="merged_mcts_samples.json path (repeatable, one per " "--difficulty, same order)",
    )
    p.add_argument(
        "--difficulty",
        type=int,
        action="append",
        default=None,
        help="1-4 (repeatable); default 1-4 (diff5 not included by default)",
    )
    p.add_argument("--n-per-difficulty", type=int, default=10)
    p.add_argument("--model", default="qwen3_8b")
    p.add_argument(
        "--model-label",
        default=None,
        help="value written to each record's 'model' field; defaults to --model. "
        "Lets the error-analysis artifact key records by model.",
    )
    p.add_argument(
        "--prompt-style",
        default="paper_minimal_fewshot",
        choices=list(PROMPT_STYLES),
        help="default matches the MCTS search's own canonical choice. Using the "
        "wrong prompt here means the regenerated identity/found-program TEXT fed "
        "to the LLM error-tagger doesn't match what MCTS search actually used.",
    )
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="debug: cap total selected samples")
    p.add_argument("--output", default="rescue_pairs_pilot.jsonl")
    p.add_argument(
        "--include-unrescuable",
        action="store_true",
        help="also include base-wrong samples where the search never found a "
        "working fix (compares identity against the shortest ATTEMPTED-but-"
        "still-failed program instead) -- default is RESCUE-only, matching "
        "the original pilot/full-500 runs",
    )
    args = p.parse_args(argv)

    difficulties = args.difficulty or [1, 2, 3, 4]
    if len(args.samples) != len(difficulties):
        raise SystemExit(
            f"--samples given {len(args.samples)} paths but {len(difficulties)} "
            f"--difficulty values ({difficulties}) -- must be 1:1, same order."
        )
    rng = random.Random(args.seed)
    select_fn = select_base_wrong if args.include_unrescuable else select_rescue

    selected: List[dict] = []
    for i, d in enumerate(difficulties):
        samples = load_samples(Path(args.samples[i]))
        selected.extend(select_fn(samples, args.n_per_difficulty, rng))
    if args.limit:
        selected = selected[: args.limit]
    kind = "base-wrong (RESCUE + unrescuable)" if args.include_unrescuable else "RESCUE"
    print(f"Selected {len(selected)} {kind} samples across difficulties {difficulties}", flush=True)
    if not selected:
        raise SystemExit("No samples selected -- nothing to do.")

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}; known: {sorted(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    tokenizer = engine.tokenizer
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"  # generation needs left padding
    prompt_fn = PROMPT_STYLES[args.prompt_style]
    identity_path = list(range(engine.num_layers))
    evaluator = EvaluatorMath(
        strict_extract=True
    )  # matches re_polar/core/grader.py's real MCTS-time protocol

    questions = [s["question"] for s in selected]
    gts = [s["gt_ans"] for s in selected]

    print("Generating under IDENTITY...", flush=True)
    identity_texts = generate_all(
        engine,
        tokenizer,
        engine.device,
        prompt_fn,
        identity_path,
        questions,
        args.max_new_tokens,
        args.batch_size,
        label="identity",
    )

    program_info = [pick_program(s) for s in selected]  # (path_or_None, sample_type) per sample
    rescue_indices = [i for i, (path, _) in enumerate(program_info) if path is not None]
    print(
        f"Generating under found RESCUE programs (one apply per sample, "
        f"{len(rescue_indices)}/{len(selected)} samples -- UNRESCUABLE ones skip this "
        f"entirely, no attempted-but-failed program is generated for them)...",
        flush=True,
    )
    program_texts = {}  # index -> text, only for rescue_indices
    for j, idx in enumerate(rescue_indices):
        path, _ = program_info[idx]
        text = generate_all(
            engine,
            tokenizer,
            engine.device,
            prompt_fn,
            path,
            [selected[idx]["question"]],
            args.max_new_tokens,
            batch_size=1,
            label=f"program sample {j + 1}/{len(rescue_indices)}",
        )[0]
        program_texts[idx] = text

    n_identity_mismatch = 0
    n_rescue_mismatch = 0  # RESCUE: expected program_correct=True, wasn't
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for idx, (s, gt, id_text) in enumerate(zip(selected, gts, identity_texts)):
            prog_path, sample_type = program_info[idx]
            # truncate_after_first_boxed BEFORE grading -- MUST match re_polar/mcts/rewards.py's
            # own GenerationReward path, else extract_boxed's unconditional
            # resp.split("oxed")[-1] grabs a hallucinated next-fewshot-block's placeholder
            # instead of the model's real (earlier, often correct) boxed answer. A real bug
            # was found and fixed here for exactly this reason -- this is the fix so
            # future runs don't reintroduce it.
            identity_correct = _safe_grade(evaluator, gt, truncate_after_first_boxed(id_text))
            if identity_correct != bool(s["initial_transition_metric"]):
                n_identity_mismatch += 1

            prog_text = program_correct = None
            if sample_type == "RESCUE":
                prog_text = program_texts[idx]
                program_correct = _safe_grade(evaluator, gt, truncate_after_first_boxed(prog_text))
                if not program_correct:
                    n_rescue_mismatch += 1

            rec = {
                "model": args.model_label or args.model,
                "query_id": s["sample_info"]["query_id"],
                "difficulty": s["sample_info"]["difficulty"],
                "source": s["sample_info"].get("source"),
                "domain": s["sample_info"].get("domain"),
                "sample_type": sample_type,
                "question": s["question"],
                "gt_ans": gt,
                "identity_path": identity_path,
                "program_path": prog_path,
                "identity_generated_text": id_text,
                "program_generated_text": prog_text,
                "identity_correct": identity_correct,
                "program_correct": program_correct,
            }
            f.write(json.dumps(rec) + "\n")

    n_rescue = len(rescue_indices)
    n_unrescuable = len(selected) - n_rescue
    print(
        f"Wrote {len(selected)} records -> {out_path} "
        f"({n_rescue} RESCUE, {n_unrescuable} UNRESCUABLE)",
        flush=True,
    )
    print(
        f"Sanity check: identity-label mismatches vs. stored initial_transition_metric: "
        f"{n_identity_mismatch}/{len(selected)}",
        flush=True,
    )
    print(
        f"Sanity check: RESCUE programs that did NOT grade correct (expected correct): "
        f"{n_rescue_mismatch}/{n_rescue}",
        flush=True,
    )


if __name__ == "__main__":
    main()
