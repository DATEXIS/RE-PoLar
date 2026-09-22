""""How wrong" closeness tagging for wrong_final_answer records. Reads
tag_with_llm.py's output, and for every record whose error_category is
"wrong_final_answer" ONLY, asks the judge LLM (see `analysis/error_analysis/llm/client.py` for
which model) to classify whether the wrong boxed answer is related or
unrelated to the ground truth. Every other error_category (unboxed /
malformed_boxed / prompt_template_echo) is skipped entirely -- there is no
comparable "wrong value" to judge closeness of.

This is a CLOSED comparison over two values the LLM is explicitly given
(the wrong answer AND the ground truth), not an inference about an unseen
derivation (see tag_with_llm.py's own docstring for why an *open-ended*
"why is it wrong" ask was dropped instead). It generalizes past pure
numbers (fractions, matrices, symbolic expressions) in a way a
numeric-distance-only metric cannot -- the wrong answer isn't always just
a number, it can also be an equation or another symbolic object.

Originally a 3-way scale (near_miss / same_ballpark / unrelated) --
collapsed to 2 (related / unrelated) after a real run showed the
near_miss-vs-same_ballpark boundary was genuinely inconsistent: e.g.
gt=2008/wrong=1004 (exactly half) got "near_miss", while gt=20/wrong=10
(the SAME relationship) got "same_ballpark". The "related vs. unrelated"
line held up much better on the same data.

Still genuinely subjective at the boundary (is a value 5x too big
"related" or "unrelated"?) -- no unit test can settle that, only:
  1. A concrete rubric with worked examples in the prompt itself (below),
     to cut arbitrariness without eliminating it.
  2. mechanical_closeness() below: for the subset where BOTH the ground
     truth and the extracted wrong answer parse as plain numbers, compute
     the SAME 2-way label from actual relative distance, no LLM involved,
     and report agreement with the judge LLM's label -- a rough
     calibration anchor (deliberately not finely tuned), not proof of
     correctness on the non-numeric majority where there's nothing to
     check against.
  3. A human spot-check of a real sample before trusting this as a paper
     claim -- not something a test suite can substitute for.

Confirmed on a real run (50 Qwen3-8B wrong_final_answer records) after the
3-way->2-way collapse: mechanical-calibration agreement rose from 73.0% to
85.1%, and the specific exact-half inconsistency that motivated the
collapse is directly fixed -- three separate exact-half pairs in that run
(e.g. 2008/1004 and 20/10) all get "related" now, previously the identical
relationship split across two different labels.

The judge LLM's own output labels (closeness, per query_id) are the one
piece of this pipeline that can't be exactly reproduced without the same
LLM, so they are the intended candidate for publishing as data alongside
this script; everything else here (the prompt, the mechanical calibration
anchor, the aggregation) is ordinary regenerable code.

    python -m analysis.error_analysis.tag_closeness \\
        --input tags.jsonl \\
        --output closeness.jsonl
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from analysis.error_analysis.llm import LLMCallTimeout, LLMClient, strip_think
from re_polar.vendor.dart_math.eval import extract_boxed

# See tag_with_llm.py's own docstring for why this sys.path trick is needed
# under both `python .../tag_closeness.py` and tests/'s importlib loading.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from tag_with_llm import _truncate_after_first_boxed  # noqa: E402

CLOSENESS_LABELS = ["related", "unrelated"]

CLOSENESS_PROMPT = (
    "A math problem's ground-truth answer and a wrong answer are given below. Classify whether "
    "the wrong answer is RELATED to the ground truth or UNRELATED, into EXACTLY one of these "
    "two categories:\n"
    "  related: any discernible connection -- close in value, same order of magnitude, a clean "
    "scaling or sign relationship (e.g. off by a factor, or a sign flip), the same expression "
    "type with a different coefficient, or similar structure (e.g. same denominator, same "
    "polynomial degree).\n"
    "  unrelated: no discernible relationship -- wildly different magnitude, or a completely "
    "different kind of mathematical object.\n\n"
    "Ground truth: {gt_ans}\n"
    "Wrong answer: {wrong_ans}\n\n"
    "Respond with EXACTLY one JSON object, no markdown code fence, no text before or after it, "
    "with one key:\n"
    f'  "closeness": one of {CLOSENESS_LABELS!r}\n'
)


def parse_closeness(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    label = data["closeness"]
    if label not in CLOSENESS_LABELS:
        raise ValueError(f"closeness {label!r} not in {CLOSENESS_LABELS!r}")
    return label


# Tolerant of the common surface forms seen in DART-Math ground truths: plain
# ints/floats, "$" money prefixes, "," thousands separators, trailing "\%".
# Deliberately narrow -- returns None (not parseable) for anything else
# (fractions, matrices, symbolic expressions) rather than guessing; those go
# through the LLM path only, with no mechanical cross-check available.
_NUMERIC_RE = re.compile(r"^-?\$?[\d,]*\.?\d+\%?$")


def try_parse_plain_number(s: str) -> float | None:
    s = s.strip()
    if not _NUMERIC_RE.match(s):
        return None
    cleaned = s.replace("$", "").replace(",", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return val / 100 if is_percent else val


def mechanical_closeness(gt_val: float, wrong_val: float) -> str:
    """Deterministic 2-way closeness from actual relative distance -- a rough
    calibration anchor, not a ground truth (the 10x threshold is a judgment
    call too, just a consistent, auditable one, unlike the LLM's -- and
    deliberately not finely tuned, see the module docstring)."""
    diff = abs(wrong_val - gt_val)
    if gt_val == 0:
        return "related" if diff <= 100 else "unrelated"
    rel_diff = diff / abs(gt_val)
    return "related" if rel_diff <= 10 else "unrelated"


def extract_wrong_answer(r: dict) -> str:
    """The SAME extraction classify_error_mechanically used to decide this
    record was wrong_final_answer in the first place -- must agree with
    tag_with_llm.py's own category assignment, not a second, possibly-
    drifting reimplementation."""
    identity_text = _truncate_after_first_boxed(r["identity_generated_text"])
    return extract_boxed(identity_text).strip()


async def tag_closeness_one(client: LLMClient, r: dict,
                             call_timeout_s: float | None = None) -> dict:
    wrong_ans = extract_wrong_answer(r)
    prompt = CLOSENESS_PROMPT.format(gt_ans=r["gt_ans"], wrong_ans=wrong_ans)

    mech = None
    gt_val = try_parse_plain_number(r["gt_ans"])
    wrong_val = try_parse_plain_number(wrong_ans)
    if gt_val is not None and wrong_val is not None:
        mech = mechanical_closeness(gt_val, wrong_val)

    try:
        raw = await client.achat([{"role": "user", "content": prompt}], max_tokens=4000,
                                  timeout_s=call_timeout_s)
    except LLMCallTimeout as e:
        print(f"WARNING: LLM call timed out for {r['query_id']}: {e}", flush=True)
        return {**r, "closeness": None, "mechanical_closeness": mech,
                "raw_llm_response": None, "llm_status": "LLM_CALL_TIMEOUT"}
    try:
        closeness = parse_closeness(strip_think(raw))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"WARNING: unparseable LLM response for {r['query_id']}: {e}", flush=True)
        return {**r, "closeness": None, "mechanical_closeness": mech,
                "raw_llm_response": raw, "llm_status": "UNPARSED_LLM_OUTPUT"}
    return {**r, "closeness": closeness, "mechanical_closeness": mech,
            "raw_llm_response": raw, "llm_status": "OK"}


async def _run_closeness_async(records: list, client: LLMClient, max_workers: int,
                                call_timeout_s: float | None, out_path: Path | None) -> list:
    sem = asyncio.Semaphore(max_workers)
    tagged = [None] * len(records)
    n_done = 0
    out_f = open(out_path, "a") if out_path is not None else None

    async def worker(i: int, r: dict) -> None:
        nonlocal n_done
        async with sem:
            result = await tag_closeness_one(client, r, call_timeout_s)
        tagged[i] = result
        n_done += 1
        if out_f is not None:
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
        agree = ("AGREE" if result["mechanical_closeness"] == result["closeness"]
                  else "disagree" if result["mechanical_closeness"] else "n/a")
        print(f"closeness {n_done}/{len(records)}: {result['query_id']} "
              f"gt={result['gt_ans']!r} -> {result['closeness']} "
              f"(mechanical={result['mechanical_closeness']}, {agree})", flush=True)

    try:
        await asyncio.gather(*(worker(i, r) for i, r in enumerate(records)))
    finally:
        if out_f is not None:
            out_f.close()
    return tagged


def run_closeness_tagging(records: list, client: LLMClient, max_workers: int,
                           call_timeout_s: float | None = None, out_path=None) -> list:
    out = Path(out_path) if out_path is not None else None
    return asyncio.run(_run_closeness_async(records, client, max_workers, call_timeout_s, out))


def load_existing(out_path: Path) -> dict:
    if not out_path.exists():
        return {}
    existing = {}
    with open(out_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: skipping unparseable line {lineno} in {out_path} "
                      f"(likely truncated by a mid-write kill) -- record will be "
                      f"reprocessed", flush=True)
                continue
            existing[rec["query_id"]] = rec
    return existing


DEFAULT_RETRY_STATUSES = frozenset({"LLM_CALL_TIMEOUT", "UNPARSED_LLM_OUTPUT"})


def partition_existing_for_retry(existing: dict, retry_statuses) -> tuple:
    keep, retry = {}, {}
    for qid, rec in existing.items():
        bucket = retry if rec.get("llm_status") in retry_statuses else keep
        bucket[qid] = rec
    return keep, retry


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="tag_with_llm.py output (must have error_category)")
    p.add_argument("--output", required=True)
    p.add_argument("--max-workers", type=int, default=15)
    p.add_argument("--call-timeout-s", type=float, default=1800.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-statuses", nargs="*", default=None)
    args = p.parse_args(argv)

    with open(args.input) as f:
        all_records = [json.loads(line) for line in f]
    records = [r for r in all_records if r["error_category"] == "wrong_final_answer"]
    print(f"{len(records)}/{len(all_records)} records are wrong_final_answer "
          f"(only category with a comparable value to judge closeness of)", flush=True)
    if args.limit:
        records = records[:args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_all = load_existing(out_path)
    retry_statuses = (set(args.retry_statuses) if args.retry_statuses is not None
                       else DEFAULT_RETRY_STATUSES)
    existing, to_retry = partition_existing_for_retry(existing_all, retry_statuses)
    if to_retry:
        print(f"Retrying {len(to_retry)} previously-degraded records", flush=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            for rec in existing.values():
                f.write(json.dumps(rec) + "\n")
        tmp_path.replace(out_path)
    remaining = [r for r in records if r["query_id"] not in existing]
    if existing:
        print(f"Resuming {out_path}: {len(existing)}/{len(records)} already tagged, "
              f"{len(remaining)} remaining", flush=True)

    with LLMClient() as client:
        newly_tagged = run_closeness_tagging(remaining, client, args.max_workers,
                                              args.call_timeout_s, out_path=out_path)
    tagged = list(existing.values()) + newly_tagged

    from collections import Counter
    counts = Counter(t["closeness"] for t in tagged)
    calibratable = [t for t in tagged if t["mechanical_closeness"] is not None]
    agreeing = sum(1 for t in calibratable if t["closeness"] == t["mechanical_closeness"])

    print(f"\n{len(tagged)} closeness-tagged records -> {out_path}", flush=True)
    print("Closeness counts:", flush=True)
    for label, n in counts.most_common():
        print(f"  {label}: {n}", flush=True)
    if calibratable:
        print(f"\nMechanical calibration: {len(calibratable)}/{len(tagged)} records had a "
              f"plain-number ground truth AND wrong answer (checkable). Agreement with "
              f"the judge LLM's own label: {agreeing}/{len(calibratable)} "
              f"({agreeing / len(calibratable):.1%})", flush=True)
    else:
        print("\nNo records had a plain-number ground truth -- no mechanical calibration "
              "possible on this input.", flush=True)


if __name__ == "__main__":
    main()
