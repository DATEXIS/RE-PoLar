"""Error tagging -- despite the filename, error_category is NOT LLM-assigned.
History, kept because it explains why the taxonomy looks the way it does:

1. Originally an LLM picked error_category from a 6-way list (arithmetic_
   error / misread_question / wrong_formula_or_method / etc.) via a fixed
   prompt.
2. That taxonomy was dropped once we established it had no grounded
   evidence behind it: `_truncate_after_first_boxed` (below) deliberately
   keeps the tagger's input identical to what the strict grader actually
   scored -- correct design, not a bug, but base models frequently emit
   `\\boxed{ANSWER}` FIRST, before any reasoning, so after truncation
   there's often no genuine PRE-answer reasoning left to show at all.
   Replaced with classify_error_mechanically() below: derives
   error_category from the SAME extract_boxed (LAST boxed span) the real
   grader (`re_polar/core/grader.py`) uses, in code, no LLM call involved.
3. The mechanical category change kept one LLM call alive, for a free-text
   "reasoning" comment only. A small preview against real RESCUE data
   showed why that has to go too: the LLM produced SPECIFIC,
   confident-sounding "reasoning" for records where it was only ever shown
   `\\boxed{-6}` -- e.g. "suggesting a miscalculation of the terms when
   substituting x = -1" -- a claim about a step it never saw. This wasn't a
   competence problem (the LLM's own DART-Math solve accuracy is ~97%+) --
   it's an information problem: asked to explain a wrong answer with only
   (question, wrong number, right number) to go on, a capable model will
   confabulate a plausible-sounding story rather than say "I don't know."
   No amount of solve competence fixes that, since the reasoning trace that
   would ground a real explanation was never shown to it in the first
   place. Since the categorization was ALREADY mechanical (point 2),
   dropping "reasoning" leaves nothing for an LLM to do in this pipeline at
   all -- so it's gone. This script no longer makes any LLM calls in its
   main path.

What's left is pure, fast, local computation: read RESCUE/UNRESCUABLE
records (written by select_and_generate.py / build_error_analysis_from_
textlog.py), classify each one's identity_generated_text via the
deterministic prompt_template_echo prefix check (analysis_utils.
starts_with_unfilled_boxed_placeholder) and classify_error_mechanically(),
write the tagged JSONL. No network calls, no concurrency, no resume/retry
machinery needed -- a full re-run is cheap enough that overwriting --output
each time is simpler than resuming a partial one.

--calibrate N is the ONE remaining place this script talks to an LLM at
all: an open-ended, manual/exploratory prompt on the first N records,
printed to stdout, nothing written to disk -- useful for a human to poke at
the judge model's raw response style (see `analysis/error_analysis/llm/client.py` for which
model), never part of the main tagging path.

    python -m analysis.error_analysis.tag_with_llm --calibrate 5
    python -m analysis.error_analysis.tag_with_llm

--calibrate requires LLM_API_KEY (and LLM_BASE_URL/LLM_MODEL, see
`analysis/error_analysis/llm/client.py`) in the environment; the
default (tagging) path needs neither.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# torch-free (see re_polar/core/grader.py's own docstring) -- same extraction the
# strict grader uses (extract_boxed: LAST "oxed{...}" span), imported here
# rather than reimplemented, so classify_error_mechanically() can't drift
# from what the grader itself actually scores.
from re_polar.vendor.dart_math.eval import extract_boxed

# tests/ loads this file via importlib.util.spec_from_file_location, which
# (unlike `python .../tag_with_llm.py` directly) does NOT add this file's own
# directory to sys.path -- so the plain `from analysis_utils import ...`
# below needs it added explicitly to work under both load paths.
import sys

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from analysis_utils import starts_with_unfilled_boxed_placeholder


def summarize_path(path: list, num_layers: int) -> dict:
    """Human-readable summary of an executed layer path: how many layers,
    which ones were skipped (never appear), which were repeated (appear
    more than once, with count)."""
    c = Counter(path)
    return {
        "len": len(path),
        "skipped": [l for l in range(num_layers) if c[l] == 0],
        "repeated": [(l, c[l]) for l in range(num_layers) if c[l] > 1],
    }


def format_program_summary(summary: dict) -> str:
    bits = [f"{summary['len']} layers"]
    if summary["skipped"]:
        bits.append(f"skips {summary['skipped']}")
    if summary["repeated"]:
        bits.append(f"repeats {summary['repeated']}")
    return ", ".join(bits)


# Only used by --calibrate (manual/exploratory, never the main tagging path).
CALIBRATION_PROMPT = (
    "Here is a math question. A base language model answered it WRONG; an edited "
    "version of the same model (different layers executed) answered it RIGHT. In "
    "your own words, describe what kind of mistake the base model made and what "
    "seems different about the correct attempt.\n\n"
    "Question: {question}\nGround truth: {gt_ans}\n\n"
    "WRONG (base) answer:\n{identity_generated_text}\n\n"
    "RIGHT (edited) answer:\n{program_generated_text}\n"
)

# "prompt_template_echo" is deliberately NOT included: it's an even earlier,
# stricter pre-filter (starts_with_unfilled_boxed_placeholder) that
# short-circuits tag_one() before classify_error_mechanically ever runs.
ERROR_CATEGORIES = [
    "unboxed",
    "malformed_boxed",
    "wrong_final_answer",
]


def classify_error_mechanically(text: str) -> str:
    """error_category for a known-wrong identity answer (RESCUE/UNRESCUABLE
    selection already guarantees it's wrong -- no gt_ans comparison needed
    here, just "is there a real answer to even be wrong about"):
      - "unboxed": no \\boxed{} span at all.
      - "malformed_boxed": a \\boxed{} span exists but its content, after
        the SAME extraction the grader uses (extract_boxed), is empty --
        nothing there for the grader to have even attempted to check.
      - "wrong_final_answer": a \\boxed{} span with real (non-empty) content
        that the grader determined isn't equal to gt_ans. Deliberately does
        NOT try to further distinguish "garbled non-math content" from "a
        legitimately-computed-but-wrong number" -- the grader's own eq()
        treats both identically (both just return False), so inventing a
        finer split here would be a distinction the grader itself doesn't
        make."""
    if "oxed{" not in text:
        return "unboxed"
    if not extract_boxed(text).strip():
        return "malformed_boxed"
    return "wrong_final_answer"


def tag_one(r: dict) -> dict:
    """Mechanically tag a single base-wrong record -- pure function, no
    network call. "prompt_template_echo" is a plain prefix check on the
    base model's raw output (analysis_utils.starts_with_unfilled_boxed_placeholder),
    checked first; everything else goes through classify_error_mechanically
    on the SAME truncated text a human/LLM would see if shown one (see
    _truncate_after_first_boxed) -- the real scored answer, not the raw
    generation."""
    if starts_with_unfilled_boxed_placeholder(r["identity_generated_text"]):
        return {**r, "error_category": "prompt_template_echo"}
    identity_text = _truncate_after_first_boxed(r["identity_generated_text"])
    return {**r, "error_category": classify_error_mechanically(identity_text)}


def _truncate_after_first_boxed(text: str) -> str:
    """MUST match re_polar/mcts/rewards.py::truncate_after_first_boxed exactly --
    reimplemented here (not imported) to keep this script torch-free
    (rewards.py imports torch at module top level for GenerationReward; this
    function itself is pure string logic). Cuts `text` right
    after the FIRST complete `\\boxed{...}` span closes -- the real answer
    the strict grader actually scored, before any trailing hallucinated
    continuation (fewshot-template echo, a rambling restart, etc.)."""
    if text.count("oxed{") < 2:
        return text
    prefix, rest = text.split("oxed{", 1)
    depth = 1
    i = 0
    for i, c in enumerate(rest):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
    return prefix + "oxed{" + rest[: i + 1]


def _log_snippet(text: str | None, n: int = 250) -> str:
    """Single-line, whitespace-collapsed preview of the REAL scored answer
    (post `_truncate_after_first_boxed`, not the raw generation) for the
    progress log -- so the console output alone shows what was actually
    flagged, not the trailing fewshot-echo noise or a bare category
    label."""
    if not text:
        return ""
    snippet = " ".join(_truncate_after_first_boxed(text).split())
    return snippet[:n] + ("..." if len(snippet) > n else "")


def tag_all(records: list) -> list:
    """Mechanically tag every record, in input order -- pure computation, no
    I/O, no network, so no need for the concurrency/checkpoint/resume
    machinery a network-bound version would need (see module docstring's
    history section for why that machinery existed and why it's gone)."""
    tagged = []
    for i, r in enumerate(records, 1):
        result = tag_one(r)
        tagged.append(result)
        print(
            f"tagged {i}/{len(records)}: {result['query_id']} "
            f"gt={result['gt_ans']!r} ans=[{_log_snippet(result['identity_generated_text'])}] "
            f"-> {result['error_category']}",
            flush=True,
        )
    return tagged


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="rescue_pairs_pilot.jsonl")
    p.add_argument("--output", default="rescue_tags_pilot.jsonl")
    p.add_argument(
        "--calibrate",
        type=int,
        default=0,
        help="if >0, run only on the first N records against the judge LLM "
        "with an open-ended prompt and print raw responses -- no output "
        "file, no mechanical tagging. The only mode that talks to an LLM.",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    with open(args.input) as f:
        records = [json.loads(line) for line in f]
    if args.limit:
        records = records[: args.limit]

    if args.calibrate:
        from analysis.error_analysis.llm import (
            LLMClient,
            strip_think,
        )  # only path that needs the network

        with LLMClient() as client:
            for r in records[: args.calibrate]:
                prompt = CALIBRATION_PROMPT.format(
                    question=r["question"],
                    gt_ans=r["gt_ans"],
                    identity_generated_text=r["identity_generated_text"],
                    program_generated_text=r["program_generated_text"],
                )
                resp = strip_think(client.chat([{"role": "user", "content": prompt}]))
                print(f"=== {r['query_id']} ===")
                print(resp)
                print()
        return

    tagged = tag_all(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in tagged:
            f.write(json.dumps(rec) + "\n")

    counts = Counter(t["error_category"] for t in tagged)
    print(f"{len(tagged)} tagged records total -> {out_path}")
    print("Category counts:")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
