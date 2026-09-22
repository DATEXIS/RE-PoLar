"""ASDiv OOD eval set, whole-corpus, eval-only, no train/val.

PoLar cites ASDiv as "Miao et al., 2020" (arXiv:2106.15772, "A Diverse Corpus
for Evaluating and Developing English Math Word Problem Solvers") as one of
its three OOD transfer benchmarks (ASDiv/MAWPS/MMLU-Pro), used purely for
zero-shot eval, PoLar never trains on it, so there is no train/val split to
build here, only a single frozen eval set. PoLar's own repo has NO ASDiv
loading code at all (verified via a direct grep over their public repo),
so the exact HF source is not pinned
upstream; the sourcing decision below is explicit, not guessed.

SOURCE: `EleutherAI/asdiv` (CC-BY-NC-4.0), not the original
chao-chun/nlu-asdiv-dataset XML or the Calc-X `MU-NLPC/Calc-asdiv_a` arithmetic
-only subset, highest download count among HF ports of the SAME corpus the
citation names (2305 rows total, matches the paper's full-corpus figure), and
it carries a `solution_type` field (Addition/Comparison/TVQ-Change/..., 24
distinct values) that `MU-NLPC/Calc-asdiv_a` drops, useful for later per-type
breakdown in the paper's analysis section.

Single split ("validation" in the HF repo, but semantically ASDiv's whole
corpus, there is no train/test partition upstream). `question` here is
`body + " " + question` (ASDiv's own two-field problem statement; `body` sets
up the scenario, `question` asks it), this is the standard concatenation
(matches lm-evaluation-harness's `asdiv` task's `doc_to_text`).

ANSWER FORMAT (checked directly against the full corpus): raw `answer` field is USUALLY
"<value> (<unit>)" (e.g. "9 (apples)") but not always:
  * 2084/2305 (90.4%) are single-part, purely numeric once the unit is stripped.
  * 131/2305 (5.7%) are single-part but NON-numeric even after stripping:
    dates ("February 3rd"), times ("3:30 p.m."), names ("Bryan"), yes/no,
    colors ("Purple"), ordinals ("6th"). These need literal string equality at
    grading time, not `EvaluatorMath`'s numeric/symbolic comparison, flagged
    per-record via `is_pure_numeric=False`, not silently coerced.
  * 90/2305 (3.9%) are MULTI-part ("5 (years old); 15 (years old); 20 (years
    old)", set/sequence-style answers). Split on ";" first, then strip each
    part's own "(unit)", a naive single regex over the whole string grabs
    only the LAST "(...)" and mis-parses everything before it. Multi-part
    records get `gt_ans_parts: [str, ...]` (each part's stripped value) AND
    `gt_ans` = those parts joined with ", " (a plain string for callers that
    only read `gt_ans`, e.g. `re_polar.mcts.rewards.GenerationReward` /
    `EvaluatorMath.eq`); `is_multi_answer=True` flags these for the eval
    script to optionally grade with `compare_sets=True` instead of trusting
    the joined-string default.
  * This module does NOT attempt to auto-detect a "querying for a set" prompt
    style (`re_polar.vendor.dart_math.eval.is_querying4set` keys off question
    phrasing DART-Math itself uses, e.g. "find the ... separate", ASDiv's
    questions don't share that phrasing, so the heuristic doesn't transfer;
    the `is_multi_answer` flag on the DATA is the intentionally simpler
    per-record signal instead).

DROP-IN for the existing eval path: records are `{"question": str, "gt_ans":
str, ...}`, the exact shape `re_polar.mcts.rewards.GenerationReward.__call__`
already consumes for DART-Math (`questions: List[str], gt_answers: List[str]`)
via the SAME `re_polar.vendor.dart_math.eval.EvaluatorMath` grader PoLar's own
generation-reward path uses, so running this OOD set is a data-swap, not new
eval code.

Output: <out_dir>/test.json (flat list) + manifest.json.

Usage:
  python -m re_polar.datasets.asdiv --out-dir ./data/asdiv
"""

import argparse
import json
import re
from pathlib import Path

SOURCE_DATASET = "EleutherAI/asdiv"
SOURCE_SPLIT = "validation"  # ASDiv's whole corpus; no train/test upstream
EXPECTED_N = 2305

_SEG_PAT = re.compile(r"^(.*?)\s*\(([^()]*)\)$")


def parse_answer(raw: str) -> list:
    """"5 (years old); 15 (years old)" -> ["5", "15"]; "9 (apples)" -> ["9"];
    "Purple" -> ["Purple"] (no unit to strip, passed through as-is)."""
    parts = [s.strip() for s in raw.split(";")]
    out = []
    for p in parts:
        m = _SEG_PAT.match(p)
        out.append(m.group(1).strip() if m else p)
    return out


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _dedup_key(row: dict) -> tuple:
    return (str(row["body"]), str(row["question"]), str(row["answer"]))


def build_eval_split(out_dir: Path) -> dict:
    import datasets

    ds = datasets.load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)
    if len(ds) != EXPECTED_N:
        raise ValueError(f"{SOURCE_DATASET}:{SOURCE_SPLIT} has {len(ds)} rows, "
                         f"expected {EXPECTED_N}, source dataset changed upstream?")

    seen = set()
    n_dupe = 0
    records = []
    n_numeric = n_other = n_multi = 0
    for row in ds:
        key = _dedup_key(row)
        if key in seen:
            n_dupe += 1
        seen.add(key)

        parts = parse_answer(str(row["answer"]))
        is_multi = len(parts) > 1
        is_pure_numeric = (not is_multi) and _is_numeric(parts[0])
        if is_multi:
            n_multi += 1
        elif is_pure_numeric:
            n_numeric += 1
        else:
            n_other += 1

        records.append({
            "id": len(records),
            "question": f"{row['body']} {row['question']}".strip(),
            "gt_ans": ", ".join(parts),
            "gt_ans_parts": parts,
            "is_multi_answer": is_multi,
            "is_pure_numeric": is_pure_numeric,
            "solution_type": str(row["solution_type"]),
            "formula": str(row["formula"]),
            "raw_answer": str(row["answer"]),
        })

    manifest = {
        "source_dataset": SOURCE_DATASET, "source_split": SOURCE_SPLIT,
        "role": "OOD eval-only (out-of-domain transfer check), whole corpus, no train/val",
        "citation": "Miao et al., 2020 (arXiv:2106.15772)",
        "license": "cc-by-nc-4.0",
        "n_total": len(records),
        "n_duplicate_content_keys": n_dupe,
        "answer_shape": {
            "single_part_pure_numeric": n_numeric,
            "single_part_non_numeric": n_other,
            "multi_part": n_multi,
        },
        "solution_types": sorted(set(r["solution_type"] for r in records)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test.json", "w") as f:
        json.dump(records, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(records)} ASDiv eval records -> {out_dir / 'test.json'} "
          f"(numeric={n_numeric} other={n_other} multi={n_multi} dupes={n_dupe})")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build_eval_split(Path(args.out_dir))


if __name__ == "__main__":
    main()
