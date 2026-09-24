"""MAWPS OOD eval set, test split only, eval-only.

PoLar cites MAWPS not via the original MAWPS paper (Koncel-Kedziorski et al.,
2016) but as "Kadlčík et al., 2023", the Calc-X/Calcformers paper (EMNLP
2023, "Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought
through Interaction with Symbolic Systems"). That points squarely at the
MAWPS split released as part of that group's Calc-X dataset collection on HF,
not a re-scrape of the original MAWPS repo, PoLar's own repo has NO MAWPS
loading code at all (verified directly, same check as asdiv.py's docstring),
so this is a deliberate sourcing decision, not a guess.

SOURCE: `MU-NLPC/Calc-mawps` (MIT). Has train(1089)/validation(1040)/
test(520) splits, those exist because Calc-X built this for calculator-tool-
use TRAINING, not because PoLar trains on any of it. **This module uses ONLY
`test`** (520 rows): PoLar's OOD role for MAWPS is zero-shot eval, mirroring
how `asdiv.py` uses ASDiv's entire (single, eval-only) corpus, using `test`
here is the same "don't touch what's meant for training" choice, and avoids
the router ever incidentally training on rows that could later end up in
someone's eval slice. Verified near-zero cross-split leakage on `test`:
520/520 questions distinct within `test`, only 1 also appears verbatim in
`validation` (0 in `train`), negligible, not investigated further since
`test` is used standalone, never unioned with the other splits.

WHY THIS DIFFERS FROM ASDIV'S "USE THE WHOLE CORPUS":
same underlying rule in both modules, use whichever portion of the source
release is eval-only/held-out, applied to two differently-shaped releases.
`EleutherAI/asdiv` ships exactly ONE split (no train portion exists at all;
Miao et al. 2020 built ASDiv purely as an eval corpus, never meant for
training), so "the whole corpus" and "the eval set" are the same thing there.
`MU-NLPC/Calc-mawps` ships THREE splits because Calc-X's own purpose was
training calculator-tool-use models, train/val are real training data in
this release, so only `test` is the eval-only portion here. Also
forward-looking: a router-side ASDiv/MAWPS training bridge is plausible
future work; keeping `test` untouched now means it stays a valid held-out
eval set if/when that bridge trains on `train`/`validation` later, the same
train/eval-disjointness invariant `re_polar/datasets/mmlu_pro_domains.py` already
asserts elsewhere.

No category/domain field exists in this dataset (checked, schema is id/
question/chain/result/result_float/equation/expression only), unlike ASDiv's
`solution_type`.

ANSWER FORMAT: `result` is already a clean plain string, e.g. "159" or a
fraction "56/9" (165/520 = 31.7% of test are fractions, not integers), no
unit-stripping needed (unlike ASDiv's "N (unit)" convention). Used directly as
`gt_ans`; `EvaluatorMath.eq` handles fractional refs natively.

DROP-IN for the existing eval path: same `{"question": str, "gt_ans": str,
...}` shape `re_polar.mcts.rewards.GenerationReward` already consumes for
DART-Math, see `asdiv.py`'s docstring for the fuller rationale, applies
identically here.

Output: <out_dir>/test.json (flat list) + manifest.json.

Usage:
  python -m re_polar.datasets.mawps --out-dir ./data/mawps
"""

import argparse
import json
from pathlib import Path

SOURCE_DATASET = "MU-NLPC/Calc-mawps"
SOURCE_SPLIT = "test"
EXPECTED_N = 520


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_eval_split(out_dir: Path) -> dict:
    import datasets

    ds = datasets.load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)
    if len(ds) != EXPECTED_N:
        raise ValueError(
            f"{SOURCE_DATASET}:{SOURCE_SPLIT} has {len(ds)} rows, "
            f"expected {EXPECTED_N}, source dataset changed upstream?"
        )

    seen_q = set()
    n_dupe_q = 0
    n_fraction = 0
    records = []
    for row in ds:
        q = str(row["question"])
        if q in seen_q:
            n_dupe_q += 1
        seen_q.add(q)

        result = str(row["result"])
        is_int_like = _is_numeric(result)
        if not is_int_like:
            n_fraction += 1

        records.append(
            {
                "id": len(records),
                "query_id": str(row["id"]),
                "question": q,
                "gt_ans": result,
                "gt_ans_float": float(row["result_float"]),
                "equation": str(row["equation"]),
                "expression": str(row["expression"]),
            }
        )

    manifest = {
        "source_dataset": SOURCE_DATASET,
        "source_split": SOURCE_SPLIT,
        "role": "OOD eval-only (out-of-domain transfer check), test split only, "
        "train/validation exist upstream but are intentionally unused",
        "citation": "Kadlčík et al., 2023 (Calc-X/Calcformers, EMNLP 2023), "
        "the MAWPS split PoLar's own citation points at, not the "
        "original 2016 MAWPS release",
        "license": "mit",
        "n_total": len(records),
        "n_duplicate_questions_within_test": n_dupe_q,
        "n_fractional_gt_ans": n_fraction,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test.json", "w") as f:
        json.dump(records, f, indent=1)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Wrote {len(records)} MAWPS eval records -> {out_dir / 'test.json'} "
        f"(fractional_gt_ans={n_fraction} dupe_q={n_dupe_q})"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build_eval_split(Path(args.out_dir))


if __name__ == "__main__":
    main()
