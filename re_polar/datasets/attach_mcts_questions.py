"""Join question/gt_ans (from `re_polar.datasets.dart_math`'s output) onto the
published mcts_results samples (which don't carry it), by query_id.

  python -m re_polar.datasets.dart_math --out-dir ./data/dart_math --include-gsm8k
  python -m re_polar.datasets.attach_mcts_questions \\
      --mcts-dir mcts_results --dart-math-dir ./data/dart_math --out-dir ./data/mcts_results_full

Output mirrors the input tree (`<out-dir>/<model>/dart-math-diff-<N>/
merged_mcts_samples.json`), records back in `schemas.sample_record()`'s
original shape. Don't commit the output dir.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from re_polar.datasets.schemas import load_samples, write_merged_samples


def build_question_lookup(dart_math_dir: Path) -> Dict[str, Tuple[str, str]]:
    """query_id -> (question, gt_ans), from every diff{N}/{train,val,test}.json
    under dart_math_dir (re_polar.datasets.dart_math's own output layout)."""
    lookup: Dict[str, Tuple[str, str]] = {}
    files = sorted(dart_math_dir.glob("diff*/*.json"))
    if not files:
        raise FileNotFoundError(
            f"No diff*/*.json files under {dart_math_dir} -- run "
            "`python -m re_polar.datasets.dart_math --out-dir <dir> --include-gsm8k` first."
        )
    for i, f in enumerate(files, 1):
        if f.name == "manifest.json":
            continue
        print(f"[lookup {i}/{len(files)}] {f}", flush=True)
        for rec in json.loads(f.read_text(encoding="utf-8")):
            lookup[rec["query_id"]] = (rec["question"], rec["gt_ans"])
    return lookup


def attach(mcts_dir: Path, out_dir: Path, lookup: Dict[str, Tuple[str, str]]) -> None:
    files = sorted(mcts_dir.glob("*/dart-math-diff-*/merged_mcts_samples.json")) + sorted(
        mcts_dir.glob("*/dart-math-diff-*/merged_mcts_samples.json.gz")
    )
    if not files:
        raise FileNotFoundError(
            f"No */dart-math-diff-*/merged_mcts_samples.json[.gz] under {mcts_dir}"
        )

    for i, f in enumerate(files, 1):
        model_path, namespace = f.parent.parent.name, f.parent.name
        print(f"[attach {i}/{len(files)}] {model_path}/{namespace}", flush=True)
        samples = load_samples(f)
        full_samples = []
        missing = []
        for s in samples:
            qid = s["sample_info"]["query_id"]
            if qid not in lookup:
                missing.append(qid)
                continue
            question, gt_ans = lookup[qid]
            full_samples.append({"question": question, "gt_ans": gt_ans, **s})
        if missing:
            raise KeyError(
                f"{f}: {len(missing)} query_id(s) not found in the reconstructed dart-math data "
                f"(first: {missing[0]!r}). Did you pass the matching --dart-math-dir "
                f"(same --include-gsm8k / --seed as used to build the published splits)?"
            )
        write_merged_samples(out_dir, model_path, namespace, full_samples)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mcts-dir", required=True, help="Root of the trimmed mcts_results tree")
    p.add_argument(
        "--dart-math-dir",
        required=True,
        help="Output dir of `re_polar.datasets.dart_math --include-gsm8k`",
    )
    p.add_argument(
        "--out-dir", required=True, help="Where to write the full merged_mcts_samples.json files"
    )
    args = p.parse_args(argv)

    lookup = build_question_lookup(Path(args.dart_math_dir))
    print(
        f"Loaded {len(lookup)} query_id -> question/gt_ans entries from {args.dart_math_dir}",
        flush=True,
    )
    attach(Path(args.mcts_dir), Path(args.out_dir), lookup)
    print(f"Done. Wrote merged samples for all files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
