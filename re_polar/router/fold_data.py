"""Router train/val/test data on the PoLar split (paper Appendix B).

One test set per level is noisy (141-395 questions), so the router and fixed-program
results of Appendix B use 4-fold cross-validation over each level's whole question pool
(`re_polar.datasets.polar_split`): a seeded shuffle of the level's query_ids, test = the
K-th quarter, val = the first round(n/8) of the remaining questions (shuffled order),
train = the rest. Over K = 0..3 every question is a test question exactly once. The
assignment depends only on the query_ids and the seed, so every search run on a level
gets the identical folds. `--fixed-split-dir` instead takes Li et al.'s inferred
train/val/test cut (`<polar split>/split`).

Writes
  --out-samples          {"samples": train + val}, in that order, for sweep_router.py /
                         fixed_program.py with --train-per-diff N_TRAIN --val-per-diff N_VAL
  --out-data-dir/diffN/  {train,val,test}.json question records, for router.infer
and prints "N_TRAIN N_VAL N_TEST" on its last line.

The MCTS file may be a released one without question text: question and answer come
from the split records.

  python -m re_polar.router.fold_data --level 1 --fold 0 \\
      --split-file ./data/polar_split_questions/pool/diff1/train.json \\
      --merged <mcts dir>/dart-math-diff-1/merged_mcts_samples.json.gz \\
      --out-samples ./results/folds/f0/samples_diff1.json --out-data-dir ./results/folds/f0
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from re_polar.datasets.schemas import load_samples


def fold_qids(
    qids: Sequence[str], fold: int, n_folds: int = 4, seed: int = 0
) -> Tuple[List[str], List[str], List[str]]:
    """(train, val, test) query_ids of one cross-validation fold."""
    qids = sorted(qids)
    random.Random(seed).shuffle(qids)
    n = len(qids)
    lo, hi = round(n * fold / n_folds), round(n * (fold + 1) / n_folds)
    test, rest = qids[lo:hi], qids[:lo] + qids[hi:]
    n_val = round(n / 8)
    return rest[n_val:], rest[:n_val], test


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--level", type=int, required=True)
    p.add_argument(
        "--split-file",
        required=True,
        help="<polar split>/pool/diffN/train.json (all of the level's questions)",
    )
    p.add_argument("--merged", required=True, help="merged_mcts_samples.json[.gz] of that level")
    p.add_argument("--out-samples", required=True)
    p.add_argument("--out-data-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="test fold (0-based)")
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument(
        "--fixed-split-dir",
        default=None,
        help="<polar split>/split: use its diffN/{train,val,test}.json instead of a fold",
    )
    args = p.parse_args(argv)
    if (args.fold is None) == (args.fixed_split_dir is None):
        raise SystemExit("pass exactly one of --fold and --fixed-split-dir")

    records = {r["query_id"]: r for r in json.load(open(args.split_file))}
    if args.fixed_split_dir:
        train, val, test = (
            [
                r["query_id"]
                for r in json.load(
                    open(Path(args.fixed_split_dir) / f"diff{args.level}" / f"{s}.json")
                )
            ]
            for s in ("train", "val", "test")
        )
        if set(train + val + test) != set(records):
            raise SystemExit("--fixed-split-dir covers a different question set than --split-file")
    else:
        train, val, test = fold_qids(list(records), args.fold, args.n_folds, args.seed)

    by_qid = {s["sample_info"]["query_id"]: s for s in load_samples(Path(args.merged))}
    missing = [q for q in train + val if q not in by_qid]
    if missing:
        raise SystemExit(
            f"{len(missing)} train/val questions have no MCTS sample, e.g. {missing[:3]}"
        )

    def sample(q: str, role: str) -> dict:
        s = {k: v for k, v in by_qid[q].items() if k != "search_trajectory"}  # not needed here
        return {
            **s,
            "question": records[q]["question"],
            "gt_ans": records[q]["gt_ans"],
            "sample_info": {**s["sample_info"], "split": role},
        }

    samples = [sample(q, "train") for q in train] + [sample(q, "val") for q in val]
    _write(Path(args.out_samples), {"samples": samples})
    ddir = Path(args.out_data_dir) / f"diff{args.level}"
    for name, qs in (("train", train), ("val", val), ("test", test)):
        _write(ddir / f"{name}.json", [records[q] for q in qs])
    print(
        f"DM-{args.level}: {len(records)} questions -> train {len(train)} / val {len(val)} / "
        f"test {len(test)} (fold {args.fold}, fixed split {args.fixed_split_dir}, seed "
        f"{args.seed}); wrote {args.out_samples} and {ddir}/",
        flush=True,
    )
    print(len(train), len(val), len(test))


if __name__ == "__main__":
    main()
