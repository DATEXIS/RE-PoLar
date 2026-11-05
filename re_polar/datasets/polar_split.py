"""Li et al.'s own DART-Math question sets ("the PoLar split", paper Appendix B).

Their released search code (github.com/tianyi-lab/PoLar, mcts_discovery/run_mcts.py,
`sample_dataset(level=N, sample_size=2000)`) takes the first 2000 ROWS of
hkust-nlp/dart-math-pool-math whose query_metadata["level"] == N, in dataset row order,
and dedupes by query_id keeping the first occurrence. Because the pool stores many rows
per question this is not "all level-N questions" but 565 / 1349 / 1579 / 1537 / 1577
questions for DM-1..5 (MATH only, no GSM8K). Re-implemented here; no code of theirs is
imported.

Their paper's train/val/test parts (62.5 / 12.5 / 25 %) are not in their code, so the
cut is inferred: their sample order, contiguous, train = first round(.625 n), val = next
round(.125 n), test = the rest. This reproduces their stated totals 4,130 / 826 / 1,651
exactly (test 141 / 337 / 395 / 384 / 394). On these test sets LLaMA-3.2-3B's Base is
within 4.1 points of their reported numbers, summed over the five levels; none of 2,000
random test sets of the same sizes from the same pools comes this close (median gap 27
points; paper Appendix B), so we take these test sets to be theirs.

Writes, per level N:
  <out-dir>/pool/diffN/train.json         every question of the level, sorted by query_id:
                                          the search input (run_search_dart_math
                                          --data-dir <out-dir>/pool --split train) and
                                          the --dart-math-dir for attach_mcts_questions
  <out-dir>/split/diffN/{train,val,test}.json   their inferred train/val/test cut
Records have the same fields as re_polar.datasets.dart_math's (question text and answer
from the public pool), so nothing here is re-hosted, see data/DATA.md.

  python -m re_polar.datasets.polar_split --out-dir ./data/polar_split_questions

The router's 4-fold cross-validation over the pool is `fold_qids` (re_polar.router.fold_data).
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from re_polar.datasets.dart_math import (
    POOL_DATASET,
    POOL_QUERY_BOUNDS,
    QUERY_INFO_DATASET,
    load_pool_queries,
)

# distinct questions their sample_dataset() yields per level
EXPECTED = {1: 565, 2: 1349, 3: 1579, 4: 1537, 5: 1577}
SAMPLE_SIZE = 2000  # their --sample_size default
SPLIT_TOTALS = (4130, 826, 1651)  # train / val / test summed over levels, as their paper states


def polar_query_ids(levels) -> Dict[int, List[str]]:
    """query_ids per level, in their sample_dataset() order."""
    import datasets

    ds = datasets.load_dataset(POOL_DATASET, split="train").select_columns(
        ["query_id", "query_metadata"]
    )
    rows: Dict[int, List[str]] = {lvl: [] for lvl in levels}
    for qid, meta in zip(ds["query_id"], ds["query_metadata"]):
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)
        lvl = meta.get("level", 0)
        if lvl in rows and len(rows[lvl]) < SAMPLE_SIZE:
            rows[lvl].append(qid)
    return {lvl: list(dict.fromkeys(r)) for lvl, r in rows.items()}


def split_sizes(n: int) -> Tuple[int, int, int]:
    """(train, val, test) sizes of a level with n questions."""
    n_train, n_val = round(0.625 * n), round(0.125 * n)
    return n_train, n_val, n - n_train - n_val


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--levels", default="1,2,3,4,5")
    args = p.parse_args(argv)
    levels = [int(x) for x in args.levels.split(",")]

    sizes = {lvl: split_sizes(n) for lvl, n in EXPECTED.items()}
    assert tuple(sum(s[i] for s in sizes.values()) for i in range(3)) == SPLIT_TOTALS

    print(f"reading {POOL_DATASET} (their sample order) ...", flush=True)
    qids = polar_query_ids(levels)
    print("joining question records ...", flush=True)
    records = {
        r["query_id"]: r
        for r in load_pool_queries(
            POOL_DATASET,
            QUERY_INFO_DATASET,
            bounds=POOL_QUERY_BOUNDS,
            source="math",
            has_level=True,
        )
    }
    out = Path(args.out_dir)
    for i, lvl in enumerate(levels, 1):
        if len(qids[lvl]) != EXPECTED[lvl]:
            raise ValueError(
                f"DM-{lvl}: {len(qids[lvl])} questions, expected {EXPECTED[lvl]}; "
                f"did {POOL_DATASET} change upstream?"
            )
        recs = [records[q] for q in qids[lvl]]
        assert all(r["math_level"] == lvl for r in recs)
        _write(
            out / "pool" / f"diff{lvl}" / "train.json", sorted(recs, key=lambda r: r["query_id"])
        )
        n_train, n_val, n_test = sizes[lvl]
        parts = {
            "train": recs[:n_train],
            "val": recs[n_train : n_train + n_val],
            "test": recs[n_train + n_val :],
        }
        for name, rs in parts.items():
            _write(out / "split" / f"diff{lvl}" / f"{name}.json", rs)
        print(
            f"[{i}/{len(levels)}] DM-{lvl}: {len(recs)} questions -> "
            f"train {n_train} / val {n_val} / test {n_test}",
            flush=True,
        )
    print(f"PoLar split written to {out}/pool and {out}/split", flush=True)


if __name__ == "__main__":
    main()
