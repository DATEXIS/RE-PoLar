"""DART-Math (DM1-5) sourcing, difficulty binning, and splits.

Difficulty definition:
- DART-Math (Tong et al., NeurIPS'24, arXiv:2407.13690) defines difficulty as
  a CONTINUOUS per-query **fail rate** = wrong responses / all sampled
  responses, measured with DeepSeekMath-7B-RL. It has no discrete levels.
- PoLar's "DM-1..DM-5" is never constructed in their paper or repo. MATH's
  own level labels are ruled out by counts (level 1: only 564 train records
  vs the 1250+250 needed). The arithmetic 5 x (1250+250) = 7500 = exact size
  of the DART-Math MATH-train pool implies: rank pool queries by fail rate,
  cut into 5 equal quantile bins of 1500 (DM-1 = easiest .. DM-5 = hardest).
  That is what we implement.
- Fail rates come from the authors' own published per-query stats:
  `hkust-nlp/dart-math-pool-math-query-info` (pass_rate over their full
  DeepSeekMath-7B-RL sampling; fail rate = 1 - pass_rate). Question text and
  gt answers come from `hkust-nlp/dart-math-pool-math`, joined on query_id.
  (Note: the released pool contains ONLY correct responses, all 1,615,233
  rows have ans_correct=True, verified directly, so fail rates could not
  be recomputed from it; the query-info artifact is the authoritative source.)

With `--include-gsm8k`, the MATH pool is combined with `hkust-nlp/dart-
math-pool-gsm8k` (same fail-rate-bounded construction) before binning,
giving each difficulty level enough queries for the paper's exact
1250 train / 250 val / 500 test split. This is the split actually used
throughout this repo. Without `--include-gsm8k`, only the MATH pool is
binned; nobody has published fail rates for MATH-test, so that mode has
no test split (train/val only, val = the rest of the bin).

Output: <out_dir>/diff{N}/{train,val[,test]}.json + manifest.json, records:
  {"query_id", "question", "gt_ans", "domain", "math_level", "fail_rate",
   "tot_n_samples", "difficulty"}

Usage:
  python -m re_polar.datasets.dart_math --out-dir ./data/dart_math --include-gsm8k
"""

import argparse
import ast
import json
import random
from pathlib import Path

POOL_DATASET = "hkust-nlp/dart-math-pool-math"
QUERY_INFO_DATASET = "hkust-nlp/dart-math-pool-math-query-info"
# GSM8K pool (for the data-expansion `dart_math_v2` namespace).
# Same schema as the MATH pool (query/gt_ans/domain/query_id + pass_rate in
# query-info) EXCEPT query_metadata carries `n_step`, not `level` (GSM8K has no
# MATH levels). Combining MATH+GSM8K and binning by fail rate reproduces the
# paper's D.1 (10k, 2000/level): GSM8K (easier) fills the low-fail-rate bins,
# raising the base accuracy toward their ~0.416 and giving the router headroom.
GSM8K_POOL_DATASET = "hkust-nlp/dart-math-pool-gsm8k"
GSM8K_QUERY_INFO_DATASET = "hkust-nlp/dart-math-pool-gsm8k-query-info"
# mirror of the delisted hendrycks/competition_math (verified field-identical;
# 7 subject configs, train/test splits preserved), used for the TEST split
MATH_MIRROR = "EleutherAI/hendrycks_math"
SUBJECT_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
# The public MATH pool holds 7473 distinct queries, NOT the full 7500 MATH-train
# (27 filtered upstream; verified directly). MATH-only near-equal bins are
# ~1494-1495 so 1250+250 per bin is unreachable: MATH-only keeps train=1250 and
# lets val absorb the shortfall (~244). The COMBINED (MATH+GSM8K) pool is ~15k
# -> ~3000/bin, so v2 takes the paper's exact 1250 train + 250 val per bin.
POOL_QUERY_BOUNDS = (7400, 7500)
GSM8K_QUERY_BOUNDS = (6900, 7600)  # ~7473 GSM8K-train queries; loose guard
TRAIN_SIZE = 1250
VAL_SIZE = 250  # v2 (combined pool) caps val at the paper's 250/level
TEST_SIZE = 500  # per level; pending the MATH-test fail-rate job
N_BINS = 5
SEED = 42


def extract_boxed_answer(solution: str) -> str | None:
    """Content of the last \\boxed{...}; None if absent/unbalanced.

    Mirrors dart-math's extract_ans_from_math_sol semantics. Needed for the
    pending MATH-test split (EleutherAI/hendrycks_math mirror), where answers
    come as full solutions.
    """
    start = solution.rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i:j]
    return None


def parse_level(level_str: str) -> int | None:
    """'Level 3' -> 3; 'Level ?' -> None. (MATH metadata, kept for analysis.)"""
    token = level_str.split(" ")[-1]
    return None if token == "?" else int(token)


def load_math_records(split: str):
    """MATH via the mirror; drops 'Level ?' and no-boxed-answer records (counted)."""
    import datasets

    records, dropped_level, dropped_ans = [], 0, 0
    for config in SUBJECT_CONFIGS:
        for idx, dp in enumerate(datasets.load_dataset(MATH_MIRROR, config, split=split)):
            level = parse_level(dp["level"])
            if level is None:
                dropped_level += 1
                continue
            gt_ans = extract_boxed_answer(dp["solution"])
            if gt_ans is None:
                dropped_ans += 1
                continue
            records.append(
                {
                    "question": dp["problem"],
                    "gt_ans": gt_ans,
                    "level": level,
                    "domain": dp["type"].replace(" ", ""),
                    "source_split": split,
                    "source_index": f"{config}/{idx}",
                }
            )
    print(
        f"{MATH_MIRROR}:{split}: {len(records)} records "
        f"({dropped_level} dropped for 'Level ?', {dropped_ans} for no boxed answer)"
    )
    return records


def load_pool_queries(
    pool_dataset=POOL_DATASET,
    query_info_dataset=QUERY_INFO_DATASET,
    *,
    bounds=POOL_QUERY_BOUNDS,
    source="math",
    has_level=True,
):
    """Per-query records: text/answers from `pool_dataset` + authors' fail rates
    from `query_info_dataset`, joined on query_id. `source` tags the origin
    (math/gsm8k); `has_level` reads MATH's query_metadata['level'] (GSM8K has none)."""
    import datasets

    ds = datasets.load_dataset(pool_dataset, split="train").select_columns(
        ["query_id", "query", "gt_ans", "domain", "query_metadata"]
    )
    queries = {}
    for row in ds:
        if row["query_id"] in queries:
            continue
        meta = row["query_metadata"]
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)
        math_level = None
        if has_level and isinstance(meta, dict) and "level" in meta:
            math_level = int(meta["level"])
        queries[row["query_id"]] = {
            "query_id": row["query_id"],
            "question": row["query"],
            "gt_ans": row["gt_ans"],
            "domain": row["domain"],
            "math_level": math_level,
            "source": source,
        }

    info = datasets.load_dataset(query_info_dataset, split="train")
    matched, info_only = [], 0
    for row in info:
        q = queries.get(row["query_id"])
        if q is None:
            info_only += 1
            continue
        q["fail_rate"] = 1.0 - float(row["pass_rate"])
        q["tot_n_samples"] = int(row["tot_n_samples"])
        matched.append(q)

    pool_only = len(queries) - len(matched)
    print(
        f"[{source}] pool queries={len(queries)}, query-info rows={len(info)}, "
        f"joined={len(matched)} (pool-only={pool_only}, info-only={info_only})"
    )
    if not (bounds[0] <= len(matched) <= bounds[1]):
        raise ValueError(f"[{source}] joined {len(matched)} queries, outside {bounds}")
    return matched


def build_train_val(
    out_dir: Path, seed: int = SEED, include_gsm8k: bool = False, full: bool = False
):
    records = load_pool_queries(
        POOL_DATASET, QUERY_INFO_DATASET, bounds=POOL_QUERY_BOUNDS, source="math", has_level=True
    )
    n_math, n_gsm8k = len(records), 0
    if include_gsm8k:
        gsm = load_pool_queries(
            GSM8K_POOL_DATASET,
            GSM8K_QUERY_INFO_DATASET,
            bounds=GSM8K_QUERY_BOUNDS,
            source="gsm8k",
            has_level=False,
        )
        n_gsm8k = len(gsm)
        records = records + gsm
    # ascending fail rate; query_id tie-break keeps bin edges deterministic
    records.sort(key=lambda r: (r["fail_rate"], r["query_id"]))

    # v2 (combined pool, ~3000/bin) takes the paper's exact 1250+250; MATH-only
    # keeps its historical behaviour (val = bin remainder, ~244). --full overrides:
    # keep the WHOLE bin (train=1250, val=remainder) so the full ~15k merge is written.
    val_size = None if full else (VAL_SIZE if include_gsm8k else None)
    manifest = {
        "pools": (["math", "gsm8k"] if include_gsm8k else ["math"]),
        "pool_distinct_queries": {"math": n_math, "gsm8k": n_gsm8k, "total": len(records)},
        "difficulty": "fail-rate quantile bins (DM-1 easiest .. DM-5 hardest); see module docstring",
        "seed": seed,
        "train_size": TRAIN_SIZE,
        "val_size": (val_size if val_size is not None else "bin size - train size (MATH-only)"),
        "test_split": (
            f"{TEST_SIZE}/level held-out from the combined pool (paper D.1)"
            if val_size is not None
            else f"PENDING: MATH-test fail-rate job, then {TEST_SIZE}/level"
        ),
        "levels": {},
    }
    # near-equal bins covering all records (sizes differ by at most 1)
    bounds = [round(i * len(records) / N_BINS) for i in range(N_BINS + 1)]

    for level in range(1, N_BINS + 1):
        binned = records[bounds[level - 1] : bounds[level]]
        # combined pool -> paper's exact 1250 train / 250 val / 500 test (2000/level = 10k).
        need = (TRAIN_SIZE + val_size + TEST_SIZE) if val_size is not None else (TRAIN_SIZE + 1)
        if len(binned) < need:
            raise ValueError(f"Bin {level} has {len(binned)} records, need >= {need}")
        for r in binned:
            r["difficulty"] = level
        rng = random.Random((seed, level).__hash__())
        rng.shuffle(binned)
        if val_size is not None:
            splits = {
                "train": binned[:TRAIN_SIZE],
                "val": binned[TRAIN_SIZE : TRAIN_SIZE + val_size],
                "test": binned[TRAIN_SIZE + val_size : TRAIN_SIZE + val_size + TEST_SIZE],
            }
        else:
            splits = {"train": binned[:TRAIN_SIZE], "val": binned[TRAIN_SIZE:]}

        lvl_dir = out_dir / f"diff{level}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        for name, recs in splits.items():
            with open(lvl_dir / f"{name}.json", "w") as f:
                json.dump(recs, f, indent=1)
        src_counts = {}
        for r in binned:
            src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
        manifest["levels"][level] = {
            **{k: len(v) for k, v in splits.items()},
            "bin_total": len(binned),
            "source_mix": src_counts,
            "fail_rate_range": [
                min(r["fail_rate"] for r in binned),
                max(r["fail_rate"] for r in binned),
            ],
        }
        fr = manifest["levels"][level]["fail_rate_range"]
        print(
            f"diff{level}: train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits.get('test', []))} mix={src_counts} "
            f"fail_rate in [{fr[0]:.3f}, {fr[1]:.3f}]"
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    if val_size is not None:
        print(f"Wrote train(1250)/val(250)/TEST({TEST_SIZE}) per level from the combined pool.")
    else:
        print("NOTE: test split not built (MATH-only), pending MATH-test fail-rate job.")
    return manifest


def build_test(out_dir: Path, fail_rates_path: Path, seed: int = SEED):
    """Bin MATH-test by OUR measured fail rates (measure_test_fail_rates.py),
    same construction as train/val: 5 near-equal quantile bins, 500 sampled each."""
    measured = {}
    for f in sorted(fail_rates_path.parent.glob(fail_rates_path.name.replace(".json", "*.json"))):
        measured.update(json.loads(f.read_text()))  # merges shards
    records = [{"query_id": qid, **rec} for qid, rec in measured.items()]
    records.sort(key=lambda r: (r["fail_rate"], r["query_id"]))
    print(f"{len(records)} measured test queries from {fail_rates_path}")

    bounds = [round(i * len(records) / N_BINS) for i in range(N_BINS + 1)]
    for level in range(1, N_BINS + 1):
        binned = records[bounds[level - 1] : bounds[level]]
        if len(binned) < TEST_SIZE:
            raise ValueError(f"Test bin {level} has {len(binned)} records < {TEST_SIZE}")
        for r in binned:
            r["difficulty"] = level
        rng = random.Random((seed, "test", level).__hash__())
        rng.shuffle(binned)
        lvl_dir = out_dir / f"diff{level}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        with open(lvl_dir / "test.json", "w") as f:
            json.dump(binned[:TEST_SIZE], f, indent=1)
        fr = (min(r["fail_rate"] for r in binned), max(r["fail_rate"] for r in binned))
        print(f"diff{level}: test={TEST_SIZE} fail_rate in [{fr[0]:.3f}, {fr[1]:.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--include-gsm8k",
        action="store_true",
        help="combine MATH+GSM8K pools (dart_math_v2 = paper D.1); default MATH-only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="keep the WHOLE bin (train=1250, val=remainder) instead of capping "
        "val at 250 -> writes the full ~15k merge, not the 10k subset",
    )
    parser.add_argument(
        "--test-fail-rates",
        default=None,
        help="path to test_fail_rates.json -> builds ONLY the test split",
    )
    args = parser.parse_args()
    if args.test_fail_rates:
        build_test(Path(args.out_dir), Path(args.test_fail_rates), seed=args.seed)
    else:
        build_train_val(
            Path(args.out_dir), seed=args.seed, include_gsm8k=args.include_gsm8k, full=args.full
        )


if __name__ == "__main__":
    main()
