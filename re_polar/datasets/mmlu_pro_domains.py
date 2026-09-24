"""mmlu_pro_domains TRAIN/VAL/TEST split (own data + bigger models):
MCTS search pool + router held-out eval, built from `mmlu_pro_official`.

SOURCE: reads a LOCAL copy of TIGER-Lab/MMLU-Pro's full "test" split --
`mmlu_pro_official/test.json` (all 14 native categories, 12,032 rows, schema
id/question/options/answer_index/category; provenance in that dataset's own
manifest.json) -- not a fresh `datasets.load_dataset` pull, so every run
against the same local copy is exactly reproducible without a network
dependency.

This design uses a single source pool (`mmlu_pro_official`) for every
domain: each domain carves its own held-out TEST from that one pool, rather
than mixing a frozen fixture for some domains with a freshly-carved split
for others -- avoiding any per-domain source-union bookkeeping.

DOMAIN SET: default is `ALL_MMLU_PRO_DOMAINS`, all 14 of MMLU-Pro's native
categories, not the paper-aligned 13 (`PAPER_MMLU_PRO_DOMAINS`, still kept
below -- computer science is absent from every PoLar table, ICML Tables
3/8/9/10 -- pass `--domains` explicitly, or `PAPER_MMLU_PRO_DOMAINS`, to
reproduce a paper-table-comparable run).

CONSTRAINT: not every domain has enough rows for a disjoint 500/domain train
split after carving 200/domain for TEST -- history (381 total) and computer
science (410 total) are the two thinnest, leaving only 181 and 210
respectively for train+val. n_per_domain is capped per-domain at whatever's
left; the manifest records which domains were capped.

ROUTER SPLIT: ``--val-frac`` carves a per-domain VAL
split out of the sampled per-domain pool (each domain's rows are already
`rng.sample`-ordered, so a positional cut is a random split, deterministic in
`seed`). ``val_frac=0.15`` mirrors DART-Math's ~1250:250 (~16.7%) train:val
ratio, kept slightly lower here because domains capped by data availability
can least afford to lose train rows. TRAIN, VAL and the carved TEST are all
disjoint by construction (one partition per domain, carved TEST removed from
the pool before train/val are drawn) -- asserted below (crash loudly, not
just documented), covered by
``tests/test_mmlu_pro_domains_data.py::test_build_train_split_val_frac_is_disjoint_from_train_and_test``.

Output: <out_dir>/train.json + (if --val-frac > 0) <out_dir>/val.json +
(if --test-n-per-domain > 0) <out_dir>/test.json (flat list, same record
shape as `mmlu_pro_official`: id/question/options/answer_index/category) +
manifest.json.

Usage:
  python -m re_polar.datasets.mmlu_pro_domains --out-dir ./data/mmlu_pro_domains_14 --official-path ./data/mmlu_pro_official/test.json --val-frac 0.15
"""

import argparse
import json
import random
from pathlib import Path

TARGET_N_PER_DOMAIN = 500
SEED = 42

# All 14 of MMLU-Pro's native `category` values, as they appear in
# mmlu_pro_official's manifest.json -- the default domain set.
ALL_MMLU_PRO_DOMAINS = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

# The 13 MMLU-Pro subjects the PoLar paper reports (ICML Tables 3 / 8 / 9 / 10,
# in the papers' own column order). Table 10 is Qwen3-8B, our model, so those
# tables double as per-subject base/POLAR reference numbers. Kept for
# reproducing paper-table-comparable runs -- NOT the default (see module
# docstring); pass `--domains` with this list explicitly.
#
# NOTE it is 13, not MMLU-Pro's full 14: **computer science is absent from every
# PoLar table** (verified directly against the ICML PDF text).
PAPER_MMLU_PRO_DOMAINS = [
    "math",
    "physics",
    "chemistry",
    "law",
    "engineering",
    "other",
    "economics",
    "health",
    "psychology",
    "business",
    "biology",
    "philosophy",
    "history",
]

# Held-out TEST rows carved per domain.
TEST_N_PER_DOMAIN = 200
VAL_FRAC = 0.0  # CLI default 0.0 (no val split) keeps build_train_split's historical
# train-only behaviour; the router split job passes --val-frac 0.15


def _dedup_key(row: dict) -> tuple:
    """(category, question, options, answer_index) -- unique per row in the
    target domains (verified directly: 0 collisions across 5674 rows), even
    though bare question text alone collides for ~86 rows (mostly `law`,
    reworded variants of the same lead-in sentence with different options)."""
    return (
        str(row["category"]),
        str(row["question"]),
        tuple(str(o) for o in row["options"]),
        int(row["answer_index"]),
    )


def build_train_split(
    out_dir: Path,
    official_path: Path | str,
    seed: int = SEED,
    n_per_domain: int = TARGET_N_PER_DOMAIN,
    val_frac: float = VAL_FRAC,
    domains: list[str] | None = None,
    test_n_per_domain: int = TEST_N_PER_DOMAIN,
) -> dict:
    """Build a train (+optional val, +optional test) split over `domains`,
    sourced from the local `mmlu_pro_official` pool at `official_path`
    (id/question/options/answer_index/category rows -- see module docstring).

    TEST is carved fresh per domain (`test_n_per_domain` rows, sampled FIRST
    and removed from the pool before train/val are drawn) -- every domain
    works this way, drawing from the same single source pool (see module
    docstring).
    """
    official_path = Path(official_path)
    rows = json.loads(official_path.read_text())

    if not (0.0 <= val_frac < 1.0):
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    target_domains = list(domains) if domains else list(ALL_MMLU_PRO_DOMAINS)

    by_domain: dict[str, list] = {d: [] for d in target_domains}
    for row in rows:
        cat = str(row.get("category", ""))
        if cat in by_domain:
            by_domain[cat].append(row)
    empty = [d for d, rs in by_domain.items() if not rs]
    if empty:
        raise ValueError(
            f"no rows found for requested domain(s) {empty} in "
            f"{official_path}, check spelling against the dataset's "
            f"own `category` values"
        )

    rng = random.Random(seed)
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    test_samples: list[dict] = []
    manifest = {
        "source": "mmlu_pro_official (local, TIGER-Lab/MMLU-Pro full test split)",
        "official_path": str(official_path),
        "seed": seed,
        "target_n_per_domain": n_per_domain,
        "val_frac": val_frac,
        "requested_domains": target_domains,
        "test_n_per_domain": test_n_per_domain,
        "domains": {},
    }
    for domain in target_domains:
        pool = list(by_domain[domain])
        total = len(pool)
        # carve held-out TEST first, so it can never overlap the train/val
        # drawn from what's left.
        carved_test: list = []
        if test_n_per_domain > 0:
            n_test = min(test_n_per_domain, len(pool))
            carved_test = rng.sample(pool, n_test)
            carved_keys = {_dedup_key(r) for r in carved_test}
            pool = [row for row in pool if _dedup_key(row) not in carved_keys]
            for row in carved_test:
                test_samples.append(
                    {
                        "id": len(test_samples),
                        "question": str(row["question"]),
                        "options": list(row["options"]),
                        "answer_index": int(row["answer_index"]),
                        "category": str(row["category"]),
                    }
                )
        target = min(n_per_domain, len(pool))
        chosen = rng.sample(
            pool, target
        )  # already a random order -> a positional cut is a random split
        n_val = int(round(target * val_frac)) if val_frac > 0.0 else 0
        val_rows, train_rows = chosen[:n_val], chosen[n_val:]
        for row in train_rows:
            train_samples.append(
                {
                    "id": len(train_samples),
                    "question": str(row["question"]),
                    "options": list(row["options"]),
                    "answer_index": int(row["answer_index"]),
                    "category": str(row["category"]),
                }
            )
        for row in val_rows:
            val_samples.append(
                {
                    "id": len(val_samples),
                    "question": str(row["question"]),
                    "options": list(row["options"]),
                    "answer_index": int(row["answer_index"]),
                    "category": str(row["category"]),
                }
            )
        manifest["domains"][domain] = {
            "total_rows": total,
            "carved_test_rows": len(carved_test),
            "available_pool": len(pool),
            "target": n_per_domain,
            "actual": target,
            "capped": target < n_per_domain,
            "train": len(train_rows),
            "val": len(val_rows),
        }
        flag = " (CAPPED, insufficient pool)" if target < n_per_domain else ""
        print(
            f"  {domain}: total={total} carved_test={len(carved_test)} "
            f"available={len(pool)} -> train={len(train_rows)} val={len(val_rows)}{flag}"
        )

    # Standing invariant: TRAIN/VAL/TEST are pairwise disjoint by construction
    # (one partition per domain, TEST removed from the pool before train/val
    # are drawn) -- assert loudly rather than trusting the construction silently.
    train_keys = {_dedup_key(r) for r in train_samples}
    val_keys = {_dedup_key(r) for r in val_samples}
    test_keys = {_dedup_key(r) for r in test_samples}
    if train_keys & val_keys:
        raise AssertionError(f"train/val overlap: {len(train_keys & val_keys)} shared content keys")
    if test_keys & (train_keys | val_keys):
        raise AssertionError(
            f"carved test overlaps train/val: {len(test_keys & (train_keys | val_keys))} keys"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=1)
    manifest["n_total"] = len(train_samples)
    if val_frac > 0.0:
        with open(out_dir / "val.json", "w") as f:
            json.dump(val_samples, f, indent=1)
        manifest["n_val_total"] = len(val_samples)
    if test_samples:
        with open(out_dir / "test.json", "w") as f:
            json.dump(test_samples, f, indent=1)
        manifest["n_test_total"] = len(test_samples)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(train_samples)} train samples -> {out_dir / 'train.json'}")
    if val_frac > 0.0:
        print(f"Wrote {len(val_samples)} val samples -> {out_dir / 'val.json'}")
    if test_samples:
        print(f"Wrote {len(test_samples)} carved test samples -> {out_dir / 'test.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--official-path",
        required=True,
        help="path to mmlu_pro_official's test.json (id/question/options/"
        "answer_index/category rows) to build this split from",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-domain", type=int, default=TARGET_N_PER_DOMAIN)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRAC,
        help="per-domain fraction of the sampled train pool carved into "
        "val.json (0 = no val split, historical behaviour)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="MMLU-Pro categories to build. Default = ALL "
        "14 native categories (ALL_MMLU_PRO_DOMAINS). Pass "
        "PAPER_MMLU_PRO_DOMAINS's 13 (excludes computer science) to "
        "reproduce a paper-table-comparable run.",
    )
    parser.add_argument(
        "--test-n-per-domain",
        type=int,
        default=TEST_N_PER_DOMAIN,
        help="Held-out TEST rows carved per domain, before train/val are drawn. "
        "0 = carve no test at all (search/analysis-only run).",
    )
    args = parser.parse_args()
    build_train_split(
        Path(args.out_dir),
        official_path=args.official_path,
        seed=args.seed,
        n_per_domain=args.n_per_domain,
        val_frac=args.val_frac,
        domains=args.domains,
        test_n_per_domain=args.test_n_per_domain,
    )


if __name__ == "__main__":
    main()
