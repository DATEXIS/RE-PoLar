"""Pull the FULL official TIGER-Lab/MMLU-Pro `test` split, CPU-only -- no
per-domain cap.

`re_polar.datasets.mmlu_pro_hf.load_mmlu_pro_domain_records`'s cached fixture
caps every domain at 200 rows and only covers 6 of the 14 official
categories -- a real undercount (official sizes range 381-1351/domain).
Use this script to run against the dataset's OWN full test split instead,
all 14 categories, no subsampling.

Output: <out-dir>/test.json (flat list, all 14 categories, full official
counts) + <out-dir>/manifest.json (source dataset/split/revision, total,
per-domain counts, generation timestamp) -- the manifest is the "log
everything" record for this dataset, independent of any one model run.

Usage:
  python -m re_polar.datasets.mmlu_pro_official_test --out-dir ./data/mmlu_pro_official
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    from re_polar.datasets.mmlu_pro_hf import load_mmlu_pro_full_test

    print("Pulling TIGER-Lab/MMLU-Pro 'test' split in full (no cap) ...", flush=True)
    records = load_mmlu_pro_full_test()
    counts = Counter(r["category"] for r in records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))

    manifest = {
        "source_dataset": "TIGER-Lab/MMLU-Pro",
        "source_split": "test",
        "n_total": len(records),
        "n_categories": len(counts),
        "per_category_n": dict(sorted(counts.items())),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {len(records)} rows across {len(counts)} categories -> {out_dir}/test.json",
          flush=True)
    for cat, n in sorted(counts.items()):
        print(f"  {cat:20s} {n}", flush=True)
    print(f"manifest -> {out_dir}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
