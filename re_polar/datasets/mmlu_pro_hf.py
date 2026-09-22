"""MMLU-Pro dataset loading -- cached domain-stratified fixture + full HF pull.

Two loaders: a domain-stratified subset (`load_mmlu_pro_domain_records`, the
paper's own mechanism for the OOD baseline track) and a full-test-set puller
(`load_mmlu_pro_full_test`) for building the official full split used by
`re_polar/datasets/mmlu_pro_domains.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"

DEFAULT_MMLU_PRO_DOMAINS_PATH = DATASETS_DIR / "mmlu_pro_domains_200per.json"

MMLU_PRO_TARGET_DOMAINS = ["math", "physics", "chemistry", "law", "history", "computer science"]


def load_mmlu_pro_full_test(n_samples: int | None = None) -> list[dict[str, Any]]:
    """Pull the full official TIGER-Lab/MMLU-Pro `test` split from HF.

    ``n_samples`` None or <0 -> all rows; N>0 -> N random rows (seed=42).
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=False)
    records = [
        {"id": i, "question": str(ds[i]["question"]), "options": list(ds[i]["options"]),
         "answer_index": int(ds[i]["answer_index"]), "category": str(ds[i].get("category", ""))}
        for i in range(len(ds))
    ]
    if n_samples and n_samples > 0:
        rng = random.Random(42)
        idx = sorted(rng.sample(range(len(records)), min(n_samples, len(records))))
        records = [dict(records[i], id=new_id) for new_id, i in enumerate(idx)]
    return records


def _generate_mmlu_pro_domain_fixture(
    out_path: Path,
    domains: list[str],
    n_per_domain: int = 200,
    seed: int = 42,
) -> None:
    """Generate fixture. n_per_domain=-1 uses all available samples per domain."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=False)
    rng = random.Random(seed)

    by_domain: dict[str, list] = {d: [] for d in domains}
    for row in ds:
        cat = str(row.get("category", ""))
        if cat in by_domain:
            by_domain[cat].append(row)

    samples: list[dict] = []
    for domain in domains:
        pool = by_domain[domain]
        if n_per_domain == -1:
            chosen = pool  # all available
        else:
            if len(pool) < n_per_domain:
                raise ValueError(
                    f"Domain '{domain}' only has {len(pool)} samples in MMLU-Pro test split, "
                    f"need {n_per_domain}"
                )
            chosen = rng.sample(pool, n_per_domain)
        for row in chosen:
            samples.append({
                "id": len(samples),
                "question": str(row["question"]),
                "options": list(row["options"]),
                "answer_index": int(row["answer_index"]),
                "category": str(row["category"]),
            })
        print(f"  {domain}: {len(chosen)} samples")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    n_label = "all" if n_per_domain == -1 else str(n_per_domain)
    print(
        f"Saved {len(samples)} MMLU-Pro domain samples "
        f"({n_label}/domain, domains={domains}, seed={seed}) -> {out_path}"
    )


def load_mmlu_pro_domain_records(
    dataset_path: str | Path | None = None,
    domains: list[str] | None = None,
    n_per_domain: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load stratified MMLU-Pro domain samples.

    n_per_domain=200, seed=42 -> uses the cached fixture (fast, no HF download).
    Any other n_per_domain    -> generates a fresh fixture named
                                mmlu_pro_domains_{n}per.json in datasets/.
    n_per_domain=-1           -> uses all available samples per domain.
    dataset_path              -> load directly from that file.
    """
    target_domains = domains or MMLU_PRO_TARGET_DOMAINS
    if dataset_path is not None:
        path = Path(dataset_path)
    elif n_per_domain == 200 and seed == 42 and not domains:
        path = DEFAULT_MMLU_PRO_DOMAINS_PATH
        if not path.exists():
            _generate_mmlu_pro_domain_fixture(
                path, domains=target_domains, n_per_domain=n_per_domain, seed=seed
            )
    elif n_per_domain == -1:
        path = DATASETS_DIR / "mmlu_pro_domains_allper.json"
        if not path.exists():
            _generate_mmlu_pro_domain_fixture(
                path, domains=target_domains, n_per_domain=-1, seed=seed
            )
    else:
        path = DATASETS_DIR / f"mmlu_pro_domains_{n_per_domain}per.json"
        if not path.exists():
            _generate_mmlu_pro_domain_fixture(
                path, domains=target_domains, n_per_domain=n_per_domain, seed=seed
            )
    records = json.loads(path.read_text(encoding="utf-8"))
    _validate_mmlu_pro(records)
    return records


def _validate_mmlu_pro(records: list[dict]) -> None:
    for i, r in enumerate(records):
        for f in ("question", "options", "answer_index"):
            if f not in r:
                raise ValueError(f"MMLU-Pro record {i}: missing '{f}'")
        if not (2 <= len(r["options"]) <= 10):
            raise ValueError(f"MMLU-Pro record {i}: {len(r['options'])} options (need 2-10)")
        r["answer_index"] = int(r["answer_index"])
        r.setdefault("category", "")
