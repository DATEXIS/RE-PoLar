"""MMLU-Pro domain-stratified benchmark, 200 samples per domain, log-likelihood scoring.

Six domains: math, physics, chemistry, law, history, computer science.
Fixed seed=42 ensures the sample is not lucky, same questions every run.

Returns both an aggregate score and per-domain breakdown, so sweeps can be
analysed by domain without re-running.
"""

from typing import Dict, Iterable, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from re_polar.datasets.mmlu_pro_hf import load_mmlu_pro_domain_records, MMLU_PRO_TARGET_DOMAINS
from .mmlu_pro_scoring import (
    MMLUProSample,
    _choice_token_ids,
    _build_prompt,
    _left_pad,
)

_DOMAIN_KEYS = {d: d.replace(" ", "_") for d in MMLU_PRO_TARGET_DOMAINS}


def load_mmlu_pro_domain_samples(
    dataset_path=None,
    domains: Optional[List[str]] = None,
    n_per_domain: int = 200,
    seed: int = 42,
) -> List[MMLUProSample]:
    records = load_mmlu_pro_domain_records(
        dataset_path=dataset_path,
        domains=domains,
        n_per_domain=n_per_domain,
        seed=seed,
    )
    return [
        MMLUProSample(
            id=int(r.get("id", i)),
            question=str(r["question"]),
            options=tuple(str(o) for o in r["options"]),
            answer_index=int(r["answer_index"]),
            category=str(r.get("category", "")),
        )
        for i, r in enumerate(records)
    ]


def prepare_mmlu_pro_domain_inputs(
    tokenizer: PreTrainedTokenizerBase,
    samples: Optional[Iterable[MMLUProSample]] = None,
    no_think: bool = True,
    device: Optional[str] = None,
) -> List[Dict]:
    items = list(samples) if samples is not None else load_mmlu_pro_domain_samples()
    prepared = []
    for s in items:
        enc = tokenizer(_build_prompt(s, no_think), return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        if device:
            ids, mask = ids.to(device), mask.to(device)
        prepared.append(
            {
                "id": s.id,
                "category": s.category,
                "answer_index": s.answer_index,
                "num_choices": len(s.options),
                "input_ids": ids,
                "attention_mask": mask,
            }
        )
    return prepared


def run_mmlu_pro_domains(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Optional[Iterable[MMLUProSample]] = None,
    batch_size: int = 8,
    no_think: bool = True,
    prepared_inputs: Optional[List[Dict]] = None,
    return_details: bool = True,
    return_probs: bool = False,
) -> Dict:
    """Run the domain-stratified benchmark.

    Returns:
        {
            "average": float,           # aggregate across all domains
            "per_domain": {             # key = domain name with spaces replaced by _
                "math":            {"average": float, "n": int},
                "physics":         {"average": float, "n": int},
                "chemistry":       {"average": float, "n": int},
                "law":             {"average": float, "n": int},
                "history":         {"average": float, "n": int},
                "computer_science": {"average": float, "n": int},
            },
            "per_prompt": [...],        # only when return_details=True; each entry also
                                         # carries "probs": [float, ...] (softmax over that
                                         # item's choice logits, same order as its options)
                                         # when return_probs=True
            "n": int,
        }

    `return_probs` is purely additive and defaults to False: with no callers passing it,
    every existing call site gets byte-identical output to before this parameter existed.
    Ignored when return_details=False (there's no per_prompt
    entry to attach probabilities to).
    """
    device = str(next(model.parameters()).device)
    max_choices = 10
    all_choice_ids = _choice_token_ids(tokenizer, max_choices)

    items = (
        list(prepared_inputs)
        if prepared_inputs is not None
        else prepare_mmlu_pro_domain_inputs(tokenizer, samples, no_think=no_think, device=device)
    )
    if not items:
        return {"average": 0.0, "per_domain": {}, "per_prompt": [], "n": 0}

    total = 0.0
    domain_totals: Dict[str, float] = {}
    domain_counts: Dict[str, int] = {}
    per_prompt = []

    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            tensors = _left_pad(batch, int(tokenizer.pad_token_id), device)
            logits = model(**tensors, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            for item, logit_row in zip(batch, logits):
                n = item["num_choices"]
                choice_tensor = torch.tensor(all_choice_ids[:n], device=device)
                choice_logits = logit_row[choice_tensor]
                pred = choice_logits.argmax().item()
                score = 1.0 if pred == item["answer_index"] else 0.0
                total += score

                cat = item["category"]
                domain_totals[cat] = domain_totals.get(cat, 0.0) + score
                domain_counts[cat] = domain_counts.get(cat, 0) + 1

                if return_details:
                    entry = {
                        "id": item["id"],
                        "category": cat,
                        "expected": item["answer_index"],
                        "predicted": pred,
                        "score": score,
                    }
                    if return_probs:
                        entry["probs"] = choice_logits.softmax(dim=-1).tolist()
                    per_prompt.append(entry)

    per_domain = {
        _DOMAIN_KEYS.get(cat, cat.replace(" ", "_")): {
            "average": float(domain_totals[cat] / domain_counts[cat]),
            "n": domain_counts[cat],
        }
        for cat in domain_totals
    }

    return {
        "average": float(total / len(items)),
        "per_domain": per_domain,
        "per_prompt": per_prompt if return_details else [],
        "n": len(items),
    }
