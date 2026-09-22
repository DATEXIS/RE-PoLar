"""MMLU-Pro benchmark helpers, log-likelihood scoring with up to 10 choices.

The sample type and prompt/tokenization helpers shared by
`mmlu_pro_domains.py` and the router/reward code.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase

_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class MMLUProSample:
    id: int
    question: str
    options: Tuple[str, ...]
    answer_index: int
    category: str


def _choice_token_ids(tokenizer: PreTrainedTokenizerBase, n: int) -> List[int]:
    ids = []
    for letter in _LETTERS[:n]:
        for candidate in [f" {letter}", letter]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
                break
        else:
            ids.append(tokenizer.encode(f" {letter}", add_special_tokens=False)[0])
    return ids


def _build_prompt(sample: MMLUProSample, no_think: bool) -> str:
    choices_text = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(sample.options))
    body = (
        f"The following is a multiple choice question. "
        f"Answer with only the letter of the correct option.\n\n"
        f"{sample.question}\n{choices_text}\n\nAnswer:"
    )
    return f"/no_think {body}" if no_think else body


def _left_pad(items: List[Dict], pad_id: int, device: str) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in items)
    rows_ids, rows_mask = [], []
    for item in items:
        ids, mask = item["input_ids"], item["attention_mask"]
        pad = max_len - ids.shape[0]
        if pad > 0:
            ids = torch.cat([torch.full((pad,), pad_id, dtype=ids.dtype, device=ids.device), ids])
            mask = torch.cat([torch.zeros(pad, dtype=mask.dtype, device=mask.device), mask])
        rows_ids.append(ids)
        rows_mask.append(mask)
    return {
        "input_ids": torch.stack(rows_ids).to(device),
        "attention_mask": torch.stack(rows_mask).to(device),
    }
