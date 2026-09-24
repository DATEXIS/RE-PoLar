"""Letter-bias probe generalized to MMLU-Pro's up-to-10-option questions.

The original letter-bias probe this generalizes measures A/B/C/D letter
preference on a content-free "what's your favorite letter" prompt by running
every one of 4!=24 orderings of the option list through every one of 20
prompt templates (480 prompts/forward-pass) and taking the argmax over the 4
choice logits. Running all orderings is what disentangles genuine
LETTER-identity bias from POSITION bias: if the model always picks whichever
option is listed first, every letter ends up at 25% in the average -- only a
letter-specific preference shifts a letter away from uniform.

MMLU-Pro is NOT 4-way: the large majority of `mmlu_pro_domains` rows have
exactly 10 options. Exhaustive 10-letter coverage is 10!=3,628,800 orderings
-- infeasible. This keeps the SAME disentangling property (every ordering is
a fresh independent draw, so averaging over enough of them still isolates
letter bias from position bias) by sampling K distinct random orderings
instead of enumerating all N!. N<=4 still uses full itertools.permutations
(cheap, matches the original exactly at N=4).

Pure stdlib logic (ordering sampling, prompt formatting, choice-token-id
generalization) is unit-testable without needing a real model.
`run_letter_bias_probe` itself needs a real `transformers` model/tokenizer --
see `letter_bias_probe_best_program_run.py` for the actual model-loading
driver (an exploratory diagnostic, not one of this pipeline's paper-cited
results -- see that file's own docstring for what those are).
"""

import random
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

_INDEX_TO_LETTER = list("ABCDEFGHIJ")

# Generic {options}-slot rewrite of the original 20 LETTER_BIAS_TEMPLATES
# phrasings, so the same phrasing set works for any num_options (the original
# hard-codes exactly 4 {a}{b}{c}{d} slots). At num_options=4 the joined
# `{options}` string reads identically to the original's "{a}, {b}, {c}, or
# {d}" phrasing.
LETTER_BIAS_TEMPLATES_GENERIC: List[str] = [
    "What's your favorite letter? {options}? Answer with one character only.",
    "Pick a letter: {options}. Reply with just that letter.",
    "Choose your preferred letter from {options}. Single character answer.",
    "Which letter do you like best? {options}? One character only.",
    "Select one letter: {options}. Answer with a single character.",
    "If you had to pick one letter, {options}, which would it be? One character.",
    "Your favorite letter is {options}? Give a one-character response.",
    "Name your favorite letter. Options: {options}. One character.",
    "Which letter do you prefer, out of {options}? One character answer.",
    "Is your favorite letter {options}? Answer with the letter only.",
    "Quick question: {options}, which is your favorite? One character.",
    "Out of {options}, pick the one you like most. Just the letter.",
    "Favorite letter? {options}. Respond with one character.",
    "Which single letter do you prefer: {options}?",
    "Tell me your favorite: {options}. One-character answer only.",
    "Of these letters, {options}, which is your favorite? One character.",
    "Please indicate your favorite letter by responding with {options}.",
    "{options}, which letter is your favorite? Single character response.",
    "Given the choices {options}, which letter would you choose as your favorite?",
    "Respond with one of: {options}, whichever is your favorite letter.",
]


def join_ordered_letters(letters: Sequence[str]) -> str:
    """["C", "A", "D", "B"] -> "C, A, D, or B" (Oxford-comma-free, matches the
    original templates' "{a}, {b}, {c}, or {d}" phrasing at any length)."""
    if len(letters) == 1:
        return letters[0]
    return ", ".join(letters[:-1]) + ", or " + letters[-1]


def sample_orderings(
    num_options: int,
    k: int,
    seed: int = 0,
) -> List[Tuple[str, ...]]:
    """K distinct orderings of the first `num_options` letters (A, B, C, ...).

    num_options<=4: exhaustive itertools.permutations, capped at k (matches the
    original script's behavior exactly at num_options=4, k=24 -> all 24).
    num_options>4: k distinct random draws (no duplicates) via a seeded RNG --
    collision probability is negligible for k in the hundreds against
    num_options=10's 3,628,800 possible orderings, so plain rejection sampling
    is cheap and simple."""
    letters = _INDEX_TO_LETTER[:num_options]
    if num_options <= 4:
        return list(permutations(letters))[:k]

    rng = random.Random(seed)
    seen = set()
    orderings: List[Tuple[str, ...]] = []
    while len(orderings) < k:
        ordering = tuple(rng.sample(letters, num_options))
        if ordering not in seen:
            seen.add(ordering)
            orderings.append(ordering)
    return orderings


def _get_choice_token_ids(tokenizer: PreTrainedTokenizerBase, num_options: int) -> List[int]:
    """One token id per letter A.. up to num_options."""
    if not (1 <= num_options <= len(_INDEX_TO_LETTER)):
        raise ValueError(f"num_options must be in [1, {len(_INDEX_TO_LETTER)}], got {num_options}")
    ids: List[int] = []
    for letter in _INDEX_TO_LETTER[:num_options]:
        for candidate in [f" {letter}", letter]:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
                break
        else:
            ids.append(tokenizer.encode(f" {letter}", add_special_tokens=False)[0])
    return ids


def build_prompts(num_options: int, k_orderings: int, seed: int = 0) -> List[str]:
    """Every (template, ordering) prompt -- len(LETTER_BIAS_TEMPLATES_GENERIC) *
    len(sample_orderings(...)) prompts, one per line, ready to tokenize."""
    orderings = sample_orderings(num_options, k_orderings, seed=seed)
    prompts = []
    for template in LETTER_BIAS_TEMPLATES_GENERIC:
        for ordering in orderings:
            prompts.append(template.format(options=join_ordered_letters(ordering)))
    return prompts


def _left_pad_batch(items: List[Dict], pad_token_id: int, device: str) -> Dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].shape[0]) for item in items)
    input_rows, mask_rows = [], []
    for item in items:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - int(ids.shape[0])
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), int(pad_token_id), dtype=ids.dtype, device=ids.device)
            pad_mask = torch.zeros((pad_len,), dtype=mask.dtype, device=mask.device)
            ids = torch.cat([pad_ids, ids], dim=0)
            mask = torch.cat([pad_mask, mask], dim=0)
        input_rows.append(ids)
        mask_rows.append(mask)
    return {
        "input_ids": torch.stack(input_rows, dim=0).to(device),
        "attention_mask": torch.stack(mask_rows, dim=0).to(device),
    }


def prepare_letter_bias_inputs(
    tokenizer: PreTrainedTokenizerBase,
    num_options: int,
    k_orderings: int = 24,
    seed: int = 0,
    device: Optional[str] = None,
) -> List[Dict]:
    prepared = []
    for prompt in build_prompts(num_options, k_orderings, seed=seed):
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        prepared.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return prepared


def run_letter_bias_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_options: int,
    k_orderings: int = 24,
    seed: int = 0,
    prepared_inputs: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """Run every sampled-ordering prompt in one batched forward pass, return
    per-letter frequencies over A..(num_options-th letter), summing to 1.0."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    if prepared_inputs is None:
        prepared_inputs = prepare_letter_bias_inputs(
            tokenizer, num_options, k_orderings=k_orderings, seed=seed, device=str(device)
        )

    choice_token_ids = _get_choice_token_ids(tokenizer, num_options)
    choice_ids_tensor = torch.tensor(choice_token_ids, device=device)

    batch = _left_pad_batch(prepared_inputs, int(pad_id), str(device))

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )

    last_logits = outputs.logits[:, -1, :]
    choice_logits = last_logits[:, choice_ids_tensor]
    predicted_indices = choice_logits.argmax(dim=-1).tolist()

    letters = _INDEX_TO_LETTER[:num_options]
    counts = {letter: 0 for letter in letters}
    for idx in predicted_indices:
        counts[letters[idx]] += 1

    n = len(prepared_inputs)
    return {letter: counts[letter] / n for letter in letters}
