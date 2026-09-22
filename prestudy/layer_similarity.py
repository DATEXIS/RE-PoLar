"""
Layer-wise representational similarity via symmetric KL divergence.

This is the code behind the paper's motivation-section figure on layer-wise
representational similarity: pairwise similarity across all 37 layer-states
of Qwen3-8B (the embedding output, plus the output of each of the 36 decoder
layers), on the same mmlu_pro_domains 6-domain prompt set `prestudy/sweep.py`'s
accuracy sweep uses.

Method: collect each layer-state's token-level hidden vectors over the
prompt set (no generation), fit a diagonal Gaussian per layer-state (mean +
variance over the hidden dimension), and compute the symmetric KL
divergence between every pair of diagonal Gaussians. Symmetric KL is a
DISTANCE (0 = identical); `distance_to_similarity` converts it to a [0, 1]
similarity (bright = redundant, matching the paper figure's color
convention) via exp(-distance / scale), scale = median positive
off-diagonal distance.

Defaults (`--n-per-domain 40 --max-tokens 20000 --batch-size 4`) match the
exact run that produced the paper's published figure: 240 prompts total
(40/domain x 6 domains), 20000 tokens/layer-state, stratified as the first
`n_per_domain` samples of each domain from the cached mmlu_pro_domains
fixture (sorted by domain name) -- not a fresh random draw, so the same
prompt set every time regardless of dataset-generation randomness elsewhere.

    python -m prestudy.layer_similarity --model qwen3_8b \\
        --output-dir ./prestudy/results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

EPS = 1e-8


def _ensure_padding_token_id(tokenizer, model) -> None:
    if tokenizer.pad_token_id is not None:
        return
    fallback_id = tokenizer.eos_token_id
    if fallback_id is None:
        cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
        fallback_id = cfg_eos[0] if isinstance(cfg_eos, list) and cfg_eos else cfg_eos
    if fallback_id is None:
        fallback_id = tokenizer.bos_token_id
    if fallback_id is None:
        fallback_id = tokenizer.unk_token_id
    if fallback_id is None:
        raise ValueError("Tokenizer has no pad/eos/bos/unk token id for padding fallback.")
    tokenizer.pad_token_id = int(fallback_id)


def build_prompts(n_per_domain: int = 40) -> List[str]:
    """The paper's own 6-domain mmlu_pro_domains prompt set, stratified: the
    first `n_per_domain` samples of each domain (sorted by domain name) from
    the cached 200-per-domain fixture -- matching the run that produced the
    published figure exactly, rather than requesting a fresh `n_per_domain`
    fixture (which would draw a different random sample per domain, see
    `re_polar.datasets.mmlu_pro_hf._generate_mmlu_pro_domain_fixture`)."""
    from re_polar.core.mmlu_pro_scoring import _build_prompt
    from re_polar.core.mmlu_pro_domain_eval import load_mmlu_pro_domain_samples

    samples = load_mmlu_pro_domain_samples()  # cached 200/domain fixture
    by_domain: Dict[str, List] = {}
    for s in samples:
        by_domain.setdefault(s.category, []).append(s)

    prompts = []
    for domain in sorted(by_domain):
        for s in by_domain[domain][:n_per_domain]:
            prompts.append(_build_prompt(s, no_think=True))
    return prompts


def collect_layer_token_vectors(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    batch_size: int = 4,
    max_tokens: int = 20000,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[str]]:
    """Token vectors for each layer-state on prompt tokens (no generation):
    the embedding output ("input_to_layer_0") plus the output of every
    decoder layer ("after_layer_0", ..., "after_layer_{N-1}") -- N+1
    layer-states total, matching the paper figure's "37 layer-states" for a
    36-layer model.

    Returns:
      vectors: list of arrays [n_tokens, hidden_size], one per layer-state
      labels: layer-state labels aligned to vectors
    """
    _ensure_padding_token_id(tokenizer, model)
    if tokenizer.pad_token is None and tokenizer.pad_token_id is not None:
        token_text = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
        if token_text is not None:
            tokenizer.pad_token = token_text
    tokenizer.padding_side = "left"
    model.eval()

    buffers: List[List[np.ndarray]] = None
    labels: List[str] = None

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"].bool()

        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)

        hidden_states = outputs.hidden_states  # len = num_layers + 1
        if hidden_states is None or len(hidden_states) < 2:
            raise ValueError("Model did not return hidden states for layer analysis.")
        n_layers = len(hidden_states) - 1

        if labels is None:
            labels = ["input_to_layer_0"] + [f"after_layer_{i}" for i in range(n_layers)]
            buffers = [[] for _ in range(len(labels))]

        for state_idx, state_tensor in enumerate(hidden_states):
            valid_vectors = state_tensor[attention_mask].float().cpu().numpy()
            if valid_vectors.shape[0] > 0:
                buffers[state_idx].append(valid_vectors)

    if labels is None:
        raise ValueError("No prompts were processed; cannot collect token vectors.")

    vectors: List[np.ndarray] = [np.concatenate(chunks, axis=0) for chunks in buffers]
    n_tokens = min(x.shape[0] for x in vectors)
    vectors = [x[:n_tokens] for x in vectors]

    if n_tokens > max_tokens:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_tokens, size=max_tokens, replace=False))
        vectors = [x[idx] for x in vectors]

    return vectors, labels


def gaussian_stats(layer_vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    means = np.stack([x.mean(axis=0) for x in layer_vectors], axis=0)
    vars_ = np.maximum(np.stack([x.var(axis=0) for x in layer_vectors], axis=0), EPS)
    return means, vars_


def symmetric_kl_diag_gaussian(
    mean_p: np.ndarray, var_p: np.ndarray, mean_q: np.ndarray, var_q: np.ndarray,
) -> float:
    """Symmetric KL between two diagonal Gaussians N(mean_p, var_p) and
    N(mean_q, var_q), summed over the hidden dimension: 0.5*(KL(p||q) +
    KL(q||p))."""
    var_p = np.maximum(var_p, EPS)
    var_q = np.maximum(var_q, EPS)
    diff = mean_p - mean_q
    kl_pq = 0.5 * np.sum(np.log(var_q / var_p) + (var_p + diff * diff) / var_q - 1.0)
    kl_qp = 0.5 * np.sum(np.log(var_p / var_q) + (var_q + diff * diff) / var_p - 1.0)
    return 0.5 * (kl_pq + kl_qp)


def symkl_distance_matrix(layer_vectors: List[np.ndarray]) -> np.ndarray:
    means, vars_ = gaussian_stats(layer_vectors)
    n = means.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = symmetric_kl_diag_gaussian(means[i], vars_[i], means[j], vars_[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def distance_to_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """Distance matrix -> similarity in [0,1] via exp(-d/scale), where scale
    is the median positive off-diagonal distance."""
    n = distance_matrix.shape[0]
    offdiag = distance_matrix[np.triu_indices(n, k=1)]
    positive = offdiag[offdiag > 0]
    scale = np.median(positive) if positive.size else 1.0
    sim = np.exp(-distance_matrix / max(scale, EPS))
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def main(argv=None) -> None:
    from re_polar.models import MODEL_REGISTRY
    from re_polar.core.model_loader import load_model_and_tokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=list(MODEL_REGISTRY.keys()))
    model_group.add_argument("--model-id", help="Arbitrary HuggingFace model id")
    p.add_argument("--n-per-domain", type=int, default=40,
                   help="mmlu_pro_domains samples per domain (paper default: 40, matching "
                        "the published figure's run -- see build_prompts docstring)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=20000,
                   help="Subsample token count per layer-state (paper default: 20000, "
                        "matching the published figure's run)")
    p.add_argument("--output-dir", default="./prestudy/results")
    args = p.parse_args(argv)

    if args.model:
        cfg = MODEL_REGISTRY[args.model]
        model_id = cfg["model_id"]
        trust_remote_code = cfg.get("trust_remote_code", True)
    else:
        model_id = args.model_id
        trust_remote_code = True

    model, tokenizer, device = load_model_and_tokenizer(model_id, trust_remote_code=trust_remote_code)

    prompts = build_prompts(n_per_domain=args.n_per_domain)
    print(f"Collecting layer-state token vectors on {len(prompts)} mmlu_pro_domains prompts ...")
    vectors, labels = collect_layer_token_vectors(
        model, tokenizer, prompts, device, batch_size=args.batch_size, max_tokens=args.max_tokens)
    print(f"Collected {len(labels)} layer-states with {vectors[0].shape[0]} tokens each.")

    distance = symkl_distance_matrix(vectors)
    similarity = distance_to_similarity(distance)

    tag = model_id.replace("/", "_").replace(".", "p")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"layer_similarity_{tag}_symkl_distance.npy", distance)
    np.save(out_dir / f"layer_similarity_{tag}_symkl_similarity.npy", similarity)

    summary = {
        "model_id": model_id, "num_layer_states": len(labels), "layer_labels": labels,
        "num_prompts": len(prompts), "num_tokens_per_layer_state": int(vectors[0].shape[0]),
    }
    (out_dir / f"layer_similarity_{tag}_symkl_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary saved -> {out_dir / f'layer_similarity_{tag}_symkl_summary.json'}")


if __name__ == "__main__":
    main()
