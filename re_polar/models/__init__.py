"""Model configuration registry -- all 6 models the paper evaluates."""

QWEN3_8B_CONFIG = {
    "model_id": "Qwen/Qwen3-8B",
    "num_layers": 36,
    "no_think": True,
    "trust_remote_code": True,
}

QWEN3_32B_CONFIG = {
    "model_id": "Qwen/Qwen3-32B",
    "num_layers": 64,
    "no_think": True,
    "trust_remote_code": True,
}

QWEN25_3B_CONFIG = {
    "model_id": "Qwen/Qwen2.5-3B-Instruct",
    "num_layers": 36,
    "no_think": False,
    "trust_remote_code": True,
}

QWEN25_7B_CONFIG = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "num_layers": 28,
    "no_think": False,  # not Qwen3 -- no /no_think prefix needed
    "trust_remote_code": True,
}

# MoE: loads ALL 60 experts regardless of the "2.7B active" branding, so it
# needs more memory than similarly-sized dense models -- budget generously.
QWEN15_MOE_A27B_CONFIG = {
    "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "num_layers": 24,
    "no_think": False,
    "trust_remote_code": True,
}

# HF-gated: accept Meta's license at
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct before downloading.
# Standard GQA full-attention. Used for baseline (Base-model) reproduction
# only -- excluded from the MCTS search / router training track (see the
# paper for the justification).
LLAMA32_3B_CONFIG = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "num_layers": 28,
    "no_think": False,
    "trust_remote_code": True,
}

MODEL_REGISTRY = {
    "qwen3_8b": QWEN3_8B_CONFIG,
    "qwen3_32b": QWEN3_32B_CONFIG,
    "qwen25_3b": QWEN25_3B_CONFIG,
    "qwen25_7b": QWEN25_7B_CONFIG,
    "qwen15_moe_a27b": QWEN15_MOE_A27B_CONFIG,
    # Baseline (Base-model) reproduction only -- excluded from the MCTS
    # search / router training track (see the paper for the justification).
    "llama32_3b": LLAMA32_3B_CONFIG,
}

__all__ = [
    "MODEL_REGISTRY",
    "QWEN3_8B_CONFIG",
    "QWEN3_32B_CONFIG",
    "QWEN25_3B_CONFIG",
    "QWEN25_7B_CONFIG",
    "QWEN15_MOE_A27B_CONFIG",
    "LLAMA32_3B_CONFIG",
]
