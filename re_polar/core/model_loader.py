"""Shared model and tokenizer loading with device detection."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_id: str,
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
):
    device = detect_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {model_id} on {device} ...")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            token=True,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            token=True,
            trust_remote_code=trust_remote_code,
        ).to(device)

    model.eval()
    _strip_vision_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=True, trust_remote_code=trust_remote_code
    )
    _ensure_pad_token(tokenizer, model)
    return model, tokenizer, device


def _strip_vision_modules(model) -> None:
    # Gemma-3-12B/27B load as Gemma3ForConditionalGeneration even for PT
    # checkpoints, pulling in a vision encoder and projector we never use.
    # Delete them immediately after load to reclaim VRAM.
    inner = getattr(model, "model", None)
    if inner is None:
        return
    stripped = False
    for attr in ("vision_tower", "multi_modal_projector"):
        if hasattr(inner, attr):
            delattr(inner, attr)
            stripped = True
    if stripped:
        torch.cuda.empty_cache()


def _ensure_pad_token(tokenizer, model) -> None:
    if tokenizer.pad_token_id is not None:
        return
    fallback = tokenizer.eos_token_id
    if fallback is None:
        cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
        fallback = cfg_eos[0] if isinstance(cfg_eos, list) else cfg_eos
    if fallback is None:
        fallback = tokenizer.bos_token_id or tokenizer.unk_token_id
    if fallback is None:
        raise ValueError("Cannot determine pad token id from tokenizer/model config.")
    tokenizer.pad_token_id = int(fallback)
    if tokenizer.pad_token is None:
        tok = tokenizer.convert_ids_to_tokens(int(fallback))
        if tok:
            tokenizer.pad_token = tok
