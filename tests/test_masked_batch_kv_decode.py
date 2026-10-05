"""End-to-end check of the KV-cached masked-batch decode
(`GenerationReward._masked_batch_decode_bucket_kv`) against the serial path.

`tests/test_mcts.py` only pins the scheduling core
(`_masked_batch_layer_pass_kv`) with fake layers. This drives the whole decode
loop (prefill, per-step cache growth, left-padding, RoPE logical positions,
the length-only mask cache) through a tiny random-weight Llama on CPU in fp32
and requires every row's tokens to equal what serial `model.generate` produces
on the rerouted model (`LayerEngine.apply_layer_rerouting`), i.e. what the
serial `GenerationReward.__call__` would generate. fp32 keeps batch-shape
rounding far below the logit gaps, so the comparison is exact token equality,
batched and alone.
"""

import os

import pytest
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from re_polar.core.executor import ProgramExecutor
from re_polar.core.layer_engine import LayerEngine, get_layers
from re_polar.mcts.rewards import GenerationReward

N_LAYERS = 4
MAX_NEW_TOKENS = 12
PAD = 0

# identity, skip, repeat mid, repeat at start, skip+repeat
PATHS = [
    [0, 1, 2, 3],
    [0, 2, 3],
    [0, 1, 1, 2, 3],
    [0, 0, 1, 2, 3],
    [1, 2, 2, 2, 3],
]
# different lengths so the batched decode has to left-pad
PROMPTS = ["5 9 3 1 7 22", "11 4", "8 8 13 40 2", "30 6 6", "17"]


class _WhitespaceTokenizer:
    """Prompt "5 9 3" -> ids [5, 9, 3]; left-pads with PAD."""

    pad_token_id = PAD
    padding_side = "left"

    def __call__(self, prompts, return_tensors="pt", padding=True):
        rows = [[int(t) for t in p.split()] for p in prompts]
        width = max(len(r) for r in rows)
        ids = [[PAD] * (width - len(r)) + r for r in rows]
        mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
        return BatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _engine(attn_implementation):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        # default 0.02 gives near-uniform attention, so RoPE position bugs
        # (e.g. ignoring left-padding in decode positions) don't change tokens
        initializer_range=0.5,
        pad_token_id=PAD,
        bos_token_id=None,
        eos_token_id=None,
    )
    config._attn_implementation = attn_implementation
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    # LayerEngine.__init__ downloads a checkpoint; wire the tiny model in directly
    engine = LayerEngine.__new__(LayerEngine)
    engine.model_id = "tiny-random-llama"
    engine.model, engine.tokenizer, engine.device = model, _WhitespaceTokenizer(), "cpu"
    engine._init_layer_access()
    engine._is_rerouted = False
    engine._patched_inner = None
    return engine


def _serial_tokens(engine, path, prompt):
    engine.apply_layer_rerouting(path)
    try:
        enc = engine.tokenizer([prompt])
        with torch.no_grad():
            out = engine.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=PAD
            )
    finally:
        engine.restore_original()
    return engine.tokenizer.decode(out[0, enc["input_ids"].shape[1] :])


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_kv_masked_batch_decode_matches_serial_generate(attn_implementation):
    engine = _engine(attn_implementation)
    reward = GenerationReward(
        ProgramExecutor(engine), max_new_tokens=MAX_NEW_TOKENS, masked_batch_kv=True
    )
    model = engine.model
    args = (get_layers(model), model.model, model, "cpu")

    serial = [_serial_tokens(engine, p, q) for p, q in zip(PATHS, PROMPTS)]
    # sanity: the programs really differ, else this test proves nothing
    assert len(set(serial)) == len(serial)

    alone = [
        reward._masked_batch_decode_bucket_kv([p], [q], *args)[0] for p, q in zip(PATHS, PROMPTS)
    ]
    assert alone == serial

    batched = reward._masked_batch_decode_bucket_kv(PATHS, PROMPTS, *args)
    assert batched == serial

    # the uncached decode is the same function of the inputs, just recomputed
    uncached = reward._masked_batch_decode_bucket(PATHS, PROMPTS, *args)
    assert uncached == serial

    # the model's own layer list is never touched by either masked-batch decode
    assert len(get_layers(model)) == N_LAYERS


QUESTIONS = [
    "What is 17 * 23?",
    "Solve for x: 3x + 7 = 31.",
    "A rectangle has perimeter 30 and width 5. What is its area?",
    "What is the remainder when 2^10 is divided by 7?",
]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("RE_POLAR_GPU_TESTS") == "1"),
    reason="needs CUDA and RE_POLAR_GPU_TESTS=1 (downloads Qwen3-8B)",
)
def test_kv_masked_batch_decode_bit_identical_to_serial_on_real_model():
    """Paper Appendix A.7 claim on the production setup (Qwen3-8B, bf16,
    flash-attention): a row decoded alone through the KV masked-batch path is
    bit-identical to serial cached generation. Once rows share forward calls
    batch-shape bf16 noise is allowed, so the batched run is only reported."""
    from re_polar.models import MODEL_REGISTRY

    engine = LayerEngine(MODEL_REGISTRY["qwen3_8b"]["model_id"])
    reward = GenerationReward(
        ProgramExecutor(engine), masked_batch_kv=True, prompt_style="paper_minimal_fewshot"
    )
    n = engine.num_layers
    paths = [
        list(range(n)),  # identity
        list(range(10)) + list(range(12, n)),  # skip 10-11
        list(range(20)) + list(range(16, n)),  # repeat 16-19
        list(range(5)) + [5, 5] + list(range(6, n)),  # single-layer repeat
    ]
    prompts = [reward._prompt_fn(engine.tokenizer, q) for q in QUESTIONS]
    model = engine.model
    args = (get_layers(model), model.model, model, reward.device)

    serial, alone = [], []
    for path, prompt in zip(paths, prompts):
        engine.apply_layer_rerouting(path)
        try:
            enc = engine.tokenizer([prompt], return_tensors="pt").to(reward.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=reward.max_new_tokens,
                    do_sample=False,
                    pad_token_id=engine.tokenizer.pad_token_id,
                )
        finally:
            engine.restore_original()
        serial.append(
            engine.tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        alone.append(reward._masked_batch_decode_bucket_kv([path], [prompt], *args)[0])

    batched = reward._masked_batch_decode_bucket_kv(paths, prompts, *args)
    for i, (s, a, b) in enumerate(zip(serial, alone, batched)):
        print(f"row {i}: alone==serial {a == s}, batched==serial {b == s}", flush=True)
        print(f"  serial : {s!r}", flush=True)
    assert alone == serial
