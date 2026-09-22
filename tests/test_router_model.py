"""PolarRouter forward shapes + grammar-constrained decode.

Fast CPU tests drive the router with synthetic token embeddings (``embed_dim``
set, no encoder download). One opt-in smoke test loads the real
Qwen/Qwen3-Embedding-0.6B; it is skipped unless POLE_ROUTER_ENCODER_SMOKE=1.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, validate_program
from re_polar.router import DEFAULT_OPS, PolarRouter, decode

D = 12          # small target-model layer count for fast tests
EMBED = 16      # synthetic encoder hidden size
T = 7           # question token count
DM = 32         # router d_model


def build_router(num_layers=D, n_ops=3, **kw):
    torch.manual_seed(0)
    return PolarRouter(
        num_layers=num_layers,
        n_ops=n_ops,
        embed_dim=EMBED,
        d_model=DM,
        nheads=4,
        n_layer_blocks=2,
        **kw,
    )


def synthetic_tokens(batch=2, tokens=T, embed=EMBED, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokens, embed, generator=g)


# --------------------------------------------------------------------------- #
# forward shapes
# --------------------------------------------------------------------------- #
def test_forward_shapes_default_ops():
    router = build_router()
    hidden = synthetic_tokens(batch=3)
    seg_logits, op_logits = router(token_hidden_states=hidden)
    assert seg_logits.shape == (3, D)          # l^seg in R^D per example
    assert op_logits.shape == (3, D, 3)        # l^op in R^{D x n_ops}


def test_forward_no_encoder_download():
    # embed_dim path must not have loaded / attached any encoder.
    router = build_router()
    assert router.encoder is None
    assert router.embed_dim == EMBED


def test_forward_with_key_padding_mask():
    router = build_router()
    hidden = synthetic_tokens(batch=2)
    mask = torch.zeros(2, T, dtype=torch.bool)
    mask[:, -2:] = True  # pad the last two positions
    seg_logits, op_logits = router(token_hidden_states=hidden, key_padding_mask=mask)
    assert seg_logits.shape == (2, D)
    assert op_logits.shape == (2, D, 3)
    assert torch.isfinite(seg_logits).all() and torch.isfinite(op_logits).all()


def test_forward_rejects_wrong_embed_dim():
    router = build_router()
    with pytest.raises(ValueError):
        router(token_hidden_states=torch.randn(1, T, EMBED + 1))


def test_forward_requires_some_input():
    router = build_router()
    with pytest.raises(ValueError):
        router()


# --------------------------------------------------------------------------- #
# n_ops forward-compat (a 4th op class, no schema change)
# --------------------------------------------------------------------------- #
def test_n_ops_4_forward_compat():
    router = build_router(n_ops=4)
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens())
    assert op_logits.shape[-1] == 4
    assert router.op_head.out_features == 4
    # A 3-op decode still works on the first three columns.
    prog = decode(seg_logits[0], op_logits[0])
    validate_program(prog)
    assert prog.num_layers == D


def test_n_ops_below_three_rejected():
    with pytest.raises(ValueError):
        build_router(n_ops=2)


# --------------------------------------------------------------------------- #
# decode: valid, round-trip, deterministic, segment-length bound
# --------------------------------------------------------------------------- #
def test_decode_returns_valid_program():
    router = build_router()
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens())
    prog = decode(seg_logits[0], op_logits[0])
    validate_program(prog)                      # raises if invalid
    assert isinstance(prog, Program)
    assert prog.num_layers == D
    assert sum(len(s) for s in prog.segments) == D  # contiguous full cover


def test_decode_roundtrips_through_validate_program():
    router = build_router()
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens(seed=5))
    prog = decode(seg_logits[0], op_logits[0])
    restored = Program.from_dict(prog.to_dict())
    validate_program(restored)
    assert restored == prog
    assert restored.to_layer_path() == prog.to_layer_path()


def test_decode_is_deterministic():
    router = build_router()
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens(seed=3))
    a = decode(seg_logits[0], op_logits[0])
    b = decode(seg_logits[0], op_logits[0])
    assert a == b


def test_decode_accepts_leading_batch_of_one():
    router = build_router()
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens(batch=1))
    prog = decode(seg_logits, op_logits)        # (1,D) / (1,D,n_ops)
    validate_program(prog)
    assert prog.num_layers == D


@pytest.mark.parametrize("seg_fill", [-10.0, 10.0, 0.0])
def test_decode_no_segment_longer_than_max(seg_fill):
    # All-below-threshold (one long span split up) and all-above (all length-1).
    seg_logits = torch.full((D,), seg_fill)
    op_logits = torch.randn(D, 3, generator=torch.Generator().manual_seed(7))
    prog = decode(seg_logits, op_logits)
    validate_program(prog)
    assert prog.num_layers == D
    assert all(len(s) <= MAX_SEGMENT_LEN for s in prog.segments)
    assert all(len(s) >= 1 for s in prog.segments)


def test_decode_all_below_threshold_makes_max_len_segments():
    seg_logits = torch.full((D,), -10.0)        # no interior boundaries
    op_logits = torch.zeros(D, 3)
    prog = decode(seg_logits, op_logits)
    validate_program(prog)
    # D=12 with only forced MAX_SEGMENT_LEN splits -> 3 segments of length 4.
    assert [len(s) for s in prog.segments] == [MAX_SEGMENT_LEN] * (D // MAX_SEGMENT_LEN)


def test_decode_avoids_all_skip_program():
    # Op logits overwhelmingly favor SKIP (index 0) at every layer.
    seg_logits = torch.full((D,), -10.0)
    op_logits = torch.zeros(D, 3)
    op_logits[:, DEFAULT_OPS.index(Op.SKIP)] = 50.0
    prog = decode(seg_logits, op_logits)
    validate_program(prog)                      # would raise on all-skip (empty path)
    assert prog.to_layer_path()                 # non-empty execution path
    assert any(s.op is not Op.SKIP for s in prog.segments)


def test_decode_all_keep_is_identity_when_no_boundaries():
    seg_logits = torch.full((D,), -10.0)
    op_logits = torch.zeros(D, 3)
    op_logits[:, DEFAULT_OPS.index(Op.KEEP)] = 50.0
    prog = decode(seg_logits, op_logits)
    validate_program(prog)
    assert prog.is_identity()
    assert prog.to_layer_path() == list(range(D))


def test_decode_repeat_carries_times_param():
    seg_logits = torch.full((D,), -10.0)
    op_logits = torch.zeros(D, 3)
    op_logits[:, DEFAULT_OPS.index(Op.REPEAT)] = 50.0
    prog = decode(seg_logits, op_logits)
    validate_program(prog)
    assert all(s.op is Op.REPEAT for s in prog.segments)
    assert all(s.times >= 2 for s in prog.segments)


def test_decode_layer_count_matches_D_via_registry_sized_router():
    # D is wired from MODEL_REGISTRY, never hard-coded to 36 here.
    router = PolarRouter.for_model("qwen3_8b", embed_dim=EMBED, d_model=DM)
    assert router.num_layers == 36
    seg_logits, op_logits = router(token_hidden_states=synthetic_tokens())
    prog = decode(seg_logits[0], op_logits[0])
    validate_program(prog)
    assert prog.num_layers == 36


# --------------------------------------------------------------------------- #
# injectable encoder (no download): a stub encoder standing in for the real one
# --------------------------------------------------------------------------- #
class _StubEncoder(torch.nn.Module):
    """Minimal encoder: exposes .config.hidden_size and returns last_hidden_state."""

    class _Cfg:
        hidden_size = EMBED

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.proj = torch.nn.Linear(EMBED, EMBED)

    def forward(self, **inputs):
        x = inputs["inputs_embeds"]
        from types import SimpleNamespace

        return SimpleNamespace(last_hidden_state=self.proj(x))


def test_injectable_encoder_is_frozen_and_sizes_embed_dim():
    enc = _StubEncoder()
    router = PolarRouter(num_layers=D, encoder=enc, d_model=DM)
    assert router.embed_dim == EMBED
    assert router.encoder is enc
    assert all(not p.requires_grad for p in router.encoder.parameters())
    assert not router.encoder.training  # eval mode
    # Router head params stay trainable.
    assert router.q_proj.weight.requires_grad


def test_train_keeps_encoder_in_eval():
    router = PolarRouter(num_layers=D, encoder=_StubEncoder(), d_model=DM)
    router.train()
    assert router.training
    assert not router.encoder.training


# --------------------------------------------------------------------------- #
# opt-in smoke: real Qwen3-Embedding-0.6B forward pass (download OK)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("POLE_ROUTER_ENCODER_SMOKE") != "1",
    reason="set POLE_ROUTER_ENCODER_SMOKE=1 to load the real encoder (downloads weights)",
)
def test_real_encoder_forward_and_decode():
    router = PolarRouter.for_model("qwen3_8b")  # loads Qwen/Qwen3-Embedding-0.6B
    router.eval()
    seg_logits, op_logits = router(questions=["What is 2 + 2?", "Compute 7 * 8."])
    assert seg_logits.shape == (2, 36)
    assert op_logits.shape == (2, 36, 3)
    prog = decode(seg_logits[0], op_logits[0])
    validate_program(prog)
    assert prog.num_layers == 36
