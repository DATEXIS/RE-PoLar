"""Standing fidelity guard: our router's architecture/loss/decode constants vs
PoLar's own released code.

So a silent future edit to either side's defaults gets caught instead of
re-derived by hand again.

Every assertion below is checked against a hand-transcribed value from
PoLar's own published model.py/train.py, cited inline by symbol name. This
file does not import from PoLar's own code at all, every value here is a
literal we verified once by reading their code/paper, not something
re-derived by executing it.

One formerly-documented default divergence (repeat-count path parsing) is
covered explicitly below, not silently matched: our default
(``strict_repeat_2x=True``) was changed to match PoLar's own 2x-only
parser, closing what used to be the one real difference.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from re_polar.core import MAX_SEGMENT_LEN
from re_polar.router import DEFAULT_EMBEDDING_MODEL, PolarRouter, decode
from re_polar.router.train import _trainable_params, compute_loss, program_from_layer_path

EMBED = 16  # synthetic encoder hidden size, keeps this file CPU-fast


def build_router(num_layers=12, **kw):
    torch.manual_seed(0)
    return PolarRouter(num_layers=num_layers, embed_dim=EMBED, **kw)


# --------------------------------------------------------------------------- #
# architecture: PoLar's PolarPredictor.__init__ defaults
# --------------------------------------------------------------------------- #
def test_embedding_model_matches_polar():
    # polar/model.py: embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    assert DEFAULT_EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"


def test_architecture_defaults_match_polar():
    sig = inspect.signature(PolarRouter.__init__)
    defaults = {name: p.default for name, p in sig.parameters.items()}
    # polar/model.py: d_model=256, nheads=4, n_layer_blocks=2
    assert defaults["d_model"] == 256
    assert defaults["nheads"] == 4
    assert defaults["n_layer_blocks"] == 2
    # polar/model.py: dropout=0.1 hardcoded on both cross_attn and encoder layer
    assert defaults["dropout"] == 0.1
    # polar/model.py: tokenizer(..., truncation=True, max_length=512)
    assert defaults["max_question_tokens"] == 512
    # polar/model.py: op_head = nn.Linear(d_model, 3) (hardcoded 3 ops: skip/keep/repeat)
    assert defaults["n_ops"] == 3


def test_head_shapes_match_polar():
    router = build_router(d_model=32, n_ops=3)
    # polar/model.py: seg_head = nn.Linear(d_model, 1)
    assert tuple(router.seg_head.weight.shape) == (1, 32)
    # polar/model.py: op_head = nn.Linear(d_model, 3)
    assert tuple(router.op_head.weight.shape) == (3, 32)


def test_cross_attn_and_layer_encoder_match_polar():
    router = build_router(d_model=32, nheads=4, n_layer_blocks=2)
    # polar/model.py: nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads,
    # batch_first=True, dropout=0.1)
    assert isinstance(router.cross_attn, torch.nn.MultiheadAttention)
    assert router.cross_attn.embed_dim == 32
    assert router.cross_attn.num_heads == 4
    assert router.cross_attn.batch_first is True
    # polar/model.py: TransformerEncoderLayer(dim_feedforward=d_model*4,
    # batch_first=True), TransformerEncoder(enc_layer, num_layers=n_layer_blocks)
    assert isinstance(router.layer_encoder, torch.nn.TransformerEncoder)
    assert len(router.layer_encoder.layers) == 2
    one_layer = router.layer_encoder.layers[0]
    assert one_layer.linear1.out_features == 32 * 4


class _StubEncoder(torch.nn.Module):
    """Frozen stub encoder (matches test_router_train.py's pattern), avoids
    downloading the real 0.6B model just to check freezing behavior."""

    class _Cfg:
        hidden_size = EMBED

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.proj = torch.nn.Linear(EMBED, EMBED)


def test_encoder_is_frozen_like_polar():
    router = PolarRouter(num_layers=12, encoder=_StubEncoder(), d_model=32)
    # polar/model.py: `for p in self.embedding_model.parameters():
    # p.requires_grad = False` -- ours freezes the same way via requires_grad_(False)
    # in _attach_encoder; optimizer only ever sees the unfrozen (head) params.
    trainable = {id(p) for p in _trainable_params(router)}
    for p in router.encoder.parameters():
        assert id(p) not in trainable
        assert p.requires_grad is False


# --------------------------------------------------------------------------- #
# decode: decode_polar_to_actions defaults
# --------------------------------------------------------------------------- #
def test_decode_defaults_match_polar():
    sig = inspect.signature(decode)
    defaults = {name: p.default for name, p in sig.parameters.items()}
    # polar/model.py: threshold: float = 0.5
    assert defaults["threshold"] == 0.5
    # polar/model.py: beam_size: int = 5
    assert defaults["beam_size"] == 5
    # polar/model.py: top_k_ops: int = 2
    assert defaults["top_k_ops"] == 2


def test_max_segment_len_matches_polar_max_pack():
    # polar/model.py: max_pack: int = 4 -- our grammar's MAX_SEGMENT_LEN is the
    # same constant under a different name (re_polar/core/ir.py, "PoLar's validated
    # constraint").
    assert MAX_SEGMENT_LEN == 4


# --------------------------------------------------------------------------- #
# loss: PoLar's train_polar()'s unweighted loss -- bce_none/ce_none
# construction, per-example aggregation (seg_loss_per = bce_none(...).mean(dim=1);
# op_loss_per = masked-mean CE; loss_per = seg_loss_per + op_loss_per; no class
# weight, no focal, no length-preference reweighting by default).
# --------------------------------------------------------------------------- #
def test_compute_loss_defaults_are_polar_unweighted():
    sig = inspect.signature(compute_loss)
    defaults = {name: p.default for name, p in sig.parameters.items()}
    assert defaults["policy_mode"] == "polar"
    assert defaults["anti_original_lambda"] == 0.0
    assert defaults["op_class_weights"] is None
    assert defaults["op_focal_gamma"] == 0.0


def test_compute_loss_formula_matches_polar_unweighted():
    router = build_router(num_layers=6, d_model=16)
    router.eval()  # disable dropout so the two forward passes below agree
    D = 6
    batch = {
        "token_hidden": torch.randn(2, 5, EMBED),
        "key_padding_mask": torch.zeros(2, 5, dtype=torch.bool),
        "seg_flip": torch.zeros(2, D),
        "op_labels": torch.full((2, D), -100, dtype=torch.long),
        "weight": torch.ones(2),  # per-sample-weight-normalize, 1.0 by default (inert)
    }
    batch["seg_flip"][:, 3] = 1.0
    batch["op_labels"][:, 0] = 1  # keep
    batch["op_labels"][:, 3] = 2  # repeat

    total, seg_loss, op_loss = compute_loss(router, batch)

    seg_logits, op_logits = router(
        token_hidden_states=batch["token_hidden"], key_padding_mask=batch["key_padding_mask"]
    )
    # polar/train.py: bce_none(seg_logits[:,1:], seg_t[:,1:]).mean(dim=1)
    expected_seg = torch.nn.functional.binary_cross_entropy_with_logits(
        seg_logits[:, 1:], batch["seg_flip"][:, 1:], reduction="none"
    ).mean(dim=1)
    # polar/train.py: ce_none(...).view(B,D) masked-summed / count
    ce = torch.nn.functional.cross_entropy(
        op_logits.reshape(-1, 3), batch["op_labels"].reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(2, D)
    mask = (batch["op_labels"] != -100).float()
    expected_op = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # compute_loss returns batch-mean scalars, not per-example -- compare means.
    assert torch.allclose(seg_loss, expected_seg.mean(), atol=1e-5)
    assert torch.allclose(op_loss, expected_op.mean(), atol=1e-5)
    # polar/train.py: loss_per = seg_loss_per + op_loss_per (unweighted mean)
    assert torch.allclose(total, (expected_seg + expected_op).mean(), atol=1e-5)


def test_repeat_2x_default_now_matches_polar():
    # PoLar's own label parser (polar/data.py) only ever matches a REPEAT
    # block against exactly two consecutive repetitions -- any path with a
    # segment repeated != 2x is dropped whole. Our default now matches that.
    sig = inspect.signature(program_from_layer_path)
    defaults = {name: p.default for name, p in sig.parameters.items()}
    assert defaults["strict_repeat_2x"] is True
