"""PoLar router, predicts a whole layer-execution program from a question.

Reimplemented from the paper's formalization ("Skip a Layer or Loop It?
Learning Program-of-Layers in LLMs", arXiv:2606.06574), no PoLar code is
copied or imported (see NOTICE.md).

Architecture (paper, Appendix "Router"):

    question string
      -> frozen Qwen3-Embedding-0.6B  -> token-level last hidden states
      -> q_proj (embed_dim -> d_model) -> cross-attention keys/values
    D learnable layer queries (one per target-model layer)
      -> cross-attention (queries attend to the question's token states)
      -> small transformer over the layer dimension
      -> seg head  l^seg in R^D          (per-layer boundary logits, BCE)
      -> op  head  l^op  in R^{D x n_ops} (keep/skip/repeat, masked CE at
                                           segment starts)

Inference: `decode(seg_logits, op_logits)` -> threshold -> grammar-constrained
beam search -> a deterministic `re_polar.core.Program` that passes
`validate_program` (contiguous segments, each <= MAX_SEGMENT_LEN).

Design notes
------------
- D is the target model's layer count. Callers get it from
  ``MODEL_REGISTRY[model]["num_layers"]`` (see :meth:`PolarRouter.for_model`);
  never hard-coded.
- The frozen encoder is *injectable / lazy*: pass a preloaded ``encoder`` to
  reuse one, or pass ``embed_dim`` to skip the download entirely and drive the
  router with precomputed token embeddings (the unit-test path). The default
  loads the real ``Qwen/Qwen3-Embedding-0.6B``.
- ``n_ops`` defaults to 3 (skip/keep/repeat); it extends to 4 (parloop) with
  no schema change: build with ``n_ops=4`` and pass a 4-op vocabulary to
  ``decode`` once the IR gains the parloop op.
- Op-index convention matches PoLar's ``polar/config.py``
  (``OP_SKIP=0, OP_EXECUTE=1, OP_REPEAT=2``) so the supervision schema stays
  cross-validated against PoLar's own bridge format. See :data:`DEFAULT_OPS`.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, is_valid

__all__ = ["PolarRouter", "decode", "decode_topk", "DEFAULT_OPS", "DEFAULT_EMBEDDING_MODEL"]

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Op vocabulary ordering, index == op-head column. Matches PoLar's
# polar/config.py (OP_SKIP=0, OP_EXECUTE/keep=1, OP_REPEAT=2). A later
# extension appends Op.PARLOOP at index 3 (n_ops=4) without disturbing
# these three.
DEFAULT_OPS: Tuple[Op, ...] = (Op.SKIP, Op.KEEP, Op.REPEAT)

_REPEAT_TIMES = 2  # a REPEAT segment executes twice (params["times"]); paper's recurrence unit


class PolarRouter(nn.Module):
    """Question -> (seg logits R^D, op logits R^{D x n_ops})."""

    def __init__(
        self,
        num_layers: int,
        n_ops: int = 3,
        *,
        encoder: Optional[nn.Module] = None,
        embed_dim: Optional[int] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        d_model: int = 256,
        nheads: int = 4,
        n_layer_blocks: int = 2,
        dropout: float = 0.1,
        max_question_tokens: int = 512,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if n_ops < len(DEFAULT_OPS):
            raise ValueError(
                f"n_ops must be >= {len(DEFAULT_OPS)} (skip/keep/repeat), got {n_ops}"
            )
        self.num_layers = num_layers
        self.n_ops = n_ops
        self.d_model = d_model
        self.embedding_model_name = embedding_model_name
        self.max_question_tokens = max_question_tokens

        # --- frozen encoder (injectable / lazy) ---
        # embed_dim resolution order: explicit arg > provided encoder > load default.
        self._tokenizer = None
        if embed_dim is not None:
            self.embed_dim = embed_dim
            self.encoder = None  # driven by precomputed embeddings; may still be lazy-loaded later
            if encoder is not None:
                self._attach_encoder(encoder)
        elif encoder is not None:
            self._attach_encoder(encoder)
            self.embed_dim = encoder.config.hidden_size
        else:
            enc = self._load_encoder(embedding_model_name)
            self._attach_encoder(enc)
            self.embed_dim = enc.config.hidden_size

        # --- trainable head ---
        self.q_proj = nn.Linear(self.embed_dim, d_model)
        # D learnable layer queries, one per target-model layer.
        self.layer_queries = nn.Embedding(num_layers, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads, batch_first=True, dropout=dropout
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer_blocks)
        self.seg_head = nn.Linear(d_model, 1)
        self.op_head = nn.Linear(d_model, n_ops)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def for_model(cls, model: str, **kwargs) -> "PolarRouter":
        """Build a router sized for a registered target model.

        D = ``MODEL_REGISTRY[model]["num_layers"]`` (never hard-coded). Extra
        kwargs (e.g. ``embed_dim`` to skip the encoder download) pass through.
        """
        from re_polar.models import MODEL_REGISTRY

        if model not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {model!r}; known: {sorted(MODEL_REGISTRY)}")
        return cls(num_layers=MODEL_REGISTRY[model]["num_layers"], **kwargs)

    @staticmethod
    def _load_encoder(name: str) -> nn.Module:
        from transformers import AutoModel  # lazy: keeps the module importable without transformers

        return AutoModel.from_pretrained(name)

    def _attach_encoder(self, encoder: nn.Module) -> None:
        """Register `encoder` frozen + in eval mode."""
        encoder.requires_grad_(False)
        encoder.eval()
        self.encoder = encoder

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # lazy

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name, padding_side="left"
            )
        return self._tokenizer

    def train(self, mode: bool = True) -> "PolarRouter":
        """Keep the frozen encoder in eval mode even when the router trains."""
        super().train(mode)
        if getattr(self, "encoder", None) is not None:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def last_token_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Qwen3-Embedding model-card pooling, the last non-pad token per row.

        The cross-attention path uses the *full* token sequence, so this is only
        for callers that need a single pooled query vector.
        """
        left_padding = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
        return last_hidden_states[batch, seq_lengths]

    def encode_questions(
        self, questions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Questions -> (token hidden states (B,T,embed_dim), key_padding_mask (B,T)).

        Lazily loads the default encoder if none was supplied at construction.
        ``key_padding_mask`` is True at padding positions (nn.MultiheadAttention
        convention).
        """
        if self.encoder is None:
            self._attach_encoder(self._load_encoder(self.embedding_model_name))
        tokenizer = self._load_tokenizer()
        device = next(self.encoder.parameters()).device
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_question_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.encoder(**inputs)
        token_hidden = out.last_hidden_state  # (B,T,embed_dim)
        key_padding_mask = inputs["attention_mask"] == 0
        return token_hidden, key_padding_mask

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        questions: Optional[List[str]] = None,
        *,
        token_hidden_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (seg_logits (B,D), op_logits (B,D,n_ops)).

        Provide either ``questions`` (encoded by the frozen encoder) or
        precomputed ``token_hidden_states`` (B,T,embed_dim), the latter is the
        download-free unit-test path.
        """
        if token_hidden_states is None:
            if questions is None:
                raise ValueError("Provide `questions` or `token_hidden_states`.")
            token_hidden_states, key_padding_mask = self.encode_questions(questions)
        if token_hidden_states.size(-1) != self.embed_dim:
            raise ValueError(
                f"token_hidden_states last dim {token_hidden_states.size(-1)} "
                f"!= embed_dim {self.embed_dim}"
            )

        # The frozen encoder may run in bf16 (checkpoint dtype); the trainable
        # head is fp32. Cast to the head's dtype at the projection boundary.
        token_hidden_states = token_hidden_states.to(self.q_proj.weight.dtype)
        feats = self.q_proj(token_hidden_states)  # (B,T,d_model)
        batch_size = feats.size(0)
        layer_ids = torch.arange(
            self.num_layers, device=self.layer_queries.weight.device
        ).unsqueeze(0).expand(batch_size, self.num_layers)
        layer_q = self.layer_queries(layer_ids)  # (B,D,d_model)

        # Cross-attend: Q = layer queries, K/V = question token states.
        attended, _ = self.cross_attn(
            query=layer_q, key=feats, value=feats, key_padding_mask=key_padding_mask
        )
        # Small transformer over the layer dimension for global boundary/op decisions.
        x = self.layer_encoder(attended)  # (B,D,d_model)

        seg_logits = self.seg_head(x).squeeze(-1)  # (B,D)
        op_logits = self.op_head(x)  # (B,D,n_ops)
        return seg_logits, op_logits

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def decode(
        self,
        seg_logits: torch.Tensor,
        op_logits: torch.Tensor,
        **kwargs,
    ) -> Program:
        """Instance shortcut for the module-level :func:`decode`."""
        return decode(seg_logits, op_logits, num_layers=self.num_layers, **kwargs)

    def decode_topk(
        self,
        seg_logits: torch.Tensor,
        op_logits: torch.Tensor,
        *,
        k: int = 5,
        **kwargs,
    ) -> List[Program]:
        """Instance shortcut for the module-level :func:`decode_topk`."""
        return decode_topk(seg_logits, op_logits, k=k, num_layers=self.num_layers, **kwargs)


# ---------------------------------------------------------------------- #
# decode: threshold -> grammar-constrained beam search -> Program
# ---------------------------------------------------------------------- #
def _segments_from_starts(
    starts: Sequence[int], num_layers: int, max_len: int
) -> List[Tuple[int, int]]:
    """Boundary starts -> contiguous [0, D) cover, every segment length in [1, max_len]."""
    bounded = sorted({0} | {s for s in starts if 0 < s < num_layers})
    segments: List[Tuple[int, int]] = []
    for idx, start in enumerate(bounded):
        end = bounded[idx + 1] if idx + 1 < len(bounded) else num_layers
        cur = start
        while cur < end:  # split any span longer than max_len (enforces <= MAX_SEGMENT_LEN)
            nxt = min(cur + max_len, end)
            segments.append((cur, nxt))
            cur = nxt
    return segments


def _build_program(
    segments: Sequence[Tuple[int, int]],
    op_choice: Sequence[int],
    ops: Sequence[Op],
    num_layers: int,
) -> Program:
    segs = []
    for (start, end), op_idx in zip(segments, op_choice):
        op = ops[op_idx]
        params = {"times": _REPEAT_TIMES} if op is Op.REPEAT else {}
        segs.append(Segment(start=start, end=end, op=op, params=params))
    return Program(num_layers=num_layers, segments=segs)


def decode(
    seg_logits: torch.Tensor,
    op_logits: torch.Tensor,
    *,
    num_layers: Optional[int] = None,
    ops: Sequence[Op] = DEFAULT_OPS,
    threshold: float = 0.5,
    beam_size: int = 5,
    top_k_ops: int = 2,
) -> Program:
    """Decode one example's logits into a deterministic, valid ``Program``.

    ``seg_logits`` R^D (per-layer boundary logits), ``op_logits`` R^{D x n_ops}.
    A leading batch dim of size 1 is accepted and squeezed. ``ops`` is the op
    vocabulary in op-head column order (default skip/keep/repeat); pass a 4-op
    vocabulary once the IR gains parloop, no other change needed.

    Steps: sigmoid(seg) >= threshold picks segment starts; spans longer than
    MAX_SEGMENT_LEN are split so every segment is valid; a beam search over the
    per-segment op log-probs (evaluated at each segment's start layer) returns
    the highest-scoring op assignment whose program passes ``validate_program``
    (i.e. not everything skipped). Deterministic: no sampling, stable tie-breaks.
    """
    seg = seg_logits.detach()
    op = op_logits.detach()
    if seg.dim() == 2 and seg.size(0) == 1:  # (1,D) -> (D,)
        seg = seg.squeeze(0)
    if op.dim() == 3 and op.size(0) == 1:  # (1,D,n_ops) -> (D,n_ops)
        op = op.squeeze(0)
    if seg.dim() != 1 or op.dim() != 2:
        raise ValueError(
            f"decode expects a single example: seg R^D and op R^(D x n_ops); "
            f"got seg {tuple(seg_logits.shape)}, op {tuple(op_logits.shape)}"
        )

    D = seg.size(0)
    if num_layers is not None and num_layers != D:
        raise ValueError(f"seg_logits length {D} != num_layers {num_layers}")
    n_vocab = len(ops)
    if op.size(0) != D:
        raise ValueError(f"op_logits has {op.size(0)} layers, seg has {D}")
    if op.size(1) < n_vocab:
        raise ValueError(f"op_logits has {op.size(1)} op columns, need >= {n_vocab}")

    seg_probs = torch.sigmoid(seg.float()).tolist()
    # Only the first n_vocab op columns are decodable to IR ops.
    op_logp = torch.log_softmax(op[:, :n_vocab].float(), dim=-1).tolist()

    starts = [i for i in range(1, D) if seg_probs[i] >= threshold]
    segments = _segments_from_starts(starts, D, MAX_SEGMENT_LEN)

    top_k = max(1, min(top_k_ops, n_vocab))
    # beam entries: (op_choice tuple, cumulative logprob)
    beams: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]
    for (start, _end) in segments:
        lp = op_logp[start]
        ranked = sorted(range(n_vocab), key=lambda o: (-lp[o], o))[:top_k]
        expanded: List[Tuple[Tuple[int, ...], float]] = []
        for choice, score in beams:
            for o in ranked:
                expanded.append((choice + (o,), score + lp[o]))
        # deterministic: higher score first, then lexicographic op-choice tie-break
        expanded.sort(key=lambda t: (-t[1], t[0]))
        beams = expanded[:beam_size]

    for choice, _score in beams:
        program = _build_program(segments, choice, ops, D)
        if is_valid(program):
            return program

    # Fallback: every beam was all-skip. Force the single cheapest-to-flip
    # segment to its best non-skip op so the program becomes valid.
    keep_idx = ops.index(Op.KEEP)
    non_skip = [i for i, o in enumerate(ops) if o is not Op.SKIP]
    best_choice = list(beams[0][0]) if beams else [ops.index(Op.SKIP)] * len(segments)
    if segments:
        best_seg, best_op, best_gain = 0, keep_idx, float("-inf")
        for s_idx, (start, _end) in enumerate(segments):
            lp = op_logp[start]
            skip_lp = lp[ops.index(Op.SKIP)]
            for o in non_skip:
                gain = lp[o] - skip_lp
                if gain > best_gain:
                    best_gain, best_seg, best_op = gain, s_idx, o
        best_choice[best_seg] = best_op
    program = _build_program(segments, best_choice, ops, D)
    return program


def _decode_normalize(seg_logits: torch.Tensor, op_logits: torch.Tensor):
    """Shared shape-normalisation for decode / decode_topk (single example)."""
    seg = seg_logits.detach()
    op = op_logits.detach()
    if seg.dim() == 2 and seg.size(0) == 1:  # (1,D) -> (D,)
        seg = seg.squeeze(0)
    if op.dim() == 3 and op.size(0) == 1:  # (1,D,n_ops) -> (D,n_ops)
        op = op.squeeze(0)
    if seg.dim() != 1 or op.dim() != 2:
        raise ValueError(
            f"decode expects a single example: seg R^D and op R^(D x n_ops); "
            f"got seg {tuple(seg_logits.shape)}, op {tuple(op_logits.shape)}"
        )
    return seg, op


def decode_topk(
    seg_logits: torch.Tensor,
    op_logits: torch.Tensor,
    *,
    k: int = 5,
    num_layers: Optional[int] = None,
    ops: Sequence[Op] = DEFAULT_OPS,
    threshold: float = 0.5,
    top_k_ops: int = 2,
) -> List[Program]:
    """Top-``k`` DISTINCT valid programs from the grammar-constrained beam.

    Same segmentation + per-segment op beam search as :func:`decode`, but returns
    up to ``k`` distinct valid programs (deduplicated by executed layer-path,
    highest-scoring first) instead of only the single best, this is the paper's
    top-k decode used for pass@k evaluation. Falls back to ``[decode(...)]`` when
    the beam yields no valid program (never returns an empty list). Deterministic.

    ``decode(...)`` is unchanged and remains the pass@1 top-1 program (== the
    first element here whenever the beam is non-empty).
    """
    seg, op = _decode_normalize(seg_logits, op_logits)
    D = seg.size(0)
    if num_layers is not None and num_layers != D:
        raise ValueError(f"seg_logits length {D} != num_layers {num_layers}")
    n_vocab = len(ops)
    if op.size(0) != D:
        raise ValueError(f"op_logits has {op.size(0)} layers, seg has {D}")
    if op.size(1) < n_vocab:
        raise ValueError(f"op_logits has {op.size(1)} op columns, need >= {n_vocab}")
    k = max(1, int(k))

    seg_probs = torch.sigmoid(seg.float()).tolist()
    op_logp = torch.log_softmax(op[:, :n_vocab].float(), dim=-1).tolist()
    starts = [i for i in range(1, D) if seg_probs[i] >= threshold]
    segments = _segments_from_starts(starts, D, MAX_SEGMENT_LEN)

    # Widen the beam so >= k distinct valid programs can survive pruning.
    beam_width = max(4 * k, 16)
    top_kk = max(1, min(top_k_ops, n_vocab))
    beams: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]
    for (start, _end) in segments:
        lp = op_logp[start]
        ranked = sorted(range(n_vocab), key=lambda o: (-lp[o], o))[:top_kk]
        expanded: List[Tuple[Tuple[int, ...], float]] = []
        for choice, score in beams:
            for o in ranked:
                expanded.append((choice + (o,), score + lp[o]))
        expanded.sort(key=lambda t: (-t[1], t[0]))
        beams = expanded[:beam_width]

    out: List[Program] = []
    seen: set = set()
    for choice, _score in beams:
        program = _build_program(segments, choice, ops, D)
        if not is_valid(program):
            continue
        key = tuple(program.to_layer_path())
        if key in seen:
            continue
        seen.add(key)
        out.append(program)
        if len(out) >= k:
            break

    if not out:  # beam produced nothing valid -> decode()'s guaranteed-valid fallback
        out = [decode(seg_logits, op_logits, num_layers=num_layers, ops=ops,
                       threshold=threshold, top_k_ops=top_k_ops)]
    return out
