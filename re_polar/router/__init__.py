"""Program router, reimplemented from the paper's formalization.

Frozen Qwen3-Embedding-0.6B -> projection -> D learnable layer queries ->
cross-attention -> small transformer over the layer dim -> segment-boundary
head (BCE) + op head (masked CE at segment starts). Predicts the whole program
once per input before the forward pass; inference = threshold -> grammar-
constrained beam search -> deterministic program. D always comes from
MODEL_REGISTRY[model]["num_layers"], never hard-coded.

Planned modules: model.py, train.py, infer.py.

Public names are exposed LAZILY (PEP 562): importing this package must not pull
in ``model.py`` (which imports torch). That keeps ``import re_polar.router.infer``
torch-free, GenerationReward's ``spawn`` grading workers reconstruct ``__main__``
by re-importing that module (``python -m re_polar.router.infer``), and re-running this
package ``__init__`` eagerly would load torch into every worker. ``from
re_polar.router import PolarRouter / decode / DEFAULT_OPS / DEFAULT_EMBEDDING_MODEL``
still works (resolved on first access); ``from re_polar.router.model import ...``
(used by train.py) is unaffected.
"""

__all__ = ["PolarRouter", "decode", "DEFAULT_OPS", "DEFAULT_EMBEDDING_MODEL"]


def __getattr__(name):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
