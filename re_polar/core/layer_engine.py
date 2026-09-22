"""
Layer manipulation engine for transformer models.

Supports virtual layer duplication and skipping via shallow copies:
no extra VRAM because weight tensors are shared across all copies.
Each copy carries a unique layer_idx so DynamicCache doesn't collide KV slots.

Adapted from the layer-duplication engine described in David Noel Ng's blog
post "LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a
Single Weight" (2026), https://dnhkng.github.io/posts/rys/, the same source
cited in this paper's motivation section.
"""

import copy
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_loader import load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def resolve_layer_parent(model: torch.nn.Module):
    """Locate the (module, attr_name) holding a model's transformer layer list.

    Handles standard decoder-only models (model.model.layers), GPT-2-style
    models (model.transformer.h), and multimodal wrappers like Gemma3's
    Gemma3ForConditionalGeneration (model.model.language_model.layers) where
    model.model is itself a wrapper with no .layers attribute.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, "h"
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm, "layers"
    raise ValueError("Cannot locate layer list in model")


def get_layers(model: torch.nn.Module):
    """Return the transformer layer list (nn.ModuleList) for any registered model."""
    layer_parent, layer_attr = resolve_layer_parent(model)
    return getattr(layer_parent, layer_attr)


def _rebind_accelerate_hook(original: torch.nn.Module, copied: torch.nn.Module) -> None:
    source_hook = getattr(original, "_hf_hook", None)
    if source_hook is None:
        return
    try:
        from accelerate.hooks import add_hook_to_module
    except Exception:
        return
    for attr in ("_hf_hook", "_old_forward"):
        if hasattr(copied, attr):
            delattr(copied, attr)
    copied.forward = type(copied).forward.__get__(copied, type(copied))
    add_hook_to_module(copied, copy.copy(source_hook), append=False)


def _shallow_copy_layer(layer: torch.nn.Module, new_layer_idx: int) -> torch.nn.Module:
    """
    Shallow-copy a decoder layer and assign a unique layer_idx to its attention.
    Weights are shared (zero extra VRAM); DynamicCache uses layer_idx for KV slots.
    """
    new_layer = copy.copy(layer)
    new_layer._modules = dict(layer._modules)

    for attn_attr in ("self_attn", "linear_attn"):
        if hasattr(layer, attn_attr):
            orig_attn = getattr(layer, attn_attr)
            new_attn = copy.copy(orig_attn)
            new_attn._modules = dict(orig_attn._modules)
            new_attn.layer_idx = new_layer_idx
            _rebind_accelerate_hook(orig_attn, new_attn)
            new_layer._modules[attn_attr] = new_attn

    _rebind_accelerate_hook(layer, new_layer)
    return new_layer


# ---------------------------------------------------------------------------
# LayerEngine
# ---------------------------------------------------------------------------

class LayerEngine:
    """
    Loads a transformer model and provides layer-level rerouting.

    Config space (N = number of layers):
    - Baseline (i==j): original sequential execution
    - Duplication (i < j, upper triangle): layers i..j-1 execute twice
      path = [0..j-1] + [i..N-1]
    - Skip (i > j, lower triangle): layers j..i-1 are omitted
      path = [0..j-1] + [i..N-1]

    For all configs weights are shared (shallow copies), no extra VRAM.
    """

    def __init__(self, model_id: str, trust_remote_code: bool = True):
        self.model_id = model_id
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_id, trust_remote_code=trust_remote_code
        )
        self._init_layer_access()
        self._is_rerouted = False
        self._patched_inner: Optional[torch.nn.Module] = None
        print(f"Model has {self.num_layers} layers")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_layer_access(self):
        model = self.model
        self._layer_parent, self._layer_attr = resolve_layer_parent(model)
        self._original_layers = getattr(self._layer_parent, self._layer_attr)
        self._original_num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
        self.num_layers = len(self._original_layers)

        cfg_layer_types = getattr(model.config, "layer_types", None)
        if isinstance(cfg_layer_types, (list, tuple)):
            lt = list(cfg_layer_types)
            self._original_layer_types = lt if len(lt) >= self.num_layers else None
            self._layer_types_was_tuple = isinstance(cfg_layer_types, tuple)
        else:
            self._original_layer_types = None
            self._layer_types_was_tuple = False

    # ------------------------------------------------------------------
    # Config generators
    # ------------------------------------------------------------------

    @staticmethod
    def generate_dup_configs(num_layers: int):
        """Yield (i, j) duplication configs including baseline (0, 0)."""
        yield (0, 0)
        for j in range(1, num_layers + 1):
            for i in range(j):
                yield (i, j)

    @staticmethod
    def generate_skip_configs(num_layers: int):
        """Yield (i, j) skip configs (lower triangle, i > j), excluding all-skip."""
        for j in range(num_layers):
            for i in range(j + 1, num_layers + 1):
                if i == num_layers and j == 0:
                    continue
                yield (i, j)

    @staticmethod
    def num_dup_configs(num_layers: int) -> int:
        return num_layers * (num_layers + 1) // 2 + 1

    @staticmethod
    def num_skip_configs(num_layers: int) -> int:
        return max(num_layers * (num_layers + 1) // 2 - 1, 0)

    def get_dup_path(self, i: int, j: int) -> List[int]:
        """Layer execution path for duplication config (i < j): [0..j-1] + [i..N-1]."""
        if not (0 <= i <= j <= self.num_layers):
            raise ValueError(f"Invalid dup config ({i}, {j}) for N={self.num_layers}")
        if i == j:
            return list(range(self.num_layers))
        return list(range(j)) + list(range(i, self.num_layers))

    def get_skip_path(self, i: int, j: int) -> List[int]:
        """Layer execution path for skip config (i > j): omit layers j..i-1."""
        if not (0 <= j < i <= self.num_layers):
            raise ValueError(f"Invalid skip config ({i}, {j}) for N={self.num_layers}")
        path = list(range(j)) + list(range(i, self.num_layers))
        if not path:
            raise ValueError("Skip config removes all layers.")
        return path

    # ------------------------------------------------------------------
    # Rerouting
    # ------------------------------------------------------------------

    def apply_layer_rerouting(self, path: List[int]):
        """
        Replace model.layers with shallow-copied layers indexed by path.
        Each position in the path gets its own attention copy with unique layer_idx.
        """
        if not path:
            raise ValueError("Path must not be empty.")
        rerouted = torch.nn.ModuleList(
            [_shallow_copy_layer(self._original_layers[idx], pos) for pos, idx in enumerate(path)]
        )
        self._set_layers(rerouted, layer_path=path)

        # Qwen3.5 hybrid-cache patch: if the path has no linear-attn layers, the
        # hybrid cache's _update_linear_attn_mask raises StopIteration. Neutralise it.
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "_update_linear_attn_mask"):
            has_linear = any(hasattr(self._original_layers[idx], "linear_attn") for idx in path)
            if not has_linear:
                self._patched_inner = inner
                inner._update_linear_attn_mask = lambda mask, pkv: None

        self._is_rerouted = True
        return self

    def restore_original(self):
        """Revert to the original layer stack."""
        if not self._is_rerouted:
            return
        self._set_layers(self._original_layers)
        if self._original_num_hidden_layers is not None:
            self.model.config.num_hidden_layers = self._original_num_hidden_layers
        if self._patched_inner is not None:
            try:
                del self._patched_inner._update_linear_attn_mask
            except AttributeError:
                pass
            self._patched_inner = None
        self._is_rerouted = False

    def _set_layers(self, new_layers, layer_path: Optional[List[int]] = None):
        setattr(self._layer_parent, self._layer_attr, new_layers)
        if hasattr(self.model.config, "num_hidden_layers"):
            self.model.config.num_hidden_layers = len(new_layers)

        if self._original_layer_types is not None:
            if layer_path is None:
                lt = list(self._original_layer_types)
            else:
                lt = [self._original_layer_types[idx] for idx in layer_path]
            if self._layer_types_was_tuple:
                self.model.config.layer_types = tuple(lt)
            else:
                self.model.config.layer_types = lt
