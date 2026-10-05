"""The program-execution stack: define a program, run it on a model, grade it.

ir.py         The Program/Segment IR (Op, Segment, Program).
grammar.py    Program validity rules (validate_program, is_valid).
layer_engine.py, model_loader.py
              LayerEngine: the generic transformer layer-manipulation engine
              (skip/repeat/keep dispatch, model loading) programs run on top
              of. Not itself part of the method's contribution, see its own
              docstring for provenance.
executor.py   ProgramExecutor: ties a Program to a LayerEngine and runs it.
grader.py     Torch-free DART-Math/ASDiv/MAWPS answer-equivalence grading
              (generation-based).
mmlu_pro_scoring.py, mmlu_pro_domain_eval.py
              MMLU-Pro's grading path instead: forced-choice log-likelihood
              scoring over answer-letter logits, no generation step.

Import the frequently-used names directly from `re_polar.core`; grader.py and
the mmlu_pro_* modules are imported from their own submodule, same as
before the merge.
"""

import importlib
from typing import TYPE_CHECKING

from .ir import Op, Segment, Program, MAX_SEGMENT_LEN
from .grammar import validate_program, is_valid

if TYPE_CHECKING:  # static analysis only; never imported at runtime
    from .executor import ProgramExecutor
    from .layer_engine import LayerEngine
    from .model_loader import detect_device, load_model_and_tokenizer

# torch/transformers-backed names load on first access (PEP 562), so importing
# any submodule -- in particular the torch-free grader.py inside spawned,
# RLIMIT_AS-capped grading workers -- doesn't drag torch in through this file.
# Eager imports here put every grading worker at ~7.2 GiB virtual size on Linux
# (vs ~2.6 GiB for the evaluator alone), leaving <1 GiB of the 8 GiB cap.
_LAZY = {
    "load_model_and_tokenizer": "model_loader",
    "detect_device": "model_loader",
    "LayerEngine": "layer_engine",
    "ProgramExecutor": "executor",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Op",
    "Segment",
    "Program",
    "MAX_SEGMENT_LEN",
    "validate_program",
    "is_valid",
    "load_model_and_tokenizer",
    "detect_device",
    "LayerEngine",
    "ProgramExecutor",
]
