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

from .ir import Op, Segment, Program, MAX_SEGMENT_LEN
from .grammar import validate_program, is_valid
from .model_loader import load_model_and_tokenizer, detect_device
from .layer_engine import LayerEngine
from .executor import ProgramExecutor

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
