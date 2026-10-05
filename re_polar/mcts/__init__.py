"""MCTS program discovery, reimplemented from the paper (PoLar publishes no MCTS code).

search.py    ProgramMCTS: one UCB tree per input; state = stack position +
             segment prefix; actions = (len 1-4, keep/skip/repeat); length
             penalty -lam*|pi|/D; GPU-free (propose/update protocol).
scheduler.py MCTSRunner: all trees in lockstep rounds, proposals grouped by
             identical program -> one batched reward call each (shared-seed
             trees diverge only where rewards diverge); EvalCache (JSONL,
             on disk) makes reruns resume free. Identity precomputed first.
rewards.py   GenerationReward: batched greedy generation through
             ProgramExecutor + dart-math answer equivalence (upstream pip,
             see requirements-mcts.txt). LogLikReward: one log-likelihood
             forward pass via re_polar.core.mmlu_pro_domain_eval argmax-option
             scoring, ~100x cheaper -> makes 27-32B MCTS feasible.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # static analysis only; never imported at runtime
    from .rewards import GenerationReward, LogLikReward
    from .scheduler import EvalCache, MCTSRunner, derive_tree_seed
    from .search import ProgramMCTS

# Lazy (PEP 562): `python -m re_polar.mcts.run_search_*` makes every spawned
# grading worker re-import its main module, and with it this package. Eager
# imports here pulled torch (via rewards.py) into each RLIMIT_AS-capped worker;
# see re_polar/core/__init__.py.
_LAZY = {
    "GenerationReward": "rewards",
    "LogLikReward": "rewards",
    "EvalCache": "scheduler",
    "MCTSRunner": "scheduler",
    "derive_tree_seed": "scheduler",
    "ProgramMCTS": "search",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ProgramMCTS",
    "MCTSRunner",
    "EvalCache",
    "GenerationReward",
    "LogLikReward",
    "derive_tree_seed",
]
