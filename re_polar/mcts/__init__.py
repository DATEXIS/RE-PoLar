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

from .rewards import GenerationReward, LogLikReward
from .scheduler import EvalCache, MCTSRunner, derive_tree_seed
from .search import ProgramMCTS

__all__ = ["ProgramMCTS", "MCTSRunner", "EvalCache", "GenerationReward", "LogLikReward",
           "derive_tree_seed"]
