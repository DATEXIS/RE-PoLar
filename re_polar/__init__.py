"""re_polar -- Program-of-Layers.

The method itself: define a program and run it on a model (core/), MCTS
search (mcts/), post-hoc analysis over MCTS output (mcts/analysis/), the
router (router/), dataset builders (datasets/), and vendored third-party
grading code (vendor/).

core/ and models/ are also nested here, but are a separate concern: a
generic transformer layer-manipulation engine plus the per-model config
registry that mcts/router are built on top of, not the paper's own
contribution. core/layer_engine.py in particular is adapted from prior
published work -- see its own docstring for the citation. The LLM-judge
client used for error analysis lives outside this package, under
analysis/error_analysis/llm/, since it's a paper-specific tool, not
part of the reusable method.
"""
