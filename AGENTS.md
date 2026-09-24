# AGENTS.md

Conventions and commands for working in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs as `python -m <package>.<module>` (`re_polar.`, `analysis.`,
or `prestudy.`) from the repository root, there is no package install
step, so always use `-m`, never `python path/to/file.py` directly (the
latter doesn't put the repo root on `sys.path`).

## Formatting

Code is formatted with `black` (line-length=100, config in `pyproject.toml`).
`re_polar/vendor/dart_math/` is excluded (third-party, kept diffable
against upstream).

## Layout

Everything lives under the single `re_polar/` package.

- `re_polar/core/`: define a program, run it on a model, grade it. The
  Program/Segment IR, the layer-execution engine programs run on top of
  (model loading, layer skip/repeat/keep dispatch; not itself part of the
  method's contribution, see its own docstring for provenance), the
  executor, and both grading paths (generation-based for DART-Math/ASDiv/
  MAWPS, log-likelihood MCQ scoring for MMLU-Pro).
- `re_polar/mcts/`: the search that discovers programs.
- `re_polar/mcts/analysis/`: metrics computed over MCTS output (program
  structure, not router-related).
- `re_polar/router/`: embedding → program predictor, trained on MCTS output.
- `re_polar/datasets/`: dataset builders (`merged_mcts_samples.json`
  read/write, DART-Math/MMLU-Pro/ASDiv/MAWPS splits).
- `re_polar/models/`: one config per evaluated model (`MODEL_REGISTRY`).
- `re_polar/vendor/dart_math/`: third-party answer grading, see `NOTICE.md`.
- `data/` (top-level, separate from `re_polar/datasets/`): the full data
  release (MCTS-discovered programs, generated text, menu cross-execution,
  router checkpoints), plus `DATA.md` describing its schema.
- `analysis/`: regenerates every paper table/figure number from code.
- `prestudy/`: regenerates the paper's motivation-section figures.
- `tests/`: unit tests, runnable on CPU.

## Commands

```bash
pytest tests/
```

## Conventions

- Router layer-count `D` always comes from `MODEL_REGISTRY[model]
  ["num_layers"]`, never hard-coded.
- Update `NOTICE.md` whenever a new external data source or dependency is
  added.
- A `Program`'s `to_layer_path()` must round-trip: decoding a layer path
  back into a `Program` and re-expanding it must reproduce the original
  path exactly (see `re_polar/mcts/analysis/segments.py::decode` for where this is
  asserted).

## Standing invariants

- The identity program (all layers kept, in order) reproduces the
  recorded baseline accuracy for a given model/benchmark.
- Every program in `merged_mcts_samples.json` is JSON-serializable and
  re-executes to the same reward it was recorded with.
