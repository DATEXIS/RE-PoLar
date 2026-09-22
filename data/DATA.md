# Data

Everything this repo's MCTS search and router produced: for each DART-Math
question, the layer-execution programs the search tried (valid and
invalid), the model's actual generated text behind a sample of those
evaluations, a real cross-execution of the top menu of programs, and the
trained router checkpoints.

## Getting DART-Math

Question text and ground-truth answers aren't stored in this repo's data
files; reconstruct them from the public DART-Math dataset:

```bash
python -m re_polar.datasets.dart_math --out-dir ./data/dart_math --include-gsm8k
```

This writes `<out-dir>/diff{1..5}/{train,val,test}.json` (1250/250/500
questions each) plus a `manifest.json`. Each record:

| field | type | meaning |
|---|---|---|
| `query_id` | str | stable id, also the join key into `mcts_results/` |
| `question` | str | the input question text |
| `gt_ans` | str | ground-truth answer |
| `domain` | str | math subject area |
| `math_level` | int or null | MATH's own difficulty level (1-5); `null` for GSM8K questions, which don't have one |
| `source` | str | `"math"` or `"gsm8k"` |
| `fail_rate` | float | DART-Math's own per-query fail rate, used to bin questions into difficulty 1 (easiest) through 5 (hardest) |
| `tot_n_samples` | int | how many responses DART-Math sampled to compute `fail_rate` |
| `difficulty` | int | 1-5, the difficulty tier this record was binned into |

`data/dart_math_splits.json` records the same train/val/test membership,
just as `query_id` lists per difficulty tier
(`{"diff1": {"train": [...], "val": [...], "test": [...]}, ...}`), without
question text — handy for checking which split a `query_id` belongs to
without loading the full reconstructed data.

## MCTS results (`data/mcts_results/`)

For every DART-Math question, every distinct layer-execution program the
search tried, and whether it answered correctly. This is what the router
is trained on.

```
mcts_results/<model>/dart-math-diff-<1..5>/merged_mcts_samples.json.gz
```

One file per (model, difficulty) combination, 2000 questions each. Models:
Qwen3-8B, Qwen3-32B, Qwen2.5-3B, Qwen2.5-7B, Qwen1.5-MoE-A2.7B (LLaMA-3.2-3B
is evaluated only as a fixed baseline, not run through MCTS search, so
there's no MCTS data for it). `sample_qwen3_8b_dart_math_diff1.json` in
this directory is 5 real records pulled from one such file, enough to see
the shape of the data without decompressing anything.

Format:

```json
{"samples": [ { ... one record per question ... } ]}
```

Each record:

| field | type | meaning |
|---|---|---|
| `final_valid_transitions` | list of int-lists | every distinct **layer path** the search found that reaches the correct answer |
| `final_invalid_transitions` | list of int-lists | every distinct layer path the search tried that did *not* reach the correct answer |
| `initial_transition_metric` | float | reward of the identity program (all layers kept, in order); `0.0` means the model gets this question wrong even with no layer edits at all |
| `sample_info` | dict | `query_id` (the join key back into the reconstructed DART-Math data), `domain`, `math_level`, `source`, `fail_rate`, `tot_n_samples`, `difficulty` |
| `search_trajectory` | list of dicts | the raw MCTS visit log, one entry per simulation: `{"path", "parent_path", "reward"}`. Lets you replay the exact tree the search built, not just its final valid/invalid path sets. |

A **layer path** is a flat list of layer indices, the exact sequence of
transformer layers the model ran, in order. It's what
`re_polar.core.layer_engine.LayerEngine.apply_layer_rerouting()` executes
directly:

- The identity path is `[0, 1, 2, ..., num_layers - 1]`, every layer once, in order.
- A **skipped** layer is simply missing from the path.
- A **repeated** layer appears consecutively more than once.

For example, `search_trajectory[0]["path"]` for one record in the sample
file is:

```
[0, 1, 2, 3, 4, 4, 4, 5, 6, ...]
```

Layer 4 runs 3 times in a row (a repeat), everything else in that stretch
runs once.

This flat path is the *ground truth* the search records. `re_polar.core`
(`Program`/`Segment`, see [`re_polar/core/ir.py`](../re_polar/core/ir.py))
re-expresses the same information as contiguous ≤4-layer segments, each
tagged `keep`/`skip`/`repeat` — the structured form the router is trained
to predict. Every path in this data round-trips through it
(`re_polar.router.train.program_from_layer_path` decodes a path back to a
`Program`; `Program.to_layer_path()` must reproduce the original path
exactly).

### Generated text (`data/generated_answers/`)

`final_valid_transitions`/`final_invalid_transitions` say *whether* a
program was correct, not what the model actually generated to earn that
label. For a sample of evaluations, the raw generated text is here:

```
generated_answers/<model>/diff<1..5>.jsonl.gz
```

One JSON object per line:

| field | meaning |
|---|---|
| `query_id` | same id as `mcts_results/`'s `sample_info.query_id` and the reconstructed DART-Math data's `query_id` — join directly on this |
| `path` | the layer path that was executed |
| `reward` | 0.0 or 1.0, same meaning as `final_valid_transitions`/`initial_transition_metric` |
| `text` | the model's raw generated output for this (question, path) |
| `batch_id`, `batch_pos`, `batch_size` | which generation batch this row was part of, and its position in it — batch composition affects bf16 rounding, so this lets an exact batch be reconstructed if needed |
| `difficulty` | 1-5, redundant with the file path, kept so a row is self-contained |

## Reading the data

Every large file here (`mcts_results/`, `generated_answers/`,
`menu_crossexec/`) ships gzip-compressed to keep the repo small. Nothing
about the content changes — decompressing one gives back exactly the
plain-text JSON or JSONL described above. You don't need to unzip anything
by hand: `re_polar.datasets.schemas.load_samples` and
`re_polar.mcts.analysis.loader.load_menu_crosschecks` accept a `.gz` path
directly. To inspect a file yourself:

```python
import json, gzip
json.load(gzip.open("path/to/file.json.gz", "rt"))          # mcts_results/, menu_crossexec/
[json.loads(l) for l in gzip.open("path/to/file.jsonl.gz", "rt")]  # generated_answers/
```

## Joining questions back onto the MCTS results

`mcts_results/` doesn't carry question text or ground-truth answers
directly (only `query_id`). Reconstruct DART-Math (above), then join:

```bash
python -m re_polar.datasets.attach_mcts_questions \
    --mcts-dir data/mcts_results --dart-math-dir ./data/dart_math \
    --out-dir ./data/mcts_results_full
```

This writes the same directory layout, `query_id`-joined against the
reconstructed data, with `question`/`gt_ans` added to every record —
this is the format `re_polar.router.train` and everything under
[`analysis/`](../analysis/) expects.

## Menu cross-execution data (`data/menu_crossexec/<model>/diff<1..5>.json.gz`)

Rank non-identity programs by how many questions they solved during
search, take the top-K, then actually execute each of those against every
question in the pool (not just the ones the original search happened to
try it on) — does real execution reach the coverage the search's own
records would suggest? Produced by
[`analysis/mcts_design/menu_coverage.py`](../analysis/mcts_design/menu_coverage.py).

```json
{"model": str, "k": int, "schema": str,
 "per_difficulty": [ { ...one entry per difficulty... } ]}
```

Each `per_difficulty` entry:

| field | meaning |
|---|---|
| `difficulty`, `n_pool`, `n_train`, `n_val`, `n_test` | pool size and split sizes for this difficulty (same split as `dart_math_splits.json`) |
| `k_requested` / `k_actual` | requested vs. actually-ranked menu size |
| `qids` | every pool question's `query_id`, in a fixed order — everything below indexes positionally against this list |
| `per_program` | one entry per menu program: `path` (its layer path), `rewards` (one 0/1 per `qids` position), `provenance` (one char per `qids` position: `v` = already known-valid from the original search and reused as-is, `i` = already known-invalid and reused, `f` = freshly executed for this cross-check, since the original search only tried each program against a subset of the pool) |
| `provenance_legend` | the `v`/`i`/`f` meanings, written into the file itself |
| `n_reused_cells` / `n_new_cells` | how many `v`/`i` vs. `f` cells, across the whole grid |
| `real_topk_coverage_curve`, `final_real_coverage` | coverage vs. menu size K |

## Router checkpoints (`data/router_checkpoints/<model>/diff<1..5>.pt`)

Trained router weights (PyTorch state dict), one per (model, difficulty),
the winning training recipe. Load via
`re_polar.router.train.load_checkpoint` (used by
`re_polar/router/infer.py --checkpoint`).

```json
{"state_dict": {...}, "meta": {"num_layers", "n_ops", "d_model", "nheads",
 "n_layer_blocks", "embed_dim", "embedding_model_name", "max_question_tokens",
 "ops"}, "train_meta": {...training config and val metrics...}}
```

`state_dict` holds only the trainable head; the frozen embedding encoder
(`meta.embedding_model_name`) reloads its own weights from Hugging Face.
