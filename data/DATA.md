# Data

Everything the paper's MCTS searches, routers and fixed programs produced. For each
DART-Math question: every program-of-layers a search evaluated with its verdict, the model's
generated answer behind each verdict, the trained routers with their test results, the
fixed program per level, and the menu cross-execution of Finding 5. The data comes in three groups: the main track, Li et
al.'s setup on the PoLar split, and the cross-checks between the two.

The small id files below are in this repository; everything else (about 6 GB) is hosted as
the Hugging Face dataset [`SWAMBIT/re-polar`](https://huggingface.co/datasets/SWAMBIT/re-polar)
and downloads into `data/` (all of it, or only the parts you need):

```bash
pip install -U huggingface_hub
hf download SWAMBIT/re-polar --repo-type dataset --local-dir data
hf download SWAMBIT/re-polar --repo-type dataset --local-dir data --include "router_results/*" "fixed_program/*"
```

```
data/
  dart_math_splits.json          our split: query_ids per level and train/val/test
  polar_split_ids.json           Li et al.'s split: query_ids per level (pool, train/val/test)
  sample_qwen3_8b_dart_math_diff1.json   5 records of one MCTS file, to see the shape

  # main track: our split, our search, oneshot prompt (paper Sections 3-4)
  mcts_results/<model>/seed<S>/dart-math-diff-<L>/merged_mcts_samples.json.gz
  generated_answers/<model>/seed<S>/diff<L>.jsonl.gz
  router_checkpoints/<model>/seed<S>/diff<L>.pt
  router_results/<model>/seed<S>/diff<L>.json
  fixed_program/<model>/seed<S>/diff<L>.json
  menu_crossexec/<model>/diff<L>.json.gz                       (seed 42, in-sample)
  menu_crossexec_heldout/<model>/diff<L>.json.gz               (seed 42, held-out)

  # PoLar split: Li et al.'s search, prompts and questions (paper Appendix B)
  polar_split/mcts_results/<model>/<prompt>/seed<S>/dart-math-diff-<L>/merged_mcts_samples.json.gz
  polar_split/generated_answers/<model>/<prompt>/seed<S>/diff<L>.jsonl.gz
  polar_split/router_results/<model>/<prompt>/seed<S>/diff<L>/{fold0..fold3,split}.json
  polar_split/fixed_program/<model>/<prompt>/seed<S>/diff<L>/fold{0..3}.json
  polar_split/router_checkpoints/<model>/<prompt>/seed<S>/diff<L>.pt   (trained on `split`)
  polar_split/ood/<model>/{router.pt, router_asdiv_mawps.json, fixed_program.json,
                           router_and_fixed_mmlu_pro.json}

  # cross-checks (paper Appendix B.4)
  cross_checks/their_search_our_split/{mcts_results,generated_answers,router_results,fixed_program}/<model>/<prompt>/...
  cross_checks/our_search_polar_split/{mcts_results,router_results,fixed_program}/qwen3_8b/<prompt>/...
```

- `<model>`: `qwen15_moe_a27b`, `qwen25_3b`, `qwen25_7b`, `qwen3_8b`, `qwen3_32b`,
  `llama32_3b` (main track); `qwen15_moe_a27b`, `qwen25_3b`, `qwen3_8b`, `llama32_3b`
  (PoLar split). `<S>`: search seed `42`, `1`, `7`. `<L>`: difficulty level 1-5.
- `<prompt>`: `polar_default` (Li et al.'s prompt, code key `raw`) or `oneshot` (ours, one
  `1+1` demo, code key `paper_minimal_fewshot`). The main track uses `oneshot` throughout.
- Search configuration: main track = `run_search_dart_math` defaults (`--split all`);
  PoLar split = `--polar-fidelity --split train` on the level's whole question pool;
  their search on our split = `--polar-fidelity --split trainval`; our search on the
  PoLar split = `--polar-fidelity --widening-search`.
- Batch shape (it only bounds memory, but batched bf16 decoding is not bit-identical across
  shapes, so an exact rerun needs the same values): `--masked-batch-buckets` 8, except DM-5 of
  LLaMA-3.2-3B (all seeds) and of Qwen3-32B (seeds 1 and 7), 16; each generated-answer row's
  `batch_size` records the bucket it was decoded in. `--masked-batch-max-group-size` of the
  seed-1/7 runs: 192 for Qwen2.5-3B/7B, 64 for Qwen1.5-MoE and LLaMA-3.2-3B, 128 for Qwen3-32B
  (DM-5: 64), 768 for Qwen3-8B; PoLar split: 64. Qwen3-8B seed 42 of the main track ran without
  the KV cache (`--no-masked-batch-kv`); every other run used it.
- Software: torch 2.11.0 (CUDA 12.8), flash-attn 2.8.3, transformers 5.12.1 and 5.17.0 for the
  searches, 5.17.0 for routers, fixed programs and baselines (`requirements.txt` pins it).
  Runs used A100, H100, H200 and B200 GPUs; greedy decoding differs slightly across
  architectures, so an exact rerun needs the same one. Router and menu result files name it in
  `checkpoint` / `gpu_arch`.
- Not available: answer texts for Qwen3-8B with `polar_default` (all seeds) and for
  our search on the PoLar split; those runs did not log answers. Their programs and
  router results are released.

## Getting the questions

Question text and reference answers aren't stored in any file here. The MATH half of
DART-Math traces back to `hendrycks/competition_math`, which is no longer distributed
(paper Appendix A.1), so we release scripts that rebuild both splits from the public
DART-Math pool instead:

```bash
python -m re_polar.datasets.dart_math --out-dir ./data/dart_math --include-gsm8k
python -m re_polar.datasets.polar_split --out-dir ./data/polar_split_questions
```

`dart_math` writes `diff{1..5}/{train,val,test}.json` (1250/250/500 questions each) plus a
`manifest.json`. Each record:

| field | type | meaning |
|---|---|---|
| `query_id` | str | stable id, the join key into every file below |
| `question` | str | the input question text |
| `gt_ans` | str | ground-truth answer |
| `domain` | str | math subject area |
| `math_level` | int or null | MATH's own difficulty level (1-5); `null` for GSM8K questions, which don't have one |
| `source` | str | `"math"` or `"gsm8k"` |
| `fail_rate` | float | DART-Math's own per-query fail rate, used to bin questions into difficulty 1 (easiest) through 5 (hardest) |
| `tot_n_samples` | int | how many responses DART-Math sampled to compute `fail_rate` |
| `difficulty` | int | 1-5, the difficulty tier this record was binned into |

`polar_split` writes `pool/diff{1..5}/train.json` (every question of the level, 565 /
1349 / 1579 / 1537 / 1577, MATH only; the search input) and `split/diff{1..5}/{train,val,
test}.json` (Li et al.'s inferred train/val/test cut, test 141 / 337 / 395 / 384 / 394),
same record fields. The router's 4-fold cross-validation over the pool is
`re_polar.router.fold_data.fold_qids(pool_query_ids, fold)`.

`dart_math_splits.json` and `polar_split_ids.json` record the same membership as plain
`query_id` lists, to check a rebuild without the question text.

## MCTS results (`mcts_results/`, `polar_split/mcts_results/`, `cross_checks/*/mcts_results/`)

For every question, every distinct program-of-layers the search evaluated, and whether it
answered correctly. This is what the routers and fixed programs are trained on. One file
per (model, [prompt,] seed, level): 2000 questions on our split, in the order train
(1250), val (250), test (500); the level's whole pool on the PoLar split.

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
| `sample_info` | dict | `query_id` (the join key back into the reconstructed DART-Math data), `domain`, `math_level`, `source`, `fail_rate`, `tot_n_samples`, `difficulty`, `split` (`train`/`val`/`test` of our split; `train` for every PoLar-split question, whose search runs over the level's whole pool) |
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

### Generated text (`generated_answers/`)

`final_valid_transitions`/`final_invalid_transitions` say *whether* a
program was correct, not what the model generated to earn that label. The model's
output for every evaluation of a search is here, one file per (model, [prompt,] seed,
level). The text is the model's continuation only (capped at 2,000 characters); under the
`oneshot` prompt it often goes on to write a new, made-up problem after its answer.

The model sometimes restates its question, so restatements are redacted: every run of 10 or
more words that copies a question of our split or the PoLar split (case-insensitive, whitespace
normalised) is replaced by `[question]`, and so is a full copy of the row's own question when
that question has fewer than 10 words. About 5% of the answers are affected; `reward` is the
verdict on the original, unredacted text.

One JSON object per line:

| field | meaning |
|---|---|
| `query_id` | same id as `mcts_results/`'s `sample_info.query_id` and the reconstructed DART-Math data's `query_id` — join directly on this |
| `path` | the layer path that was executed |
| `reward` | 0.0 or 1.0, same meaning as `final_valid_transitions`/`initial_transition_metric` |
| `text` | the model's generated output for this (question, path), restatements of the question replaced by `[question]` (above) |
| `batch_id`, `batch_pos`, `batch_size` | which generation batch this row was part of, and its position in it — batch composition affects bf16 rounding, so this lets an exact batch be reconstructed if needed |
| `difficulty` | 1-5, redundant with the file path, kept so a row is self-contained |

## Router results (`router_results/`)

`re_polar.router.infer --top-k-paths 5` on the test questions (main track: the 500 test
questions of the level; PoLar split: the fold's test quarter, or Li et al.'s test set in
`split.json`).

```json
{"model": str, "checkpoint": str, "per_difficulty": [ {...} ], "overall": {...},
 "programs": {"<level>": [ {...one entry per test question...} ]}}
```

`per_difficulty`: `n`, `identity_acc`, `router_acc` (pass@1), `pass_at` (pass@1..5),
`router_acc_at_k`, `rescue_at1` / `harm_at1` (questions where the top-1 program is right
and identity wrong, and vice versa), `mcnemar_p`, `n_nonidentity_top1`,
`n_distinct_top1`, `top_nonidentity_share`, and program-shape statistics. `programs`:
per question `query_id`, the top-1 `layer_path`, `identity_correct`, `nonidentity`, and
`candidates` (the 5 programs, top-1 first, each with `layer_path` and `correct`).

## Fixed programs (`fixed_program/`)

`re_polar.router.fixed_program`: one program per level, the same for every question,
picked on the validation questions among the 20 programs with the highest shrunk success
rate on the training questions.

```json
{"difficulty": int, "menu_size": int, "prompt_style": str, "n": {"train", "val", "test"},
 "methods": {"fixed": {"winner": {"layer_path", "train_prior", ...},
                       "val": {...}, "test": {...}, "all_val": [...]}}}
```

`test`: `identity_acc`, `pass_at` (pass@1 is the fixed program's accuracy), `rescue_at1`,
`harm_at1`, `mcnemar_p`, and `per_question` (`query_id`, `identity_correct`,
`top1_correct`). `all_val`: every candidate's validation result.

## Out-of-distribution routers (`polar_split/ood/<model>/`)

One router and one fixed program per model, trained on all five levels of Li et al.'s
train/val split with the `polar_default` prompt (seed-42 search), evaluated on ASDiv and
MAWPS (`router_asdiv_mawps.json`, `fixed_program.json`) and MMLU-Pro
(`router_and_fixed_mmlu_pro.json`, from `analysis.mmlu_pro_track.infer_router_mmlu
--fixed-program-json`). Paper Appendix B.7.

## Reading the data

Every large file ships gzip-compressed. Decompressing one gives back exactly the
plain-text JSON or JSONL described above. `re_polar.datasets.schemas.load_samples` and
`re_polar.mcts.analysis.loader.load_menu_crosschecks` accept a `.gz` path directly. To
inspect a file yourself:

```python
import json, gzip
json.load(gzip.open("path/to/file.json.gz", "rt"))          # mcts_results/, menu_crossexec/
[json.loads(l) for l in gzip.open("path/to/file.jsonl.gz", "rt")]  # generated_answers/
```

## Joining questions back onto the MCTS results

Rebuild the questions (above), then join, per model:

```bash
python -m re_polar.datasets.attach_mcts_questions \
    --mcts-dir data/mcts_results/qwen3_8b --dart-math-dir ./data/dart_math \
    --out-dir ./data/mcts_results_full/qwen3_8b
python -m re_polar.datasets.attach_mcts_questions \
    --mcts-dir data/polar_split/mcts_results/qwen3_8b/oneshot \
    --dart-math-dir ./data/polar_split_questions/pool \
    --out-dir ./data/mcts_results_full/polar_split/qwen3_8b/oneshot
```

This writes the same directory layout (`seed<S>/dart-math-diff-<L>/`), with
`question`/`gt_ans` added to every record: the format `re_polar.router.train`,
`re_polar.router.fixed_program` and everything under [`analysis/`](../analysis/) expect.
For the PoLar-split routers, `re_polar.router.fold_data` joins the text itself.

## Menu cross-execution data (`menu_crossexec/`, `menu_crossexec_heldout/`)

Rank programs, identity included, by how many questions they solved during the seed-42
search, take the top 100, then execute each of them against every question of a question set
(not just the ones the search happened to try it on), with the search's own `oneshot`
prompt. Produced by
[`analysis/mcts_design/menu_coverage.py`](../analysis/mcts_design/menu_coverage.py); both
sets are compared by [`analysis/mcts_design/menu_heldout.py`](../analysis/mcts_design/menu_heldout.py).

- `menu_crossexec/<model>/diff<L>.json.gz`: in-sample, ranked and executed on all 2,000
  questions of the level (`--rank-pool all --eval-pool all`; paper Fig. 4a and Fig. 9).
- `menu_crossexec_heldout/<model>/diff<L>.json.gz`: held-out, ranked on the 1,500 train+val
  questions and executed on the 500 test questions (`--rank-pool trainval --eval-pool test`).
  Cells already in the in-sample grid were taken from it (`--reuse-grid`, provenance `g`).

```json
{"model": str, "k": int, "prompt_style": str, "schema": "menu_crossexec_v2",
 "per_difficulty": [ { ...one entry per difficulty... } ]}
```

Each `per_difficulty` entry:

| field | meaning |
|---|---|
| `difficulty`, `n_pool`, `n_train`, `n_val`, `n_test` | pool size and split sizes for this difficulty (same split as `dart_math_splits.json`) |
| `k_requested` / `k_actual` | requested vs. actually-ranked menu size |
| `qids` | every pool question's `query_id`, in a fixed order — everything below indexes positionally against this list |
| `per_program` | one entry per menu program: `path` (its layer path), `rewards` (one 0/1 per `qids` position) and `provenance` (one letter per `qids` position saying where that 0/1 comes from: `v` / `i` = the search already ran this program on this question (valid / invalid), so it was not run again; `f` = run for this grid, because the search tried each program on only some questions; `g` = copied from another grid that already ran this program on this question: the in-sample grid for held-out grids (`reuse_grid`), the earlier grid of the same level for the LLaMA-3.2-3B and Qwen1.5-MoE DM-5 in-sample grids) |
| `provenance_legend` | the same letter meanings, stored in the file |
| `n_reused_cells` / `n_new_cells` | how many reused (`v`/`i`/`g`) vs. `f` cells, across the whole grid |
| `rank_pool`, `eval_pool`, `reuse_grid` | held-out grids only: `trainval`, `test`, and the in-sample grid the `g` cells come from |
| `real_topk_coverage_curve`, `final_real_coverage` | coverage vs. menu size K |

## Router checkpoints (`router_checkpoints/`, `polar_split/router_checkpoints/`, `polar_split/ood/`)

Trained router weights (PyTorch state dict). Load via
`re_polar.router.train.load_checkpoint` (used by
`re_polar/router/infer.py --checkpoint`).

```json
{"state_dict": {...}, "meta": {"num_layers", "n_ops", "d_model", "nheads",
 "n_layer_blocks", "embed_dim", "embedding_model_name", "max_question_tokens",
 "ops"}, "train_meta": {...training config and val metrics...}}
```

`state_dict` holds only the trainable head; the frozen embedding encoder
(`meta.embedding_model_name`) reloads its own weights from Hugging Face.
