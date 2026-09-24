"""Supervision-data emitter: PoLar's exact merged_mcts_samples.json format.

Schema learned from PoLar's own repo (`polar/{data,config}.py`; consulted, never
imported) and VALIDATED end-to-end by training their `run_polar.py`
on a file we generated:

  <root>/<model_path>/dart-math-diff-<N>/merged_mcts_samples.json
  {"samples": [{"question": str, "gt_ans": str,
                "final_valid_transitions": [[int, ...], ...],
                "final_invalid_transitions": [[int, ...], ...],   # extra; their reader ignores it
                "initial_transition_metric": float,
                "search_trajectory": [{"path", "parent_path", "reward"}, ...],  # extra; see below
                "sample_info": {...}}]}                            # extra provenance; their reader tolerates it

Their reader uses: question, gt_ans, final_valid_transitions,
initial_transition_metric. We additionally record invalid paths and
provenance, harmless to their trainer, useful to ours.

search_trajectory: ProgramMCTS.trajectory verbatim, one entry
per update() call in the exact order simulations happened -- {"path": the
executed layer-path of the node just evaluated, "parent_path": its parent's
executed layer-path (None for root), "reward": the reward it got}. Lets a
consumer reconstruct the ACTUAL tree the search built (parent/child edges) and
replay it in visit order, for visualization or auditing -- not just the final
valid/invalid path sets. None (default) when the caller doesn't pass one
(offline scripts that build sample_record from something other than a live
ProgramMCTS, e.g. tests) -- omitted from the record entirely in that case, not
written as a null, so old readers see exactly the schema they always did.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def sample_record(
    inp: dict,
    valid_paths: List[List[int]],
    invalid_paths: List[List[int]],
    initial_metric: float,
    trajectory: Optional[List[dict]] = None,
) -> dict:
    record = {
        "question": inp["question"],
        "gt_ans": inp["gt_ans"],
        "final_valid_transitions": valid_paths,
        "final_invalid_transitions": invalid_paths,
        "initial_transition_metric": initial_metric,
        "sample_info": {k: v for k, v in inp.items() if k not in ("question", "gt_ans")},
    }
    if trajectory is not None:
        record["search_trajectory"] = trajectory
    return record


def write_merged_samples(
    out_root: Path, model_path: str, namespace: str, samples: List[dict]
) -> Path:
    out = Path(out_root) / model_path / namespace / "merged_mcts_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"samples": samples}, f)
    print(f"Wrote {len(samples)} samples -> {out}")
    return out


def load_samples(path: Path) -> List[dict]:
    """Read a merged_mcts_samples.json[.gz] file back into its list of sample dicts."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as f:
            return json.load(f)["samples"]
    with open(path) as f:
        return json.load(f)["samples"]
