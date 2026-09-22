"""Loads `merged_mcts_samples.json` sources and groups them for an insight run.

Deliberately thin, reuses `re_polar.datasets.schemas.load_samples` (the same reader
`re_polar/mcts` and the router pipeline use) rather than reimplementing JSON
loading, so this stays in sync with the schema by construction.
"""
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from re_polar.datasets.schemas import load_samples


@lru_cache(maxsize=None)
def _load_source_cached(path: str) -> tuple:
    return tuple(load_samples(Path(path)))


def load_source(path) -> List[dict]:
    """One `merged_mcts_samples.json` file -> its list of sample dicts.

    Memoized by path: profiling a full report build showed the SAME file
    being parsed multiple times per build -- once for the main groups,
    again for the pooled group, again per axis breakdown, etc. -- with raw
    JSON parsing alone dominating total build time. No sample dict is ever
    mutated in place anywhere in `re_polar/mcts/analysis` (checked), so sharing the
    cached list across callers is safe; still returns a fresh `list(...)`
    per call so a caller mutating the OUTER list (e.g. `.append`) can't
    corrupt the cache."""
    return list(_load_source_cached(str(path)))


def build_groups(sources: List[dict], group_by: Optional[str]) -> Dict[str, List[dict]]:
    """`sources` = [{"label": str, "path": str}, ...].

    If `group_by` is set, every loaded sample from every source is bucketed
    by `sample_info[group_by]` (e.g. "difficulty", "domain", "category"):
    samples from different source files with the same group value are
    pooled together (e.g. pooling per-domain slices across several
    per-difficulty source files into one per-domain group). If `group_by`
    is None, each source's own `label` is the group (one group per source
    file, e.g. one per DART difficulty), which is also correct for a
    single-source report.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for src in sources:
        samples = load_source(src["path"])
        if group_by:
            for s in samples:
                key = s.get("sample_info", {}).get(group_by, "unknown")
                groups[str(key)].append(s)
        else:
            groups[src["label"]].extend(samples)
    return dict(groups)


def build_pooled_group(sources: List[dict], label: str = "pooled") -> Dict[str, List[dict]]:
    """Every sample from every source, in ONE bucket -- for insights that
    report one pooled reading (e.g. "DART pooled" / "mmlu pooled") rather
    than a per-difficulty/per-domain breakdown."""
    samples: List[dict] = []
    for src in sources:
        samples.extend(load_source(src["path"]))
    return {label: samples}


def _dense_coverage_curve(programs: List[dict]) -> List[dict]:
    """Cumulative-OR real-coverage curve at EVERY integer k = 1..len(programs)
    (never sparse checkpoints), from a list of `{"rewards": [0/1 x n_pool]}`
    dicts already in rank order. Pure local recomputation from data already
    present in `per_program` -- no re-execution, no new grading."""
    n_pool = len(programs[0]["rewards"]) if programs else 0
    covered = [False] * n_pool
    curve = []
    for k, prog in enumerate(programs, start=1):
        for i, r in enumerate(prog["rewards"]):
            if r:
                covered[i] = True
        curve.append({"k": k, "real_coverage": (sum(covered) / n_pool) if n_pool else 0.0})
    return curve


def _with_excl_identity_curve(entry: dict) -> dict:
    """Recomputes BOTH the identity-included and identity-excluded real-
    coverage curves for one `per_difficulty` crosscheck entry, densely (every
    integer k, not sparse checkpoints), from the same `per_program` reward
    data by the exact same method -- so the two are guaranteed consistent
    with each other and with themselves. A version that densified only one
    of the two curves and left the other at its raw JSON's sparse
    checkpoints (e.g. only k in [1,3,5,10,20,30,50,100]) risks a rendering
    gap: a chart line-drawing routine that skips a null-valued point can
    leave the very next real point undrawn too, if it treats "starts right
    after a null" as "nothing to connect" -- recomputing every k for both
    curves up front avoids that class of bug entirely, independent of
    whatever a later rendering step chooses to display.

    Overwrites `real_topk_coverage_curve`/`final_real_coverage` (originally
    the raw JSON's sparse values) with the dense recomputation -- asserts the
    k=100 recomputed value matches the JSON's own recorded `final_real_
    coverage` first (same reward data, so they must agree; a mismatch means
    something is actually wrong, not just sparse-vs-dense, and should fail
    loud rather than silently diverge).

    Silently returns `entry` unchanged if `per_program` is missing/empty, or
    its programs don't carry a `rewards` array -- both are real cases, not
    just hypothetical: an older crosscheck-file format has `per_program`
    entries shaped `{path, n_solved, n_found_valid_by_search}` -- aggregate
    counts only, no per-question reward vector to recompute a dense curve
    from. Same fallback covers the minimal fixtures a unit test writes by
    hand. Any downstream renderer is expected to treat all four fields this
    function touches as optional (`.get(...)`), since a report can mix old-
    and new-format crosscheck data across difficulties.
    """
    programs = entry.get("per_program")
    if not programs or not all("rewards" in p for p in programs):
        return entry
    identity_path = list(range(len(programs[0]["path"])))
    if programs[0]["path"] != identity_path:
        raise ValueError(
            f"expected per_program[0] to be the identity program {identity_path}, "
            f"got {programs[0]['path']} (difficulty={entry.get('difficulty')}) -- "
            "refusing to silently drop the wrong program"
        )
    incl_curve = _dense_coverage_curve(programs)
    recorded_final = entry.get("final_real_coverage")
    if recorded_final is not None and abs(incl_curve[-1]["real_coverage"] - recorded_final) > 1e-9:
        raise ValueError(
            f"recomputed k={len(programs)} coverage {incl_curve[-1]['real_coverage']} != "
            f"recorded final_real_coverage {recorded_final} (difficulty={entry.get('difficulty')}) "
            "-- same reward data should agree exactly, this points at a real bug, not sparse-vs-dense"
        )
    excl_curve = _dense_coverage_curve(programs[1:])
    return {
        **entry,
        "real_topk_coverage_curve": incl_curve,
        "final_real_coverage": incl_curve[-1]["real_coverage"],
        "real_topk_coverage_curve_excl_identity": excl_curve,
        "final_real_coverage_excl_identity": excl_curve[-1]["real_coverage"] if excl_curve else 0.0,
    }


def load_menu_crosschecks(crosscheck_dir, model: str) -> Dict[str, dict]:
    """Loads menu cross-execution check output for one model: scans
    `crosscheck_dir/<model>/diff*.json[.gz]` and indexes every `per_difficulty`
    entry inside by `str(difficulty)`, so it lines up directly with
    `build_groups(..., group_by="difficulty")`'s own group labels.

    Returns `{}` (not an error) if the directory doesn't exist yet or is
    empty -- the caller is meant to render that as a nudge, not fail on it.
    A later file wins over an earlier one for the same difficulty (a rerun
    is assumed to supersede, not duplicate).
    """
    out: Dict[str, dict] = {}
    d = Path(crosscheck_dir) / model
    if not d.is_dir():
        return out
    for f in sorted(d.glob("diff*.json")) + sorted(d.glob("diff*.json.gz")):
        try:
            if f.suffix == ".gz":
                import gzip
                with gzip.open(f, "rt") as fh:
                    data = json.load(fh)
            else:
                data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for entry in data.get("per_difficulty", []):
            key = str(entry["difficulty"])
            entry = _with_excl_identity_curve(entry)
            out[key] = {**entry, "_source_file": f.name, "_k_requested": data.get("k")}
    return out
