"""JSON report-config schema + loader.

One config = one report: `{"model": ..., "sources": [{"label", "path"}, ...],
"insights": [insight_id, ...]}`, optionally "title"/"group_by"/"description"/
"crosscheck_dir" (see `ReportConfig` below for the full field list). Kept as
a plain dataclass (no pydantic dependency), this package should stay
import-light.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

# Human-readable headings for known group_by axes -- falls back to "by {key}"
# for anything not listed here, so a new sample_info key needs no code change,
# just a nicer label added here if you want one.
AXIS_LABELS = {
    "difficulty": "by difficulty",
    "domain": "by MATH domain",
    "category": "by category",
}


@dataclass
class ReportConfig:
    model: str                      # key into MODEL_REGISTRY (fixes num_layers)
    sources: List[dict]             # [{"label": str, "path": str}, ...]
    insights: List[str]             # registered insight IDs to include, in order
    title: str = "Analysis report"
    # One or more sample_info keys to group by, e.g. "difficulty" or
    # ["difficulty", "domain"] -- each axis is rendered as its own parallel
    # breakdown within every group-aware insight section (matches the
    # reference artifact's "by difficulty" + "by MATH domain" side-by-side
    # structure). A single string is still accepted for backward compat and
    # normalized to a one-element list. None = one group per source file.
    group_by: List[Optional[str]] = field(default_factory=lambda: [None])
    description: str = ""
    # Optional: directory to auto-scan for `menu_coverage.py` output JSONs
    # (`<crosscheck_dir>/<model>/diff*.json[.gz]`, see loader.py's
    # `load_menu_crosschecks`). Not an insight ID in `insights` -- it's a
    # standing mechanism, not something to
    # remember to opt into per report: when `menu_concentration` is included
    # and this is set, the report auto-appends a comparison section right
    # after it for whichever groups have a matching file in this directory,
    # and a "not run yet" nudge for the ones that don't. None = feature off
    # (mirrors the "if it doesn't exist, don't include it" ask -- an unset
    # dir means this report never opted in, so nothing is added at all).
    crosscheck_dir: Optional[str] = None


def load_config(path) -> ReportConfig:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    required = {"model", "sources", "insights"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{path}: report config missing required key(s): {sorted(missing)}")
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ValueError(f"{path}: 'sources' must be a non-empty list of {{label, path}} objects")
    for src in raw["sources"]:
        if "label" not in src or "path" not in src:
            raise ValueError(f"{path}: every source needs 'label' and 'path', got {src!r}")
    if not isinstance(raw["insights"], list) or not raw["insights"]:
        raise ValueError(f"{path}: 'insights' must be a non-empty list of insight IDs")

    # Relative source paths resolve against the CONFIG FILE's own directory,
    # not the current working directory -- so `--config a/b/cfg.json` works
    # identically no matter where you run the CLI from. Absolute paths pass
    # through untouched.
    base_dir = path.resolve().parent
    sources = [
        {**src, "path": str((base_dir / src["path"]).resolve()) if not Path(src["path"]).is_absolute()
                 else src["path"]}
        for src in raw["sources"]
    ]

    crosscheck_dir = raw.get("crosscheck_dir")
    if crosscheck_dir and not Path(crosscheck_dir).is_absolute():
        crosscheck_dir = str((base_dir / crosscheck_dir).resolve())

    raw_group_by = raw.get("group_by")
    if raw_group_by is None:
        group_by = [None]
    elif isinstance(raw_group_by, str):
        group_by = [raw_group_by]
    elif isinstance(raw_group_by, list) and raw_group_by:
        group_by = raw_group_by
    else:
        raise ValueError(f"{path}: 'group_by' must be a string, a non-empty list of strings, "
                          f"or omitted, got {raw_group_by!r}")

    return ReportConfig(
        model=raw["model"],
        sources=sources,
        insights=raw["insights"],
        title=raw.get("title", "Analysis report"),
        group_by=group_by,
        description=raw.get("description", ""),
        crosscheck_dir=crosscheck_dir,
    )
