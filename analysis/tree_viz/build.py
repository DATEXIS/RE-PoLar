#!/usr/bin/env python3
"""Merge one or more tree-data JSONs (see extract.py) into template.html -> one
standalone HTML file with no external dependencies (open it directly in a
browser, no server needed). Several JSONs become model, run and seed pickers in the viewer,
named by each one's --model-label, --label and --seed; they must hold the same questions in
the same order, so that switching stays on the question.

Usage:
    python -m analysis.tree_viz.build data.json out.html
    python -m analysis.tree_viz.build ours.json theirs.json out.html
    python -m analysis.tree_viz.build *.json out.html --lazy tree_data
    python -m analysis.tree_viz.build data.json out.html --template /path/to/template.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PLACEHOLDER = "__DATA_JSON__"
DEFAULT_TEMPLATE = Path(__file__).parent / "template.html"


def build(data_json: str, template_path: Path = DEFAULT_TEMPLATE) -> str:
    json.loads(data_json)  # fail fast on malformed data before embedding it
    tpl = Path(template_path).read_text()
    if PLACEHOLDER not in tpl:
        raise ValueError(f"{template_path} is missing the {PLACEHOLDER} placeholder")
    return tpl.replace(PLACEHOLDER, data_json)


def add_fit(runs: list[dict]) -> None:
    """Give a question the same tree width and depth in all seeds of a run, so the viewer draws
    it at the same zoom and switching seeds moves no layer."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["model_label"], r["label"]), []).append(r)
    for group in groups.values():
        for trees in zip(*(r["trees"] for r in group)):
            leaves = max(sum(not n["children"] for n in t["nodes"].values()) for t in trees)
            depth = max(t["max_edit_count"] for t in trees)
            for t in trees:
                t["fit_leaves"], t["fit_depth"] = leaves, depth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+", help="tree-data JSON(s) produced by extract.py")
    ap.add_argument("output", help="path to write the HTML file")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument(
        "--lazy",
        metavar="DIR",
        help="write each run to DIR/<key>.json next to the output and load it only when "
        "selected; the HTML then needs a web server",
    )
    args = ap.parse_args()

    out = Path(args.output)
    runs = [json.loads(Path(p).read_text()) for p in args.data]
    if len(runs) > 1:
        ids = [[t["query_id"] for t in r["trees"]] for r in runs]
        assert all(x == ids[0] for x in ids), "runs must hold the same questions in the same order"
        add_fit(runs)
        for r in runs:
            r["key"] = re.sub(
                r"\W+", "_", f"{r['model_label']}_{r['label']}_seed{r['seed']}"
            ).lower()
            for n, t in enumerate(r["trees"], 1):
                t["label"] = f"Question {n}"
        if args.lazy:
            (out.parent / args.lazy).mkdir(exist_ok=True)
            for r in runs:
                data = dict(base_depth=r["base_depth"], trees=r.pop("trees"))
                (out.parent / args.lazy / f"{r['key']}.json").write_text(json.dumps(data))
                r["src"] = f"{args.lazy}/{r['key']}.json"
        data_json = json.dumps(dict(models=runs))
    else:
        data_json = json.dumps(runs[0])
    html = build(data_json, Path(args.template))
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
