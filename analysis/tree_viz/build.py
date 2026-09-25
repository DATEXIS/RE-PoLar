#!/usr/bin/env python3
"""Merge a tree-data JSON (see extract.py) into template.html -> one
standalone HTML file with no external dependencies (open it directly in a
browser, no server needed).

Usage:
    python build.py data.json out.html
    python build.py data.json out.html --template /path/to/template.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "__DATA_JSON__"
DEFAULT_TEMPLATE = Path(__file__).parent / "template.html"


def build(data_json: str, template_path: Path = DEFAULT_TEMPLATE) -> str:
    json.loads(data_json)  # fail fast on malformed data before embedding it
    tpl = Path(template_path).read_text()
    if PLACEHOLDER not in tpl:
        raise ValueError(f"{template_path} is missing the {PLACEHOLDER} placeholder")
    return tpl.replace(PLACEHOLDER, data_json)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", help="path to a tree-data JSON produced by extract.py")
    ap.add_argument("output", help="path to write the standalone HTML file")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    args = ap.parse_args()

    data_json = Path(args.data).read_text()
    html = build(data_json, Path(args.template))
    out = Path(args.output)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
