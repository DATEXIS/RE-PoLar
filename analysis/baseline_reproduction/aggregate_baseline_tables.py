"""Aggregate baseline-reproduction result JSONs (from `baseline_passk.py`,
`baseline_mmlu_pro.py`, `baseline_identity_ood.py`, `chat_template_probe.py`)
into paper-Table-shaped markdown tables.

Paper reference numbers (Table 2/5/6/7, for the 4 models PoLar published)
are hardcoded below, transcribed from PoLar's paper (arXiv:2606.06574).

Usage:
    python -m analysis.baseline_reproduction.aggregate_baseline_tables
    python -m analysis.baseline_reproduction.aggregate_baseline_tables \
        --raw-dir raw --out-dir .
"""
import argparse
import json
from pathlib import Path

MODEL_ORDER = [
    "llama32_3b",
    "qwen25_3b",
    "qwen25_7b",
    "qwen15_moe_a27b",
    "qwen3_8b",
    "qwen3_32b",
]

# Paper Table 2/5/6/7 (DART-Math), verified directly against the ICML PDF.
# Only the 4 models PoLar actually published have a paper table.
PAPER_DART_MATH = {
    "llama32_3b": {
        "table": "Table 2",
        "tau0": [0.424, 0.286, 0.272, 0.276, 0.286],
        "p1": [0.406, 0.286, 0.274, 0.280, 0.292],
        "p2": [0.440, 0.350, 0.296, 0.304, 0.328],
        "p3": [0.462, 0.388, 0.312, 0.314, 0.344],
        "p4": [0.470, 0.414, 0.322, 0.322, 0.348],
        "p5": [0.476, 0.432, 0.328, 0.328, 0.356],
    },
    "qwen25_3b": {
        "table": "Table 6",
        "tau0": [0.220, 0.136, 0.110, 0.078, 0.052],
        "p1": [0.242, 0.160, 0.116, 0.082, 0.054],
        "p2": [0.326, 0.228, 0.160, 0.122, 0.086],
        "p3": [0.378, 0.268, 0.182, 0.138, 0.100],
        "p4": [0.406, 0.292, 0.192, 0.152, 0.106],
        "p5": [0.422, 0.302, 0.204, 0.158, 0.130],
    },
    "qwen15_moe_a27b": {
        "table": "Table 5",
        "tau0": [0.380, 0.246, 0.206, 0.152, 0.142],
        "p1": [0.354, 0.220, 0.146, 0.126, 0.066],
        "p2": [0.364, 0.242, 0.158, 0.136, 0.082],
        "p3": [0.388, 0.246, 0.166, 0.140, 0.096],
        "p4": [0.396, 0.250, 0.178, 0.148, 0.112],
        "p5": [0.400, 0.256, 0.186, 0.150, 0.118],
    },
    "qwen3_8b": {
        "table": "Table 7",
        "tau0": [0.416, 0.288, 0.190, 0.120, 0.130],
        "p1": [0.370, 0.230, 0.092, 0.096, 0.070],
        "p2": [0.420, 0.256, 0.116, 0.106, 0.088],
        "p3": [0.458, 0.260, 0.124, 0.114, 0.098],
        "p4": [0.470, 0.266, 0.126, 0.128, 0.102],
        "p5": [0.484, 0.278, 0.136, 0.134, 0.106],
    },
}
METRIC_KEYS = ["tau0", "p1", "p2", "p3", "p4", "p5"]
METRIC_LABELS = {"tau0": "τ=0", "p1": "p@1", "p2": "p@2", "p3": "p@3", "p4": "p@4", "p5": "p@5"}


def load(raw_dir: Path, pattern: str):
    return sorted(raw_dir.glob(pattern))


def fmt(x):
    return f"{x:.3f}" if x is not None else "—"


def fmt_delta(a, b):
    if a is None or b is None:
        return "—"
    d = (a - b) * 100
    return f"{d:+.1f}"


def model_sort_key(model):
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def build_dart_math(raw_dir: Path) -> str:
    files = load(raw_dir, "baseline_passk_*.json")
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        rows.append(d)
    rows.sort(key=lambda d: model_sort_key(d["model"]))

    lines = [
        "# DART-Math baseline: Base(τ=0) + Base(sampling) p@1..5",
        "",
        "Source: `baseline_passk.py`, `dart_math_v2` TEST split (500 q/difficulty, "
        "DM-1..5), prompt = `paper_minimal_fewshot`. Raw JSONs: "
        "`raw/baseline_passk_*.json`.",
        "",
        "Paper reference numbers (where a table exists) verified directly "
        "against the ICML PDF.",
        "",
    ]
    for d in rows:
        model = d["model"]
        env = d.get("env", {})
        pd = {r["difficulty"]: r for r in d["per_difficulty"]}
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"GPU: `{env.get('gpu_name', '?')}` &middot; n=500/difficulty")
        lines.append("")
        lines.append("### Ours")
        lines.append("")
        lines.append("| metric | DM-1 | DM-2 | DM-3 | DM-4 | DM-5 |")
        lines.append("|---|---|---|---|---|---|")
        ours = {}
        for mk in METRIC_KEYS:
            vals = []
            for diff in range(1, 6):
                r = pd.get(diff, {})
                if mk == "tau0":
                    v = r.get("base_greedy_p1")
                else:
                    v = r.get("base_sampling_pk_best_per_k", {}).get(mk[1:])
                vals.append(v)
            ours[mk] = vals
            lines.append(f"| {METRIC_LABELS[mk]} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")

        paper = PAPER_DART_MATH.get(model)
        if paper:
            lines.append(f"### Paper ({paper['table']})")
            lines.append("")
            lines.append("| metric | DM-1 | DM-2 | DM-3 | DM-4 | DM-5 |")
            lines.append("|---|---|---|---|---|---|")
            for mk in METRIC_KEYS:
                vals = paper[mk]
                lines.append(f"| {METRIC_LABELS[mk]} | " + " | ".join(fmt(v) for v in vals) + " |")
            lines.append("")
            lines.append("### Δ = ours − paper (pp)")
            lines.append("")
            lines.append("| metric | DM-1 | DM-2 | DM-3 | DM-4 | DM-5 |")
            lines.append("|---|---|---|---|---|---|")
            for mk in METRIC_KEYS:
                deltas = [fmt_delta(o, p) for o, p in zip(ours[mk], paper[mk])]
                lines.append(f"| {METRIC_LABELS[mk]} | " + " | ".join(deltas) + " |")
            lines.append("")
        else:
            lines.append("*No paper table exists for this model (not one of PoLar's 4 published study models).*")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_mmlu_pro(raw_dir: Path) -> str:
    files = load(raw_dir, "baseline_mmlu_pro_*.json")
    rows = [json.loads(f.read_text()) for f in files]
    rows.sort(key=lambda d: model_sort_key(d["model"]))

    lines = [
        "# MMLU-Pro baseline: Base(τ=0) per domain",
        "",
        "Source: `baseline_mmlu_pro.py`, official full `mmlu_pro_official` test "
        "split (12,032 rows / 14 categories, no per-domain cap), prompt = "
        "`paper_minimal_fewshot`. Raw JSONs: `raw/baseline_mmlu_pro_*.json`.",
        "",
        "**Paper reference numbers are NOT included here**, Tables 3/8/9/10 exist "
        "for llama32_3b/qwen3_8b/qwen25_3b/qwen15_moe_a27b; cross-check against "
        "the ICML PDF directly before citing.",
        "",
    ]
    for d in rows:
        model = d["model"]
        env = d.get("env", {})
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"GPU: `{env.get('gpu_name', '?')}` &middot; n={env.get('n', '?')} &middot; "
            f"overall Base(τ=0) = **{fmt(d.get('base_greedy_average'))}**"
        )
        lines.append("")
        lines.append("| domain | ours | n |")
        lines.append("|---|---|---|")
        for domain, v in sorted(d.get("per_domain", {}).items()):
            lines.append(f"| {domain} | {fmt(v.get('average'))} | {v.get('n', '?')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_ood(raw_dir: Path) -> str:
    files = load(raw_dir, "baseline_identity_ood_*.json")
    rows = [json.loads(f.read_text()) for f in files]
    by_model = {}
    for d in rows:
        by_model.setdefault(d["model"], {})[d["dataset"]] = d

    lines = [
        "# ASDiv + MAWPS OOD baseline: Base(τ=0)",
        "",
        "Source: `baseline_identity_ood.py`, prompt = `paper_minimal_fewshot`. "
        "Raw JSONs: `raw/baseline_identity_ood_*.json`.",
        "",
        "| model | ASDiv (n) | MAWPS (n) |",
        "|---|---|---|",
    ]
    for model in sorted(by_model, key=model_sort_key):
        ds = by_model[model]
        asdiv = ds.get("asdiv")
        mawps = ds.get("mawps")
        asdiv_s = f"{fmt(asdiv['identity_acc'])} ({asdiv['n']})" if asdiv else "—"
        mawps_s = f"{fmt(mawps['identity_acc'])} ({mawps['n']})" if mawps else "—"
        lines.append(f"| {model} | {asdiv_s} | {mawps_s} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _probe_table(rows, title, note):
    lines = [f"## {title}", "", note, "", "| model | raw τ=0 | chat/faithful τ=0 | Δ(chat−raw) τ=0 (pp) |", "|---|---|---|---|"]
    for d in sorted(rows, key=lambda d: model_sort_key(d["model"])):
        model = d["model"]
        c = d.get("conditions", {})
        raw_v = c.get("raw", {}).get("base_greedy_tau0")
        chat_v = c.get("chat", {}).get("base_greedy_tau0")
        delta = d.get("delta_chat_minus_raw", {}).get("tau0")
        delta_s = f"{delta * 100:+.1f}" if delta is not None else "—"
        lines.append(f"| {model} | {fmt(raw_v)} | {fmt(chat_v)} | {delta_s} |")
    lines.append("")
    return lines


def build_chat_template_probe(raw_dir: Path) -> str:
    generic_files = [f for f in load(raw_dir, "chat_template_probe_*.json") if "polarfaithful" not in f.name]
    faithful_files = load(raw_dir, "chat_template_probe_polarfaithful_*.json")
    generic = [json.loads(f.read_text()) for f in generic_files]
    faithful = [json.loads(f.read_text()) for f in faithful_files]

    lines = [
        "# Chat-template mechanism probe: raw vs. chat, content held fixed",
        "",
        "Source: `chat_template_probe.py`, 200-question pooled sample (seed 42), "
        "`dart_math_v2`. \"chat\" = chat template applied, no system message "
        "(generic probe) or + system message (PoLar-faithful probe). Raw "
        "JSONs: `raw/chat_template_probe*.json`.",
        "",
    ]
    lines += _probe_table(
        generic, "Generic (raw vs. chat, no system message)",
        "Mechanism-only ablation across all 6 models, content is `paper_minimal_fewshot` on both sides.",
    )
    lines += _probe_table(
        faithful, "PoLar-faithful (raw vs. chat + system message)",
        "Same ablation, but the \"chat\" condition adds PoLar's real system message "
        "for the models whose PoLar mechanism includes one. qwen25_3b's run here "
        "was superseded by the fuller PoLar-literal reproduction track (see "
        "literal_repro.md) but is kept for provenance.",
    )
    return "\n".join(lines) + "\n"


def build_literal_repro(raw_dir: Path) -> str:
    files = load(raw_dir, "literal_repro_*.json")
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        comparable = "NOTCOMPARABLE" not in f.name
        variant = "fewshot" if "fewshot" in f.name else "zeroshot"
        rows.append((d, variant, comparable, f.name))
    rows.sort(key=lambda t: (model_sort_key(t[0]["model"]), t[1]))

    lines = [
        "# PoLar-literal reproduction: exact chat-template mechanism, full data",
        "",
        "Source: `baseline_passk.py` re-run with each model's PoLar-exact "
        "mechanism (`chat` / `paper_chat_sys` = PoLar's literal zero-shot "
        "Appendix D.4 content; `paper_minimal_fewshot_chat` / "
        "`paper_minimal_fewshot_chat_sys` = same mechanism + our fewshot content), "
        "full `dart_math_v2` TEST (2500 q, all 5 difficulties). Raw JSONs: "
        "`raw/literal_repro_*.json`.",
        "",
        "One file (`qwen15_moe_a27b` zeroshot) is an early, **non-comparable** "
        "data point, produced on different hardware than the rest of this "
        "table, which matters because bf16 numerics are known to diverge "
        "non-associatively across GPU architectures (a documented general "
        "property of this project's setup, not specific to this one file); "
        "included for provenance, excluded from any cross-model comparison.",
        "",
    ]
    for d, variant, comparable, fname in rows:
        model = d["model"]
        env = d.get("env", {})
        pd = {r["difficulty"]: r for r in d["per_difficulty"]}
        tag = "" if comparable else ", **NOT COMPARABLE (different hardware)**"
        lines.append(f"## {model}, {variant}{tag}")
        lines.append("")
        lines.append(
            f"GPU: `{env.get('gpu_name', '?')}` &middot; prompt_style=`{env.get('prompt_style', '?')}` "
            f"&middot; source: `{fname}`"
        )
        lines.append("")
        lines.append("| metric | DM-1 | DM-2 | DM-3 | DM-4 | DM-5 |")
        lines.append("|---|---|---|---|---|---|")
        for mk in METRIC_KEYS:
            vals = []
            for diff in range(1, 6):
                r = pd.get(diff, {})
                if mk == "tau0":
                    v = r.get("base_greedy_p1")
                else:
                    v = r.get("base_sampling_pk_best_per_k", {}).get(mk[1:])
                vals.append(v)
            lines.append(f"| {METRIC_LABELS[mk]} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "dart_math.md": build_dart_math,
        "mmlu_pro.md": build_mmlu_pro,
        "ood.md": build_ood,
        "chat_template_probe.md": build_chat_template_probe,
        "literal_repro.md": build_literal_repro,
    }
    n_files_total = 0
    for out_name, builder in builders.items():
        content = builder(raw_dir)
        (out_dir / out_name).write_text(content)
        n_files_total += 1
        print(f"wrote {out_dir / out_name} ({len(content)} bytes)")

    print(f"Aggregation complete: {n_files_total} markdown tables written to {out_dir}")


if __name__ == "__main__":
    main()
