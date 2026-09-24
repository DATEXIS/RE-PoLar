"""Generate the paper's echo-sweep LaTeX table from raw data, instead of
hand-typing numbers into LaTeX -- generating the table (and its McNemar
significance markers) directly from the source files removes the
transcription step where that kind of error creeps in (an earlier
hand-typed version of this table carried a real error for a while: the
prose claimed both intervention variants beat the default prompt on every
model, when the raw data actually shows paper_default is LLaMA-3.2-3B's
single best-accuracy variant).

Reads all 30 full10k_*.jsonl files (5 models x 6 prompt variants, 10,000
samples each -- one `prompt_variant_pilot.py --full` run per cell),
computes accuracy/has_boxed/echo per cell plus McNemar's paired test of
each non-default row against that model's own paper_default row (same
10,000 queries under both prompts), and emits the exact `tabular` LaTeX
block used in the paper appendix.

Pure functions (cell_stats, mcnemar_p, significance_marker, format table
rows) are unit-testable without needing the real data files. main() does
the real file I/O.

    python -m analysis.error_analysis.generate_echo_sweep_latex_table \\
        --data-dir full10k_for_gen \\
        --output generated_echo_sweep_table.tex
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Paper display order.
MODEL_ORDER = [
    ("Qwen1.5-MoE-A2.7B", "qwen15_moe_a27b"),
    ("Qwen2.5-3B", "qwen25_3b"),
    ("Qwen2.5-7B", "qwen25_7b"),
    ("LLaMA-3.2-3B", "llama32_3b"),
    ("Qwen3-8B", "qwen3_8b"),
    ("Qwen3-32B", "qwen3_32b"),
]
# (paper display name, filename-slug, tex-name)
# The filename-slug (2nd element, what's actually on disk / in prompt_variant_pilot.py's
# PROTOCOLS dict) differs from the paper's own display name for two rows
# ("paper_default"/"paper_minimal_fewshot" print as "polar_default"/"polar_oneshot" --
# "oneshot" is more precise too, it's genuinely one demonstration) -- this only
# affects what LaTeX name gets printed, not what's read from disk.
VARIANT_ORDER = [
    ("polar_default", "paper_default", "polar\\_default"),
    ("drllm (faithful)", "drllm_faithful", "drllm} (faithful"),  # special-cased in format_row
    ("drllm_chat_prefill", "drllm_chat_prefill", "drllm\\_chat\\_prefill"),
    ("polar_oneshot", "paper_minimal_fewshot", "polar\\_oneshot"),
    ("drllm_chat_oneshot", "drllm_chat_minimal_fewshot", "drllm\\_chat\\_oneshot"),
    (
        "drllm_chat_oneshot_prefill",
        "drllm_chat_minimal_fewshot_prefill",
        "drllm\\_chat\\_oneshot\\_prefill",
    ),
]

# Which grader each variant is scored with (prompt_variant_pilot.py's PROTOCOLS
# dict, `grader` field -- "ours" only for the two polar_*-wording variants,
# "mathruler" for every drllm* variant, so the table can show how each cell was
# actually evaluated).
GRADER_BY_VARIANT = {
    "paper_default": "PoLar",
    "drllm_faithful": "mathruler",
    "drllm_chat_prefill": "mathruler",
    "paper_minimal_fewshot": "PoLar",
    "drllm_chat_minimal_fewshot": "mathruler",
    "drllm_chat_minimal_fewshot_prefill": "mathruler",
}


def data_filename(model_slug: str, variant_slug: str) -> str:
    return f"full10k_{variant_slug}_{model_slug}.jsonl"


def load_records(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_stats(rows: List[dict]) -> Dict[str, float]:
    """{n, acc, has_boxed_rate, unboxed_rate, echo_rate} -- pure aggregation, no I/O.

    unboxed_rate is has_boxed_rate's complement (share of generations with NO
    \\boxed{} span at all, i.e. the budget-cutoff failure mode) -- the table
    column is labeled/errtag'd \\errtag{unboxed} and must report the unboxed
    share, not has_boxed_rate itself. has_boxed_rate is kept alongside rather
    than dropped: echo_sweep_significance_report.py's own prose-support output
    still reads it directly."""
    n = len(rows)
    if n == 0:
        raise ValueError("cell_stats called on empty rows")
    has_boxed_rate = sum(bool(r["has_boxed"]) for r in rows) / n
    return {
        "n": n,
        "acc": sum(bool(r["correct"]) for r in rows) / n,
        "has_boxed_rate": has_boxed_rate,
        "unboxed_rate": 1.0 - has_boxed_rate,
        "echo_rate": sum(bool(r["is_placeholder_echo"]) for r in rows) / n,
    }


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mcnemar_p(rows_a: List[dict], rows_b: List[dict]) -> float:
    """Paired McNemar's test p-value, matched by query_id. `rows_a`/`rows_b`
    must cover the exact same set of query_ids (same underlying samples,
    two different prompts) -- raises if they don't, rather than silently
    computing a wrong answer over a partial/misaligned join."""
    ca = {r["query_id"]: bool(r["correct"]) for r in rows_a}
    cb = {r["query_id"]: bool(r["correct"]) for r in rows_b}
    if set(ca) != set(cb):
        raise ValueError(
            f"mcnemar_p: query_id sets differ ({len(ca)} vs {len(cb)}) -- not a valid pairing"
        )
    b = sum(1 for q in ca if ca[q] and not cb[q])
    c = sum(1 for q in ca if not ca[q] and cb[q])
    if b + c == 0:
        return 1.0
    z = (c - b) / math.sqrt(b + c)
    return 2 * (1 - _phi(abs(z)))


def significance_marker(p: Optional[float]) -> str:
    """LaTeX superscript for a McNemar p-value against the model's own default
    row. None (the default row itself, nothing to compare) -> no marker."""
    if p is None:
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def format_row(
    variant_tex: str, stats: Dict[str, float], sig: str, is_best: bool, grader: str
) -> str:
    acc_str = f"{stats['acc'] * 100:.1f}\\%"
    if is_best:
        acc_str = f"\\textbf{{{acc_str}}}"
    return (
        f" & \\texttt{{{variant_tex}}} & {grader} & {acc_str}{sig} & "
        f"{stats['unboxed_rate'] * 100:.1f}\\% & {stats['echo_rate'] * 100:.2f}\\% \\\\"
    )


def generate_table_tex(data_dir: Path) -> str:
    """Reads all 30 full10k_*.jsonl files under `data_dir` and returns the
    complete `table` environment (caption/label/tabular) as a LaTeX string."""
    n_models = len(MODEL_ORDER)
    n_variants = len(VARIANT_ORDER)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{Accuracy, \errtag{{unboxed}} rate, and \errtag{{prompt template echo}} rate, all "
        rf"${n_models}\times{n_variants}={n_models * n_variants}$ "
        r"model/prompt-variant combinations, on our $10{,}000$-query DART-Math split "
        r"(\cref{sec:appendix_dart_math}). \textbf{Bold} = best accuracy for that model. \textbf{Grader}: "
        r"\texttt{polar\_default}/\texttt{polar\_oneshot} use \texttt{PoLar}, \citet{li2026polar}'s own "
        r"last-\texttt{\textbackslash boxed\{\}}-match DART-Math extraction (byte-identical to their vendored "
        r"copy); the four \texttt{drllm} variants use \texttt{mathruler}, \citeauthor{heakl2025drllm}'s own "
        r"grading library \citep{heakl2025drllm} -- each graded the way its own source protocol would grade it. "
        r"\texttt{polar\_oneshot} repeats the instruction-and-answer block twice inside the 50-token budget, "
        r"risking a hallucinated third, incomplete \texttt{\textbackslash boxed\{...\}} span that a naive "
        r"last-match extraction would grab instead of the real answer; we truncate before grading at the first "
        r"complete span, so it is, in effect, graded on the \emph{first} \texttt{\textbackslash boxed\{...\}} "
        r"span rather than the last (\texttt{polar\_default}'s single, unrepeated block never needs this). "
        r"Superscripts "
        r"are McNemar's paired test against that model's own \texttt{polar\_default} row: $^{***}$ $p<0.001$, "
        r"$^{*}$ $p<0.05$, unmarked = not significant. \Cref{tab:prompt-templates} gives the exact wording of "
        r"every variant swept here. Generated by "
        r"\texttt{paper\_results/error\_analysis/generate\_echo\_sweep\_latex\_table.py} -- do not hand-edit.}",
        r"\label{tab:echo-sweep}",
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Model & Variant & Grader & Accuracy & \errtag{unboxed} & \errtag{prompt template echo} \\",
        r"\midrule",
    ]
    for model_label, model_slug in MODEL_ORDER:
        rows_by_variant: Dict[str, List[dict]] = {}
        for _, variant_slug, _ in VARIANT_ORDER:
            fp = data_dir / data_filename(model_slug, variant_slug)
            rows_by_variant[variant_slug] = load_records(fp)

        stats_by_variant = {v: cell_stats(rows_by_variant[v]) for v in rows_by_variant}
        best_variant = max(stats_by_variant, key=lambda v: stats_by_variant[v]["acc"])
        default_rows = rows_by_variant["paper_default"]

        lines.append(f"\\multirow{{{len(VARIANT_ORDER)}}}{{*}}{{{model_label}}}")
        for _, variant_slug, variant_tex in VARIANT_ORDER:
            stats = stats_by_variant[variant_slug]
            grader = f"\\texttt{{{GRADER_BY_VARIANT[variant_slug]}}}"
            if variant_slug == "paper_default":
                sig = ""
                variant_tex_final = "polar\\_default"
            else:
                sig = significance_marker(mcnemar_p(default_rows, rows_by_variant[variant_slug]))
                variant_tex_final = (
                    variant_tex if variant_slug != "drllm_faithful" else "drllm} (faithful)"
                )
            is_best = variant_slug == best_variant
            if variant_slug == "drllm_faithful":
                # special-cased tex (closing brace mid-token for the "(faithful)" annotation)
                acc_str = f"{stats['acc'] * 100:.1f}\\%"
                if is_best:
                    acc_str = f"\\textbf{{{acc_str}}}"
                lines.append(
                    f" & \\texttt{{drllm}} (faithful) & {grader} & {acc_str}{sig} & "
                    f"{stats['unboxed_rate'] * 100:.1f}\\% & {stats['echo_rate'] * 100:.2f}\\% \\\\"
                )
            else:
                lines.append(format_row(variant_tex_final, stats, sig, is_best, grader))
        lines.append(
            r"\midrule" if (model_label, model_slug) != MODEL_ORDER[-1] else r"\bottomrule"
        )
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", required=True, help="dir containing the 30 full10k_*.jsonl files"
    )
    parser.add_argument("--output", required=True, help="where to write the generated .tex snippet")
    args = parser.parse_args(argv)

    tex = generate_table_tex(Path(args.data_dir))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex + "\n")
    print(f"Wrote {len(tex.splitlines())} lines -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
