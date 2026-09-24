"""Exact McNemar p-values and accuracy/has_boxed/echo deltas for the echo-sweep
data, supporting the significance-pattern prose in the paper's appendix
(the placeholder-echo section). generate_echo_sweep_latex_table.py only
emits significance markers (*/***/blank) into the table itself; this script
prints the exact p-values and deltas that prose paragraph needs, reusing the
same tested load_records/mcnemar_p/cell_stats/data_filename functions so the
numbers can't drift from the table.

    python -m analysis.error_analysis.echo_sweep_significance_report \\
        --data-dir full10k_for_gen
"""

import argparse
import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "generate_echo_sweep_latex_table.py"
_spec = importlib.util.spec_from_file_location("generate_echo_sweep_latex_table", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)

    for model_label, model_slug in _mod.MODEL_ORDER:
        rows_by_variant = {}
        for _, variant_slug, _ in _mod.VARIANT_ORDER:
            fp = data_dir / _mod.data_filename(model_slug, variant_slug)
            rows_by_variant[variant_slug] = _mod.load_records(fp)
        stats = {v: _mod.cell_stats(rows_by_variant[v]) for v in rows_by_variant}
        default_rows = rows_by_variant["paper_default"]
        default_acc = stats["paper_default"]["acc"]

        print(f"\n=== {model_label} ===")
        print(
            f"  paper_default: acc={default_acc:.1%} has_boxed={stats['paper_default']['has_boxed_rate']:.1%} "
            f"echo={stats['paper_default']['echo_rate']:.2%}"
        )
        for _, variant_slug, _ in _mod.VARIANT_ORDER:
            if variant_slug == "paper_default":
                continue
            p = _mod.mcnemar_p(default_rows, rows_by_variant[variant_slug])
            acc = stats[variant_slug]["acc"]
            print(
                f"  {variant_slug:38s} acc={acc:.1%} (delta={acc - default_acc:+.1%}) "
                f"has_boxed={stats[variant_slug]['has_boxed_rate']:.1%} "
                f"echo={stats[variant_slug]['echo_rate']:.2%}  p={p:.3g}"
            )

        # head-to-head: polar_oneshot vs drllm_chat_prefill (the two "interventions"
        # compared directly in the paper's prose)
        p_h2h = _mod.mcnemar_p(
            rows_by_variant["paper_minimal_fewshot"], rows_by_variant["drllm_chat_prefill"]
        )
        acc_oneshot = stats["paper_minimal_fewshot"]["acc"]
        acc_prefill = stats["drllm_chat_prefill"]["acc"]
        print(
            f"  [head-to-head] polar_oneshot vs drllm_chat_prefill: "
            f"delta={acc_oneshot - acc_prefill:+.1%} p={p_h2h:.3g}"
        )


if __name__ == "__main__":
    main()
