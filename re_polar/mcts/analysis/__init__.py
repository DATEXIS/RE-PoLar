"""Post-hoc analysis of MCTS supervision data (`merged_mcts_samples.json`).

Modules:
  segments.py   Program/segment math (decode, edit_signature, entropy/gini/
                topk-coverage, op_class, shortest_valid, layer_freq) --
                pure, layer-count-parameterized.
  registry.py   `@insight(id, title, chart_kind)` decorator + lookup -- the
                "select what insights I want" mechanism.
  insights.py   The registered insights themselves (menu_concentration,
                op_mix, segmentation_structure, rescue_breakdown,
                op_class_coverage, menu_size_distribution, layer_ops_heatmap,
                shorter_than_identity, dismemberment), each
                `fn(groups, num_layers) -> dict`.
  loader.py     Loads merged_mcts_samples.json sources (reuses
                re_polar.datasets.schemas.load_samples) and groups them by a
                sample_info key (difficulty/domain/category/...) or by
                source.
  config.py     JSON report-config schema (`ReportConfig`/`load_config`) --
                which model/sources/insights a given run covers.
  polar_comparison.py  Checks against PoLar's own diagnostic claims
                (compute_skip_loop_accuracy, compute_accuracy_by_depth_
                budget, compute_mean_executed_depth, compute_valid_
                coverage_by_recurrence_budget, compute_recurrence_and_
                skip_requirement, compute_accuracy_by_executed_depth,
                compute_segment_length_distribution, compute_segment_
                recurrence_distribution, compute_findings), same
                `(groups, num_layers) -> dict` shape as every insight.

Usage: load samples with `loader.py`, group them, then call any insight or
polar_comparison function directly, e.g.
`insights.menu_concentration(groups, num_layers)` or
`polar_comparison.compute_skip_loop_accuracy(groups, num_layers)`.
"""
