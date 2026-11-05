"""analysis.mcts_design.menu_coverage helpers on hand-made inputs (no model needed)."""

import json

from analysis.mcts_design.menu_coverage import build_candidate_pool, grid_labels, known_labels

SAMPLES = [
    {
        "sample_info": {"query_id": "a"},
        "final_valid_transitions": [[0, 1]],
        "final_invalid_transitions": [[1]],
    },
    {
        "sample_info": {"query_id": "b"},
        "final_valid_transitions": [],
        "final_invalid_transitions": [[0, 1]],
    },
    {
        "sample_info": {"query_id": "c"},
        "final_valid_transitions": [[1]],
        "final_invalid_transitions": [],
    },
]


def test_known_labels_only_on_the_given_questions():
    k = known_labels(SAMPLES, {"a", "b"}, [(0, 1), (1,)])
    assert k == {(0, 1): {"a": 1.0, "b": 0.0}, (1,): {"a": 0.0}}  # "c" is outside the question set


def test_ranking_uses_only_the_rank_pool():
    # on a+b only (0,1) was ever valid; (1,) only solved c
    assert [p for p, _ in build_candidate_pool(SAMPLES, {"a", "b"}, 10)] == [(0, 1)]
    assert [p for p, _ in build_candidate_pool(SAMPLES, {"a", "b", "c"}, 10)] == [(0, 1), (1,)]


def test_grid_labels(tmp_path):
    g = tmp_path / "grid.json"
    g.write_text(
        json.dumps(
            {
                "per_difficulty": [
                    {
                        "difficulty": 2,
                        "qids": ["a", "b"],
                        "per_program": [{"path": [0, 1], "rewards": [0, 1], "provenance": "fi"}],
                    }
                ]
            }
        )
    )
    assert grid_labels(g, 2) == {(0, 1): {"a": 0.0, "b": 1.0}}
    assert grid_labels(g, 2, drop="i") == {(0, 1): {"a": 0.0}}


def test_menu_stats_identity_and_pass_at_k():
    from analysis.mcts_design.menu_heldout import menu_stats

    grid = {
        "per_difficulty": [
            {
                "qids": ["a", "b", "c", "d"],
                "per_program": [
                    {"path": [0, 1, 2], "rewards": [1, 0, 0, 0]},  # identity, first in the menu
                    {"path": [0, 2], "rewards": [1, 1, 0, 0]},
                    {"path": [0, 1, 1, 2], "rewards": [0, 0, 1, 0]},
                ],
            }
        ]
    }
    s = menu_stats(grid, num_layers=3, ks=(1, 2, 3))
    assert s["identity"] == 0.25 and s["n"] == 4
    assert s["pass_at"] == {1: 0.25, 2: 0.5, 3: 0.75}
