"""Unit tests for the OOM budget math of the reward-fn replica pool.

Only `safe_replica_count` / the estimators are exercised, they are pure
arithmetic and torch-free. The actual model-building (`build_*_replicas`)
needs a GPU and real loaded models, so it's validated by a real run, not
unit-tested here.
"""
from re_polar.mcts.replica_pool import (
    estimate_activation_bytes,
    safe_replica_count,
)

GB = 1024 ** 3


def test_safe_replica_count_respects_budget():
    # 0.9*80GB = 72GB budget; per replica = 16+4 = 20GB -> 72//20 = 3
    assert safe_replica_count(80 * GB, 16 * GB, 4 * GB, max_replicas=8, safety=0.9) == 3


def test_safe_replica_count_floor_is_one():
    # not even one extra fits under budget -> still 1 (a single replica == today's job)
    assert safe_replica_count(1 * GB, 16 * GB, 4 * GB, max_replicas=8) == 1


def test_safe_replica_count_caps_at_max():
    assert safe_replica_count(1000 * GB, 1 * GB, 0, max_replicas=4) == 4


def test_safe_replica_count_degenerate_inputs():
    assert safe_replica_count(0, 16 * GB, 4 * GB, max_replicas=8) == 1
    assert safe_replica_count(80 * GB, 0, 0, max_replicas=8) == 1


def test_safe_replica_count_never_exceeds_budget():
    # brute check: chosen N * per must stay under the safety budget for a range
    for free in (24, 40, 48, 80, 141):
        for model in (8, 16, 32):
            for act in (2, 4, 8):
                n = safe_replica_count(free * GB, model * GB, act * GB, max_replicas=16, safety=0.9)
                assert n >= 1
                if n > 1:  # floor-1 may legitimately exceed budget on tiny GPUs
                    assert n * (model + act) * GB <= 0.9 * free * GB + 1


def test_estimate_activation_bytes_scales_with_batch():
    a1 = estimate_activation_bytes(num_layers=36, hidden_size=4096, batch_size=1, seq_len=256)
    a4 = estimate_activation_bytes(num_layers=36, hidden_size=4096, batch_size=4, seq_len=256)
    assert a4 == 4 * a1 and a1 > 0
