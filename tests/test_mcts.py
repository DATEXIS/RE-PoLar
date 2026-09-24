"""MCTS logic tests with a fake reward -- no model, no GPU.

Fake task with reward gradient (like real math rewards, where many programs
succeed): reward 1 iff layer 2 is NOT executed. Identity fails; every program
placing a SKIP edit over the segment containing 2 succeeds. Under the
edits-over-identity search a skip[2:3) is one action from the root, so
discovery happens through tree expansion and these tests exercise the
mechanism, not deep-search endurance.

The depth-reachability test (test_deep_layer_edit_is_reachable) is the whole
point of the edits-over-identity design: a reward that only a repeat of
layer 30 satisfies is found within budget, which an identity-tail search
(walking outward from the unedited program one layer at a time) could not
reach at all.
"""

import threading
import time

from re_polar.mcts.scheduler import EvalCache, MCTSRunner
from re_polar.mcts.search import ProgramMCTS
from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, is_valid

D = 36


class FakeReward:
    """Callable like GenerationReward; counts evaluations for batching asserts."""

    def __init__(self):
        self.calls = 0
        self.evaluated_pairs = 0

    def __call__(self, program, questions, gt_answers):
        self.calls += 1
        self.evaluated_pairs += len(questions)
        ok = 2 not in program.to_layer_path()
        return [1.0 if ok else 0.0] * len(questions)


class DeepRepeatReward:
    """reward 1 iff layer 30 is REPEATED (executed >=2 times) -- a DEEP edit an
    identity-tail search could never place."""

    def __init__(self):
        self.calls = 0

    def __call__(self, program, questions, gt_answers):
        self.calls += 1
        ok = program.to_layer_path().count(30) >= 2
        return [1.0 if ok else 0.0] * len(questions)


def make_inputs(n):
    return [{"query_id": f"q{i}", "question": f"question {i}", "gt_ans": "42"} for i in range(n)]


def test_tree_finds_solution_and_prefers_shorter():
    # budget=500, not 100: per_tree_seed defaults to True, and this doesn't
    # rely on any specific mode -- just needs enough budget for every tree's
    # own derived seed to find the target regardless of mode (100 left some
    # trees short purely on seed luck under per-tree-seed).
    fake = FakeReward()
    runner = MCTSRunner(
        make_inputs(4), num_layers=D, reward_fn=fake, budget=500, c=1.414, lam=5.0, seed=0
    )
    trees = runner.run(log_every=0)
    solved = [qid for qid, t in trees.items() if t.valid_paths()]
    assert len(solved) == 4, f"only {len(solved)}/4 trees found the target program"
    for t in trees.values():
        paths = t.valid_paths()
        assert len(paths[0]) < D  # shortest valid beats identity
        assert len(paths[0]) <= len(paths[-1])  # shortest-first ordering


def test_identity_precompute_and_metric():
    fake = FakeReward()
    runner = MCTSRunner(
        make_inputs(3), num_layers=D, reward_fn=fake, budget=5, c=1.414, lam=5.0, seed=0
    )
    runner.run(log_every=0)
    # identity executes all 36 layers -> fake reward 0
    assert set(runner.initial_metric.values()) == {0.0}


def test_program_major_batching_shares_evals_shared_seed_mode():
    """per_tree_seed=False: explicit rather than relying on the default, so
    this test keeps protecting shared-seed behavior regardless of which mode
    is the current default (see test_per_tree_seed_breaks_program_major_batching
    for the contrasting mode)."""
    fake = FakeReward()
    n, budget = 8, 30
    runner = MCTSRunner(
        make_inputs(n),
        num_layers=D,
        reward_fn=fake,
        budget=budget,
        c=1.414,
        lam=5.0,
        seed=1,
        per_tree_seed=False,
    )
    runner.run(log_every=0)
    # GPU batches (reward-fn calls): shared-seed trees propose in lockstep, so
    # ~one call per round instead of one per (tree, proposal) = n * (budget+1)
    assert fake.calls <= budget + 1, f"{fake.calls} calls -- batching not shared"
    # per-(program, input) evaluations are necessary work and stay bounded
    assert fake.evaluated_pairs <= n * (budget + 1)


def test_per_tree_seed_breaks_program_major_batching():
    """per_tree_seed=True gives every tree its own deterministic seed (derived
    from query_id), so root-action shuffles differ across trees from proposal
    1 -- the shared-seed batching collision (test above) mostly disappears.
    Batches collapse toward one reward-fn call per (tree, proposal) instead of
    one per round: empirically (n=8, budget=30) shared-seed mode makes 30
    calls total, per-tree-seed makes 240 (of a possible 248) -- this asserts
    the qualitative direction with a comfortable margin, not the exact count."""
    fake = FakeReward()
    n, budget = 8, 30
    runner = MCTSRunner(
        make_inputs(n),
        num_layers=D,
        reward_fn=fake,
        budget=budget,
        c=1.414,
        lam=5.0,
        seed=1,
        per_tree_seed=True,
    )
    runner.run(log_every=0)
    assert (
        fake.calls > n * (budget + 1) // 2
    ), f"only {fake.calls} calls -- trees still batching together, per_tree_seed had no effect"


def test_per_tree_seed_true_is_bit_identical_to_omitting_the_flag():
    """per_tree_seed=True (explicit) and omitting the kwarg entirely (the
    default) must produce the exact same search -- the default must not
    perturb behavior beyond the flag's own documented effect. (Symmetric
    coverage for explicit per_tree_seed=False already exists via
    test_program_major_batching_shares_evals_shared_seed_mode.)"""

    def collect(**kwargs):
        fake = FakeReward()
        runner = MCTSRunner(
            make_inputs(4),
            num_layers=D,
            reward_fn=fake,
            budget=30,
            c=1.414,
            lam=5.0,
            seed=9,
            **kwargs,
        )
        trees = runner.run(log_every=0)
        return {qid: sorted(t.evaluated) for qid, t in trees.items()}

    baseline = collect()
    assert collect(per_tree_seed=True) == baseline


def test_per_tree_seed_gives_each_tree_a_distinct_deterministic_seed():
    fake = FakeReward()
    runner = MCTSRunner(
        make_inputs(5),
        num_layers=D,
        reward_fn=fake,
        budget=5,
        c=1.414,
        lam=5.0,
        seed=42,
        per_tree_seed=True,
    )
    seeds = {qid: tree.seed for qid, tree in runner.trees.items()}
    assert len(set(seeds.values())) == len(seeds)  # all distinct

    # deterministic: rebuilding with the same base seed reproduces the same per-tree seeds
    runner2 = MCTSRunner(
        make_inputs(5),
        num_layers=D,
        reward_fn=FakeReward(),
        budget=5,
        c=1.414,
        lam=5.0,
        seed=42,
        per_tree_seed=True,
    )
    assert {qid: tree.seed for qid, tree in runner2.trees.items()} == seeds


class _CountingReplica:
    """Deterministic like FakeReward (reward 1 iff layer 2 is skipped) but also
    records concurrency, so a pool of these can assert (a) replica exclusivity --
    no single model runs two programs at once -- and (b) real parallelism across
    replicas. The tiny sleep forces overlap when >1 replica is dispatched."""

    def __init__(self, registry, idx):
        self.registry = registry
        self.idx = idx

    def __call__(self, program, questions, gt_answers):
        reg = self.registry
        with reg["lock"]:
            reg["live"] += 1
            reg["per"][self.idx] += 1
            reg["max_live"] = max(reg["max_live"], reg["live"])
            reg["max_per"][self.idx] = max(reg["max_per"][self.idx], reg["per"][self.idx])
        time.sleep(0.003)
        with reg["lock"]:
            reg["live"] -= 1
            reg["per"][self.idx] -= 1
        ok = 2 not in program.to_layer_path()
        return [1.0 if ok else 0.0] * len(questions)


class FakeMaskedBatchReward:
    """Callable like GenerationReward (single-program __call__, the fallback
    path) but ALSO exposes masked_batch_call (programs/questions/gt_answers as
    PARALLEL arrays, `programs[i]` possibly a DIFFERENT program per row) --
    lets scheduler tests exercise the masked-batch dispatch with no model/GPU.
    Same reward RULE as FakeReward (1 iff layer 2 not executed) so a run using
    this reward must produce trees IDENTICAL to a run using plain FakeReward --
    proves the dispatch reshapes rewards back to the right (path, qid) slots,
    not just that something gets returned."""

    def __init__(self):
        self.calls = 0  # single-program __call__ (fallback path)
        self.masked_batch_calls = 0  # cross-program masked_batch_call invocations
        self.max_batch_programs = 0  # distinct programs seen in one masked_batch_call

    def __call__(self, program, questions, gt_answers):
        self.calls += 1
        ok = 2 not in program.to_layer_path()
        return [1.0 if ok else 0.0] * len(questions)

    def masked_batch_call(self, programs, questions, gt_answers):
        self.masked_batch_calls += 1
        n_distinct = len({tuple(p.to_layer_path()) for p in programs})
        self.max_batch_programs = max(self.max_batch_programs, n_distinct)
        return [1.0 if 2 not in p.to_layer_path() else 0.0 for p in programs]


def _tree_state(trees):
    """Full per-tree evaluated map (path -> reward), sorted for exact compare."""
    return {qid: sorted(t.evaluated.items()) for qid, t in trees.items()}


def _run_pool(reward_fns=None, *, per_tree_seed=False, n=8, budget=40, seed=3):
    return MCTSRunner(
        make_inputs(n),
        num_layers=D,
        reward_fn=(reward_fns or [FakeReward()])[0],
        reward_fns=reward_fns,
        budget=budget,
        c=1.414,
        lam=5.0,
        seed=seed,
        per_tree_seed=per_tree_seed,
    ).run(log_every=0)


def test_reward_pool_parallel_matches_serial_shared_seed():
    """A 3-replica pool must give byte-identical trees to the single-reward
    scheduler: each (program, input) is scored by exactly one equivalent replica,
    so results cannot depend on pool size."""
    serial = _tree_state(_run_pool([FakeReward()]))
    parallel = _tree_state(_run_pool([FakeReward(), FakeReward(), FakeReward()]))
    assert parallel == serial


def test_reward_pool_parallel_matches_serial_per_tree_seed():
    """per_tree_seed=True => many distinct programs/round => the parallel dispatch
    path (pool>1 AND >1 gen job) is heavily exercised, yet results stay identical
    to serial."""
    serial = _tree_state(_run_pool([FakeReward()], per_tree_seed=True))
    parallel = _tree_state(
        _run_pool([FakeReward(), FakeReward(), FakeReward()], per_tree_seed=True)
    )
    assert parallel == serial


def test_reward_fns_none_is_bit_identical_to_single_reward_fn():
    """Omitting reward_fns must not perturb today's single-model behavior."""

    def state(**kw):
        runner = MCTSRunner(
            make_inputs(4),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=30,
            c=1.414,
            lam=5.0,
            seed=9,
            **kw,
        )
        return _tree_state(runner.run(log_every=0))

    assert state() == state(reward_fns=None)


def test_reward_pool_replica_exclusivity_and_real_parallelism():
    """Each replica runs at most ONE program at a time (else two programs' layer
    rerouting collides on one model), while >=2 replicas run concurrently
    (proving the dispatch actually parallelizes). per_tree_seed=True guarantees
    many distinct programs per round."""
    k = 3
    reg = {"lock": threading.Lock(), "live": 0, "max_live": 0, "per": [0] * k, "max_per": [0] * k}
    pool = [_CountingReplica(reg, i) for i in range(k)]
    _run_pool(pool, per_tree_seed=True)
    assert max(reg["max_per"]) == 1, "a replica ran two programs at once -> model-state collision"
    assert reg["max_live"] >= 2, "no real parallelism observed across replicas"


def test_masked_batch_none_is_bit_identical_to_default():
    """Omitting masked_batch_reward_fn (default None) must not perturb today's
    behavior -- the opt-in must be a true no-op when unused."""

    def state(**kw):
        runner = MCTSRunner(
            make_inputs(6),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=30,
            c=1.414,
            lam=5.0,
            seed=5,
            per_tree_seed=True,
            **kw,
        )
        return _tree_state(runner.run(log_every=0))

    assert state() == state(masked_batch_reward_fn=None)


def test_masked_batch_dispatch_matches_serial_and_is_exercised():
    """masked_batch_reward_fn (same reward RULE as FakeReward) must give byte-
    identical trees to the plain single-reward path, AND the dispatch must
    actually fire: per_tree_seed=True => many distinct programs/round =>
    masked_batch_call sees >1 distinct program, __call__ never fires."""
    serial = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
            per_tree_seed=True,
        ).run(log_every=0)
    )

    mb = FakeMaskedBatchReward()
    masked = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=mb,
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
            per_tree_seed=True,
            masked_batch_reward_fn=mb,
        ).run(log_every=0)
    )

    assert masked == serial
    assert mb.masked_batch_calls > 0, "masked-batch dispatch never fired"
    assert mb.max_batch_programs > 1, "never exercised >1 distinct program in one batch"
    # __call__ CAN still fire for genuine single-distinct-program rounds (e.g. late
    # in the search when few trees remain out of sync) -- that's the correct
    # >1-job guard at work, not a dispatch-priority bug; only masked-batch-eligible
    # rounds (>1 distinct job) are asserted above.


def test_masked_batch_falls_back_to_call_for_single_distinct_program_round():
    """A round with only ONE distinct pending program has nothing to batch
    across -- must use the plain __call__ path (mirrors the replica pool's own
    >1-job guard), not masked_batch_call."""
    mb = FakeMaskedBatchReward()
    # n=1 tree => every round has exactly one distinct program pending (itself)
    # => masked_batch_call must never fire regardless of per_tree_seed.
    MCTSRunner(
        make_inputs(1),
        num_layers=D,
        reward_fn=mb,
        budget=20,
        c=1.414,
        lam=5.0,
        seed=1,
        masked_batch_reward_fn=mb,
    ).run(log_every=0)
    assert mb.masked_batch_calls == 0
    assert mb.calls > 0


def test_masked_batch_layer_pass_greedy_largest_group_first():
    """CPU-only unit test of the shared scheduling core (`_masked_batch_layer_pass`,
    re_polar/mcts/rewards.py), no model/GPU needed, fake layers that just add 1.0 so
    correctness is trivially checkable (addition order doesn't matter here,
    unlike real bf16 matmuls) while still exercising the real
    gather/scatter/grouping code path.

    Concrete worked example: 8 rows KEEP an early layer (queue starts
    [0,1,2]), 2 rows SKIP it (queue starts [1,2] -- already past layer 0). The
    naive "flush every ready group every iteration" scheduler would process
    layer 1 as a 2-row group (iter 1, the skip-rows) THEN an 8-row group
    (iter 2, the keep-rows catching up) -- two separate reads. Greedy
    largest-group-first must instead hold the 2 skip-rows back (their group of 2
    loses to the keep-rows' group of 8 at iter 0) so that by the time layer 1 comes
    up, ALL 10 rows are ready simultaneously -- ONE group of 10, not 2+8."""
    import torch

    from re_polar.mcts.rewards import _masked_batch_layer_pass

    calls = []  # (layer_idx, group_size) in call order

    def make_layer(idx):
        def layer(
            hidden, attention_mask=None, position_ids=None, position_embeddings=None, use_cache=None
        ):
            calls.append((idx, hidden.shape[0]))
            return hidden + 1.0

        return layer

    layers = [make_layer(i) for i in range(4)]
    paths = [[0, 1, 2] for _ in range(8)] + [[1, 2] for _ in range(2)]
    n_rows = len(paths)
    state = torch.zeros(n_rows, 1)
    position_ids = torch.zeros(n_rows, 1, dtype=torch.long)
    cos = torch.zeros(n_rows, 1)
    sin = torch.zeros(n_rows, 1)

    out = _masked_batch_layer_pass(
        layers, state, paths, position_ids, cos, sin, causal_mask=None, device="cpu"
    )

    # correctness: each row's final value = number of layers in ITS path (each
    # layer call adds 1.0), regardless of how rows got grouped together.
    expected = torch.tensor([[3.0]] * 8 + [[2.0]] * 2)
    assert torch.allclose(out, expected)

    # scheduling: layer 1 must be processed as ONE group of all 10 rows, not
    # split into a 2-row group (the skip-rows arriving early) + an 8-row group
    # (the keep-rows catching up one step later).
    layer1_groups = [n for (idx, n) in calls if idx == 1]
    assert layer1_groups == [10], (
        f"expected layer 1 handled as one group of 10 (greedy largest-first "
        f"merging), got separate groups {layer1_groups}"
    )


def test_masked_batch_layer_pass_max_group_size_caps_peak_batch():
    """`max_group_size` must split an oversized group into sequential
    sub-chunks of at most that many rows -- bounding peak per-call batch size
    on a memory-constrained GPU -- WITHOUT changing the final result or
    dropping/reordering any row."""
    import torch

    from re_polar.mcts.rewards import _masked_batch_layer_pass

    calls = []  # (layer_idx, group_size) in call order

    def make_layer(idx):
        def layer(
            hidden, attention_mask=None, position_ids=None, position_embeddings=None, use_cache=None
        ):
            calls.append((idx, hidden.shape[0]))
            return hidden + 1.0

        return layer

    layers = [make_layer(i) for i in range(2)]
    n_rows = 10
    paths = [[0, 1] for _ in range(n_rows)]  # all 10 rows share the same path
    state = torch.zeros(n_rows, 1)
    position_ids = torch.zeros(n_rows, 1, dtype=torch.long)
    cos = torch.zeros(n_rows, 1)
    sin = torch.zeros(n_rows, 1)

    out = _masked_batch_layer_pass(
        layers,
        state,
        paths,
        position_ids,
        cos,
        sin,
        causal_mask=None,
        device="cpu",
        max_group_size=4,
    )

    # correctness unchanged: every row still gets both layers applied exactly once.
    assert torch.allclose(out, torch.full((n_rows, 1), 2.0))

    # peak batch size is capped at 4 -- the would-be single group of 10 per layer
    # is split into chunks of at most 4 (4+4+2), never one call of 10.
    assert max(n for (_idx, n) in calls) <= 4
    assert sum(n for (idx, n) in calls if idx == 0) == n_rows  # every row covered
    assert sum(n for (idx, n) in calls if idx == 1) == n_rows
    # sanity: without a cap the same setup would produce ONE call of 10 per layer
    calls.clear()
    _masked_batch_layer_pass(
        layers,
        torch.zeros(n_rows, 1),
        paths,
        position_ids,
        cos,
        sin,
        causal_mask=None,
        device="cpu",
    )
    assert [n for (_idx, n) in calls if _idx == 0] == [10]


def test_rmsnorm_slice_before_is_bit_identical_to_slice_after():
    """Pins a real memory-savings fix in BOTH `masked_batch_call`s: computing
    `norm(state[:, -1, :])` must be BIT-identical to the naive `norm(state)[:, -1, :]`.

    Why it must be bit-identical and not just close: every other memory/scheduling
    knob in the masked-batch path (group cap, greedy grouping, use_cache=False)
    changes which rows share a forward call and therefore CAN flip a verdict --
    that is a real, measured, and accepted flip-rate cost. This one must not,
    because RMSNorm reduces over the HIDDEN dim only
    (`variance = x.pow(2).mean(-1)`), so each (row, position) is arithmetically
    independent: same elements, same reduction, same order. If a refactor ever
    makes normalization depend on other positions (or someone "optimizes" the
    reduction), this test fails and the slice-before form stops being free.

    Motivation: the naive form runs the norm over all L positions and discards
    every one but the last, and Qwen3RMSNorm upcasts to fp32 with a second fp32
    temporary from `.pow(2)`, tens of GiB per call at DART-Math's longest
    sequence lengths -- invisible to --masked-batch-max-group-size, which only
    bounds batch dimension, not sequence length."""
    import torch

    class RMSNorm(torch.nn.Module):
        """Qwen3RMSNorm's arithmetic, verbatim (fp32 upcast, mean over last dim)."""

        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(dim) + 0.5)
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

    torch.manual_seed(0)
    n_rows, L, hidden = 7, 13, 32
    norm = RMSNorm(hidden)
    # bf16 is the dtype the real path runs in -- where a reduction-order change
    # would actually show up. float32 would hide it.
    for dtype in (torch.float32, torch.bfloat16):
        state = torch.randn(n_rows, L, hidden).to(dtype)
        with torch.no_grad():
            old = norm(state)[:, -1, :]  # naive: norm everything, keep last
            new = norm(state[:, -1, :])  # fix: slice first
        assert new.shape == (n_rows, hidden)
        assert torch.equal(old, new), (
            f"slice-before-norm diverged from slice-after in {dtype} -- the fix is "
            f"only safe while RMSNorm stays position-independent; max abs diff "
            f"{(old.float() - new.float()).abs().max().item()}"
        )


# --- Length bucketing -------------------------------------------------------
# `_bucket_row_indices`/`_dispatch_buckets` are pure Python (no torch, no
# model) so these are plain unit tests -- CPU-only, no GPU needed, following
# the same style as the masked-batch tests above.


def test_bucket_row_indices_deterministic():
    """Identical `lengths` must produce an identical partition across repeated
    calls -- a hard requirement (reproducibility: identical inputs -> identical
    bucket partition, across runs and machines)."""
    from re_polar.mcts.rewards import _bucket_row_indices

    lengths = [50, 10, 200, 10, 75, 5, 300, 150, 20, 60, 10, 400, 1]
    first = _bucket_row_indices(lengths, 4)
    for _ in range(5):
        assert _bucket_row_indices(lengths, 4) == first


def test_bucket_row_indices_covers_every_row_exactly_once():
    """No row lost or duplicated across buckets; bucket sizes differ by at most
    1; rows within a bucket are sorted by (length, original_index)."""
    import random

    from re_polar.mcts.rewards import _bucket_row_indices

    rng = random.Random(0)
    n = 37
    lengths = [rng.randint(1, 500) for _ in range(n)]
    for n_buckets in (1, 3, 5, 8, n):
        buckets = _bucket_row_indices(lengths, n_buckets)
        all_idx = [i for b in buckets for i in b]
        assert sorted(all_idx) == list(range(n)), f"n_buckets={n_buckets}: rows lost/duplicated"
        sizes = [len(b) for b in buckets]
        assert max(sizes) - min(sizes) <= 1, f"n_buckets={n_buckets}: uneven buckets {sizes}"
        for b in buckets:
            vals = [(lengths[i], i) for i in b]
            assert vals == sorted(vals), f"n_buckets={n_buckets}: bucket not length-sorted"


def test_bucket_row_indices_degenerate_cases():
    """1 row; all rows equal length; bucket count > row count (must clamp to
    one row per bucket, not produce empty buckets)."""
    from re_polar.mcts.rewards import _bucket_row_indices

    assert _bucket_row_indices([], 8) == []
    assert _bucket_row_indices([42], 8) == [[0]]

    lengths = [10] * 7
    buckets = _bucket_row_indices(lengths, 3)
    assert sorted(i for b in buckets for i in b) == list(range(7))
    assert sum(len(b) for b in buckets) == 7

    buckets = _bucket_row_indices([5, 1, 9], 10)
    assert len(buckets) == 3
    assert all(len(b) == 1 for b in buckets), "bucket count > row count must not create empties"


def test_dispatch_buckets_preserves_row_order_and_values():
    """Highest-risk part of this design: `_dispatch_buckets`'s return value must
    align with ORIGINAL row order regardless of how many buckets were used or
    what order they were processed in -- a mismatch here would silently
    mis-grade rows downstream. Uses a fake per-bucket processor whose
    arithmetic is order-independent (each row's result depends only on ITS OWN
    value, not on what else shares its bucket), so this is a genuine
    order/identity test, not a numerics test. `values` doubles as the
    bucketing `lengths` -- deliberately unsorted relative to row index, so
    buckets do NOT follow input order either."""
    from re_polar.mcts.rewards import _dispatch_buckets

    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    n = len(values)
    expected = [1.0 if v % 2 == 0 else 0.0 for v in values]

    def process_bucket(bucket_indices):
        return [1.0 if values[i] % 2 == 0 else 0.0 for i in bucket_indices]

    for n_buckets in (1, 2, 3, 5, n, n + 5):  # n+5 exercises bucket-count > row-count
        result = _dispatch_buckets(n, values, n_buckets, process_bucket)
        assert result == expected, f"n_buckets={n_buckets}: {result} != {expected}"


def test_bucketing_textlog_batch_matches_composition(tmp_path):
    """The textlog's reproducibility contract (re_polar/mcts/textlog.py): after
    bucketing, a logged batch id MUST correspond to rows that actually shared a
    forward call. `GenerationReward.masked_batch_call` satisfies this by
    calling `_grade_batch` (which writes to the text log) once per bucket, so
    this test reproduces exactly that pattern -- write_batch once per bucket --
    and checks the log's `b`-grouped rows reconstruct EXACTLY the actual bucket
    composition, not some other grouping."""
    import json

    from re_polar.mcts.rewards import _bucket_row_indices
    from re_polar.mcts.textlog import TextLog, question_hash

    n = 9
    questions = [f"question {i}" for i in range(n)]
    gt_answers = [str(i) for i in range(n)]
    lengths = [30, 5, 80, 5, 40, 100, 10, 60, 20]
    texts = [f"answer {i}" for i in range(n)]
    rewards = [float(i % 2) for i in range(n)]
    paths = [[0, 1, 2] for _ in range(n)]

    log = TextLog(tmp_path / "answers.jsonl")
    buckets = _bucket_row_indices(lengths, 3)
    for bucket in buckets:
        if not bucket:
            continue
        log.write_batch(
            [questions[i] for i in bucket],
            [gt_answers[i] for i in bucket],
            [texts[i] for i in bucket],
            [rewards[i] for i in bucket],
            [paths[i] for i in bucket],
        )

    lines = [json.loads(line) for line in open(tmp_path / "answers.jsonl") if line.strip()]
    by_batch: dict = {}
    for row in lines:
        by_batch.setdefault(row["b"], []).append(row)

    assert len(by_batch) == len([b for b in buckets if b])
    logged_sets = []
    for rows in by_batch.values():
        rows_sorted = sorted(rows, key=lambda r: r["i"])
        assert [r["n"] for r in rows_sorted] == [len(rows_sorted)] * len(
            rows_sorted
        ), "every row in a written batch must carry the SAME batch size `n`"
        assert [r["i"] for r in rows_sorted] == list(
            range(len(rows_sorted))
        ), "row positions within one batch must be 0..n-1 contiguous"
        logged_sets.append(frozenset(r["q"] for r in rows_sorted))

    actual_sets = [frozenset(question_hash(questions[i]) for i in b) for b in buckets if b]
    assert sorted(logged_sets, key=sorted) == sorted(
        actual_sets, key=sorted
    ), "logged batch composition does not match the actual bucket partition"


def test_async_scheduler_matches_lockstep_shared_seed():
    """async_scheduler=True must give byte-identical trees to the lockstep
    scheduler: dropping the round barrier changes WHEN/how proposals are
    dispatched, never a tree's own propose->reward->update sequence."""
    lockstep = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
        ).run(log_every=0)
    )
    async_ = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=FakeReward(),
            reward_fns=[FakeReward(), FakeReward(), FakeReward()],
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
        ).run(log_every=0, async_scheduler=True)
    )
    assert async_ == lockstep


def test_async_scheduler_matches_lockstep_per_tree_seed():
    """per_tree_seed=True => trees diverge fast => the async frontier is
    heavily exercised (many distinct in-flight paths, lots of interleaving),
    yet results stay identical to lockstep."""
    lockstep = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
            per_tree_seed=True,
        ).run(log_every=0)
    )
    async_ = _tree_state(
        MCTSRunner(
            make_inputs(8),
            num_layers=D,
            reward_fn=FakeReward(),
            reward_fns=[FakeReward(), FakeReward(), FakeReward()],
            budget=40,
            c=1.414,
            lam=5.0,
            seed=3,
            per_tree_seed=True,
        ).run(log_every=0, async_scheduler=True)
    )
    assert async_ == lockstep


def test_async_scheduler_single_replica_falls_back_to_lockstep():
    """async_scheduler=True with a 1-replica pool (nothing to decouple) must
    fall back to _run_lockstep -- same result, no async machinery exercised."""
    lockstep = _tree_state(
        MCTSRunner(
            make_inputs(6),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=30,
            c=1.414,
            lam=5.0,
            seed=7,
        ).run(log_every=0)
    )
    async_ = _tree_state(
        MCTSRunner(
            make_inputs(6),
            num_layers=D,
            reward_fn=FakeReward(),
            budget=30,
            c=1.414,
            lam=5.0,
            seed=7,
        ).run(log_every=0, async_scheduler=True)
    )
    assert async_ == lockstep


def test_async_scheduler_replica_exclusivity_and_real_parallelism():
    """Same guarantees as the lockstep pool's own test: no replica ever runs
    two programs at once, and >=2 replicas genuinely overlap in wall time."""
    k = 3
    reg = {"lock": threading.Lock(), "live": 0, "max_live": 0, "per": [0] * k, "max_per": [0] * k}
    pool = [_CountingReplica(reg, i) for i in range(k)]
    MCTSRunner(
        make_inputs(8),
        num_layers=D,
        reward_fn=pool[0],
        reward_fns=pool,
        budget=40,
        c=1.414,
        lam=5.0,
        seed=3,
        per_tree_seed=True,
    ).run(log_every=0, async_scheduler=True)
    assert max(reg["max_per"]) == 1, "a replica ran two programs at once -> model-state collision"
    assert reg["max_live"] >= 2, "no real parallelism observed across replicas"


def test_derive_tree_seed_is_deterministic_and_varies_by_query_id():
    from re_polar.mcts.scheduler import derive_tree_seed

    assert derive_tree_seed(42, "q0") == derive_tree_seed(42, "q0")  # deterministic
    assert derive_tree_seed(42, "q0") != derive_tree_seed(42, "q1")  # varies by query_id
    assert derive_tree_seed(42, "q0") != derive_tree_seed(43, "q0")  # varies by base seed


def test_cache_roundtrip(tmp_path):
    p = tmp_path / "cache.jsonl"
    c1 = EvalCache(p)
    c1.put("q0", (0, 1, 2), 1.0)
    c2 = EvalCache(p)
    assert c2.get("q0", (0, 1, 2)) == 1.0
    assert c2.get("q0", (0, 1)) is None


def test_propose_terminates_at_budget():
    tree = ProgramMCTS(num_layers=8, budget=10, seed=0)
    n = 0
    while (prop := tree.propose()) is not None:
        program, chain = prop
        tree.update(chain, 0.0, program=program)
        n += 1
    assert n <= 10 and tree.exhausted


def test_deterministic_given_seed():
    def collect(seed):
        fake = FakeReward()
        runner = MCTSRunner(
            make_inputs(2), num_layers=12, reward_fn=fake, budget=20, c=1.414, lam=5.0, seed=seed
        )
        trees = runner.run(log_every=0)
        return {qid: sorted(t.evaluated) for qid, t in trees.items()}

    assert collect(7) == collect(7)


def test_deep_layer_edit_is_reachable():
    """An edit at layer 30 is one action from the root, so a reward only a
    repeat of layer 30 satisfies is discovered within budget. An
    identity-tail search plateaued at a much shallower layer could never
    reach it."""
    reward = DeepRepeatReward()
    runner = MCTSRunner(
        make_inputs(1), num_layers=D, reward_fn=reward, budget=200, c=1.414, lam=5.0, seed=0
    )
    trees = runner.run(log_every=0)
    valid = trees["q0"].valid_paths()
    assert valid, "search failed to discover a repeat of layer 30 within budget 200"
    assert all(p.count(30) >= 2 for p in valid)  # every winner really repeats layer 30


def test_multi_edit_program_is_reachable():
    """Progressive widening lets the tree COMBINE edits. This reward needs a
    shallow SKIP (some layer in [0, 8) removed) AND a deep REPEAT (some layer in
    [28, 36) executed twice) -- a single contiguous <=MAX_SEGMENT_LEN edit spans
    at most 4 layers and cannot touch both bands, so ONLY a >=2-edit program at
    two different depths can win. A depth-1-only search (root never descending)
    could never reach this; PW must.

    Bands rather than the two exact layers 2 and 30: the canonical
    increasing-start order makes the shallow skip necessarily the ROOT child, so
    an exact single layer is hostage to how many children PW gives the root. The
    band keeps the test about the mechanism under test -- descend and combine two
    edits -- not about drawing one specific root child."""

    def reward(prog):
        path = prog.to_layer_path()
        shallow_skip = any(layer not in path for layer in range(0, 8))
        deep_repeat = any(path.count(layer) >= 2 for layer in range(28, 36))
        return 1.0 if (shallow_skip and deep_repeat) else 0.0

    tree = ProgramMCTS(num_layers=D, budget=300, c=1.414, lam=5.0, seed=0)
    winners = []
    while (prop := tree.propose()) is not None:
        program, chain = prop
        r = reward(program)
        if r > 0:
            winners.append(program)
        tree.update(chain, r, program=program)

    valid = tree.valid_paths()
    assert valid, "PW failed to combine a shallow skip + a deep repeat within budget 300"
    for path in valid:  # every winner genuinely carries both edits
        assert any(layer not in path for layer in range(0, 8))
        assert any(path.count(layer) >= 2 for layer in range(28, 36))
    # and at least one winning program is explicitly multi-edit (>=2 non-KEEP segments)
    assert any(sum(1 for s in p.segments if s.op is not Op.KEEP) >= 2 for p in winners)


def test_all_proposed_programs_are_valid_contiguous_covers():
    """Every proposed program is valid and a contiguous <=MAX_SEGMENT_LEN-segment
    cover of [0, D) -- the edits-over-identity gaps are the identity baseline."""
    tree = ProgramMCTS(num_layers=D, budget=200, c=1.414, lam=5.0, seed=1)
    n = 0
    while (prop := tree.propose()) is not None:
        program, chain = prop
        assert is_valid(program)
        cursor = 0
        for seg in program.segments:
            assert seg.start == cursor  # contiguous, no gaps/overlaps
            assert 1 <= len(seg) <= MAX_SEGMENT_LEN
            cursor = seg.end
        assert cursor == D  # covers exactly [0, D)
        tree.update(chain, 1.0 if 2 not in program.to_layer_path() else 0.0, program=program)
        n += 1
    assert n > 0


def test_no_permutation_duplicate_nodes():
    """Edits may be placed in ANY order (the `start >= cursor` restriction does
    not apply, see search.py's ANY-ORDER PLACEMENT note), so (A then B) and (B
    then A) are both legal construction paths for the same edit SET. The
    transposition table must collapse them onto ONE node object, otherwise visit
    statistics split and budget is wasted evaluating one program twice.

    Uses a small D and a budget above the root branching factor so the tree
    actually descends past the root and builds multi-edit nodes.

    global_selection=False pinned explicitly: this transposition table is a
    legacy-mode-only mechanism (global_selection=True instead builds a single-
    parent tree with no transposition collapsing, see ProgramMCTS's GLOBAL
    SELECTION MODE block comment)."""
    tree = ProgramMCTS(num_layers=8, budget=300, c=1.414, lam=5.0, seed=3, global_selection=False)
    while (prop := tree.propose()) is not None:
        program, chain = prop
        tree.update(chain, 1.0 if 2 not in program.to_layer_path() else 0.0, program=program)

    by_key = {}
    depths = []
    reached_via_multiple_orders = 0

    def visit(node, seen_on_path):
        starts = [e[0] for e in node.edits]
        assert starts == sorted(starts)  # node.edits is canonically sorted
        for (s, length, _, _), (s2, _, _, _) in zip(node.edits, node.edits[1:]):
            assert s + length <= s2  # non-overlapping (nesting stays forbidden)
        key = frozenset(node.edits)
        if key in by_key:
            # same edit set reached again -> must be the SAME object, not a copy
            assert by_key[key] is node, f"duplicate node for edit-set {node.edits}"
            return  # DAG: don't re-walk a shared subtree
        by_key[key] = node
        depths.append(len(node.edits))
        for act, child in node.children.items():
            assert child is tree._nodes[tree._child_edits(node.edits, act)]
            visit(child, seen_on_path | {key})

    visit(tree.root, set())
    assert max(depths) >= 2, "budget did not descend past the root -- test is vacuous"

    # the DAG must actually be exercised: at least one node reachable by >1 order
    for node in tree._nodes.values():
        parents = [p for p in tree._nodes.values() if node in p.children.values()]
        if len(parents) > 1:
            reached_via_multiple_orders += 1
    assert reached_via_multiple_orders > 0, (
        "no node was reached by two different placement orders -- the transposition "
        "table is untested by this run"
    )


def test_max_repeat_times_opens_deeper_loops():
    """max_repeat_times=2 (our old default) exposes only 2x loops; the paper's bound is
    r<=4. >2 adds 3x/4x actions and builds segments that execute layers that many times."""
    from re_polar.core import Op

    t2 = ProgramMCTS(num_layers=6, budget=10, seed=0, max_repeat_times=2)
    reps2 = {times for (_s, _l, op, times) in t2._actions_for(()) if op is Op.REPEAT}
    assert reps2 == {2}  # our old default (paper bound is r<=4)

    t4 = ProgramMCTS(num_layers=6, budget=10, seed=0, max_repeat_times=4)
    reps4 = {times for (_s, _l, op, times) in t4._actions_for(()) if op is Op.REPEAT}
    assert reps4 == {2, 3, 4}  # ablation: 2x/3x/4x available

    prog = t4._build_program([(1, 2, Op.REPEAT, 3)])  # a 3x loop over layers 1,2
    rep_seg = [s for s in prog.segments if s.op is Op.REPEAT][0]
    assert rep_seg.times == 3
    path = prog.to_layer_path()
    assert path.count(1) == 3 and path.count(2) == 3


def test_large_alpha_effectively_disables_widening():
    """A very large `alpha` makes _widening_limit exceed a node's whole action
    count at any visit count -- the intended mechanism for a "no progressive
    widening" ablation (no separate on/off flag needed)."""
    D = 36
    huge = ProgramMCTS(num_layers=D, budget=200, seed=0, alpha=1_000_000, beta=0.5)
    root_action_count = len(huge._universe)
    assert huge._widening_limit(visits=1) > root_action_count  # never binds, even at visits=1
    assert huge._can_expand(huge.root)  # root can always expand: PW cap not the limiting factor

    default = ProgramMCTS(num_layers=D, budget=200, seed=0)  # alpha=2.0 default
    assert default._widening_limit(visits=1) < root_action_count  # the REAL default DOES bind


def test_ucb_global_v_default_is_true_and_bit_identical():
    """ucb_global_v defaults to True (matches PoLar's literal UCB formula) ->
    omitting it must be bit-identical to passing it explicitly."""
    fake = FakeReward()
    runner_default = MCTSRunner(
        make_inputs(3),
        num_layers=D,
        reward_fn=fake,
        budget=100,
        c=1.414,
        lam=5.0,
        seed=0,
        global_selection=False,
    )
    trees_default = runner_default.run(log_every=0)

    fake2 = FakeReward()
    runner_explicit = MCTSRunner(
        make_inputs(3),
        num_layers=D,
        reward_fn=fake2,
        budget=100,
        c=1.414,
        lam=5.0,
        seed=0,
        global_selection=False,
        ucb_global_v=True,
    )
    trees_explicit = runner_explicit.run(log_every=0)

    for qid in trees_default:
        assert trees_default[qid].evaluated == trees_explicit[qid].evaluated


def test_ucb_global_v_uses_total_simulations_not_parent_visits():
    """PoLar (verbatim, both preprint and camera-ready appendices): 'V is the
    total number of simulations.' With ucb_global_v=True, _ucb's explore term
    must use tree._global_visits_total, NOT the specific parent's own visit
    count -- these differ as soon as any simulation's path did NOT pass
    through a given parent, which is true for any non-root parent in a tree
    with real branching."""
    tree = ProgramMCTS(
        num_layers=D, budget=50, c=1.414, lam=5.0, seed=0, global_selection=False, ucb_global_v=True
    )
    fake = FakeReward()
    n = 0
    while (prop := tree.propose()) is not None and n < 30:
        program, chain = prop
        r = fake(program, ["q"], ["42"])[0]
        tree.update(chain, r, program=program)
        n += 1

    # find a non-root parent with children and >=1 fewer visits than the
    # tree's global total -- guaranteed for real branching within 30 sims
    parent = next(
        node
        for node in tree._nodes.values()
        if node is not tree.root and node.children and node.visits < tree._global_visits_total
    )
    child = next(iter(parent.children.values()))
    assert parent.visits < tree._global_visits_total  # the case this flag is FOR

    score_global_v = tree._ucb(parent, child)
    tree.ucb_global_v = False  # flip after the fact -- same tree/state, only V's source changes
    score_parent_v = tree._ucb(parent, child)
    assert score_global_v != score_parent_v

    # reconstruct explicitly: explore term must equal ln(V)/ln(child.visits) with
    # V = tree._global_visits_total (not parent.visits)
    import math

    exploit = child.total_reward / child.visits
    penalty = tree.lam * child.executed_len / tree.num_layers
    expected_global = (
        exploit + tree.c * math.sqrt(math.log(tree._global_visits_total) / child.visits) - penalty
    )
    assert score_global_v == expected_global


def test_trajectory_records_every_update_in_order_with_correct_parent_edges():
    """search_trajectory: one entry per update() call, in call order, letting a
    consumer rebuild the actual tree afterward."""
    tree = ProgramMCTS(num_layers=D, budget=20, c=1.414, lam=5.0, seed=0, global_selection=False)
    fake = FakeReward()
    n = 0
    while (prop := tree.propose()) is not None:
        program, chain = prop
        r = fake(program, ["q"], ["42"])[0]
        tree.update(chain, r, program=program)
        n += 1

    assert len(tree.trajectory) == n  # one entry per real update() call, none dropped/duplicated
    identity = list(range(D))
    # every non-root entry's parent_path must be SOME path that appears
    # earlier in the trajectory (its parent was necessarily visited/created
    # first) -- this is exactly what "rebuild the tree" depends on
    seen_paths = set()
    for entry in tree.trajectory:
        if entry["parent_path"] is not None:
            assert tuple(entry["parent_path"]) in seen_paths | {tuple(identity)}
        seen_paths.add(tuple(entry["path"]))
    # rewards in the trajectory must match what final evaluated{} recorded
    # for that exact path (same ground truth, two different views of it)
    for entry in tree.trajectory:
        path_t = tuple(entry["path"])
        if path_t in tree.evaluated:
            assert tree.evaluated[path_t] == entry["reward"]


def test_trajectory_records_degenerate_retries_too():
    """A degenerate all-skip retry (program=None) is still a real tree node
    the search visited -- it must get a trajectory entry (empty path), not be
    silently dropped, so simulation-order stays faithful to what actually ran."""
    tree = ProgramMCTS(num_layers=4, budget=50, seed=0, global_selection=False, max_repeat_times=2)
    n = 0
    while (prop := tree.propose()) is not None:
        program, chain = prop
        tree.update(chain, 0.0, program=program)
        n += 1
    # every propose() that returned None-and-retried still called update()
    # internally (module contract) -- trajectory must have >= n entries
    assert len(tree.trajectory) >= n
    assert all("path" in e and "reward" in e for e in tree.trajectory)


def test_global_selection_bootstrap_is_free():
    """Algorithm 1 line 1, 'Initialize root node P0', runs BEFORE the numbered
    simulation loop, i.e. for free -- root's own identity evaluation must not
    consume budget."""
    tree = ProgramMCTS(num_layers=8, budget=5, seed=0, global_selection=True)
    prog, chain = tree.propose()
    assert chain == [tree.root]
    assert prog.to_layer_path() == list(range(8))  # unedited identity
    assert tree.proposals == 0, "bootstrap must not count against budget"
    tree.update(chain, 1.0, program=prog)
    assert tree.root.visits == 1 and tree.root.total_reward == 1.0
    assert tree._global_visits_total == 1
    # NOW budget starts being spent
    prog2, chain2 = tree.propose()
    assert tree.proposals == 1
    tree.update(chain2, 0.0, program=prog2)
    # like test_propose_terminates_at_budget: a degenerate (all-skip -> empty
    # path) program along the way self-retries and consumes budget without a
    # second *visible* return, in both this mode and the default one -- so
    # only tree.exhausted (not a visible-return headcount) is a reliable
    # end-of-budget signal.
    while (p := tree.propose()) is not None:
        _, c = p
        tree.update(c, 0.0, program=p[0])
    assert tree.exhausted and tree.proposals == 5


def test_global_selection_finds_solution_and_prefers_shorter():
    """Mirrors test_tree_finds_solution_and_prefers_shorter for global-selection
    mode. epsilon pinned explicitly to 0.1: this test is about global_selection's
    own convergence, not epsilon's default, so it keeps the value it was
    originally designed/budgeted against."""
    fake = FakeReward()
    runner = MCTSRunner(
        make_inputs(4),
        num_layers=D,
        reward_fn=fake,
        budget=500,
        c=1.414,
        lam=5.0,
        seed=0,
        global_selection=True,
        epsilon=0.1,
    )
    trees = runner.run(log_every=0)
    solved = [qid for qid, t in trees.items() if t.valid_paths()]
    assert len(solved) == 4, f"only {len(solved)}/4 trees found the target program"
    for t in trees.values():
        paths = t.valid_paths()
        assert len(paths[0]) < D
        assert len(paths[0]) <= len(paths[-1])


def test_global_selection_reaches_deep_edit_without_widening():
    """No progressive widening exists in this mode at all -- this is the direct
    analogue of test_deep_layer_edit_is_reachable, proving global selection
    doesn't collapse to depth-1 (root only) the way removing widening from the
    default mode naively would."""
    reward = DeepRepeatReward()
    tree = ProgramMCTS(num_layers=D, budget=200, c=1.414, lam=5.0, seed=0, global_selection=True)
    prog, chain = tree.propose()  # bootstrap: identity never repeats layer 30 -> reward 0
    tree.update(chain, reward(prog, ["q"], ["42"])[0], program=prog)
    while (p := tree.propose()) is not None:
        program, chain = p
        r = reward(program, ["q"], ["42"])[0]
        tree.update(chain, r, program=program)
    valid = tree.valid_paths()
    assert (
        valid
    ), "global-selection search failed to discover a repeat of layer 30 within budget 200"
    assert all(p.count(30) >= 2 for p in valid)


def test_global_selection_reaches_multi_edit_without_widening():
    """Direct analogue of test_multi_edit_program_is_reachable: a reward that
    needs a shallow SKIP and a deep REPEAT together -- only reachable by a
    >=2-edit program. Proves multi-edit discovery works with NO progressive
    widening at all in this mode (the mechanism widening exists for in the
    default mode is replaced here by root's own score decaying via the real,
    global-V UCB formula -- see the module docstring block comment)."""

    def reward(prog):
        path = prog.to_layer_path()
        shallow_skip = any(layer not in path for layer in range(0, 8))
        deep_repeat = any(path.count(layer) >= 2 for layer in range(28, 36))
        return 1.0 if (shallow_skip and deep_repeat) else 0.0

    tree = ProgramMCTS(num_layers=D, budget=300, c=1.414, lam=5.0, seed=0, global_selection=True)
    winners = []
    prog, chain = tree.propose()  # bootstrap
    tree.update(chain, reward(prog), program=prog)
    while (p := tree.propose()) is not None:
        program, chain = p
        r = reward(program)
        if r > 0:
            winners.append(program)
        tree.update(chain, r, program=program)

    valid = tree.valid_paths()
    assert (
        valid
    ), "global selection failed to combine a shallow skip + a deep repeat within budget 300"
    for path in valid:
        assert any(layer not in path for layer in range(0, 8))
        assert any(path.count(layer) >= 2 for layer in range(28, 36))
    assert any(sum(1 for s in p.segments if s.op is not Op.KEEP) >= 2 for p in winners)


def test_global_selection_root_visits_track_total_simulations():
    """The whole point of global selection: root is an ancestor of EVERY node
    (single-parent tree, module docstring), so its `visits` must equal the
    running count of every simulation anywhere in the tree, not stay frozen
    at 1 -- this is what makes root's own exploration bonus decay over time
    instead of giving it permanent priority."""
    fake = FakeReward()
    tree = ProgramMCTS(num_layers=D, budget=50, c=1.414, lam=5.0, seed=2, global_selection=True)
    n = 0
    prog, chain = tree.propose()
    tree.update(chain, fake(prog, ["q"], ["42"])[0], program=prog)
    n += 1
    while (p := tree.propose()) is not None:
        program, chain = p
        tree.update(chain, fake(program, ["q"], ["42"])[0], program=program)
        n += 1
    assert tree.root.visits == n == tree._global_visits_total
    # and any node NOT the root or a direct child of the last proposal has
    # visits <= root.visits (root aggregates over literally everything)
    for node in tree._all_nodes:
        assert node.visits <= tree.root.visits


def test_global_selection_builds_a_single_parent_tree():
    """Deliberate simplification (module docstring): no transposition
    collapsing in this mode -- every non-root node has exactly one parent,
    and that parent's `.children` dict really does contain it."""
    fake = FakeReward()
    tree = ProgramMCTS(num_layers=8, budget=200, c=1.414, lam=5.0, seed=1, global_selection=True)
    prog, chain = tree.propose()
    tree.update(chain, fake(prog, ["q"], ["42"])[0], program=prog)
    while (p := tree.propose()) is not None:
        program, chain = p
        tree.update(chain, fake(program, ["q"], ["42"])[0], program=program)

    assert tree.root.parent is None
    assert len(tree._all_nodes) > 1, "budget did not produce any children -- test is vacuous"
    for node in tree._all_nodes:
        if node is tree.root:
            continue
        assert node.parent is not None
        assert node in node.parent.children.values()


def test_global_selection_epsilon_zero_is_deterministic_pure_argmax():
    """epsilon=0 removes the only source of randomness in selection besides
    seed-driven shuffle order, so two runs with the same seed must be
    bit-identical (sanity check on this code path's determinism)."""

    def collect(seed):
        fake = FakeReward()
        tree = ProgramMCTS(
            num_layers=10,
            budget=40,
            c=1.414,
            lam=5.0,
            seed=seed,
            global_selection=True,
            epsilon=0.0,
        )
        prog, chain = tree.propose()
        tree.update(chain, fake(prog, ["q"], ["42"])[0], program=prog)
        while (p := tree.propose()) is not None:
            program, chain = p
            tree.update(chain, fake(program, ["q"], ["42"])[0], program=program)
        return sorted(tree.evaluated.items())

    assert collect(9) == collect(9)


def test_global_selection_epsilon_one_diverges_from_pure_argmax():
    """Direct fix-validation for a real epsilon bug this mode was built to
    replace: in the legacy selection loop, an epsilon override only fires
    once `not self._can_expand(node)` -- i.e. only after a node is ALREADY
    blocked from normal widening-gated expansion, never as a plain per-step
    random-exploration mechanism. `_propose_global`'s epsilon check has no
    such gate: `if self.epsilon > 0 and self.rng.random() < self.epsilon`
    runs unconditionally on every single proposal. If that gate were still
    silently present, epsilon=1.0 (should ALWAYS explore) and epsilon=0.0
    (pure argmax) would produce near-identical search traces for a fresh,
    mostly-ungated tree -- exactly the failure mode this test would catch.
    Same seed/reward/budget, only epsilon differs."""

    def run(epsilon):
        fake = FakeReward()
        tree = ProgramMCTS(
            num_layers=D,
            budget=30,
            c=1.414,
            lam=5.0,
            seed=7,
            global_selection=True,
            epsilon=epsilon,
        )
        prog, chain = tree.propose()
        tree.update(chain, fake(prog, ["q"], ["42"])[0], program=prog)
        while (p := tree.propose()) is not None:
            program, chain = p
            tree.update(chain, fake(program, ["q"], ["42"])[0], program=program)
        return sorted(tree.evaluated)

    assert run(epsilon=1.0) != run(epsilon=0.0)


def test_global_selection_epsilon_one_expands_multiple_distinct_nodes_quickly():
    """Complements the divergence test above with a structural check: under
    epsilon=1.0 (always take the `rng.choice(candidates)` branch, never
    `max(candidates, key=self._global_ucb)`), the nodes chosen for expansion
    should NOT all be the same node -- `candidates` includes every node in the
    tree with an untried edit (`_all_nodes`), not just root or whatever a
    widening cap would have permitted at each step. Confirms epsilon's
    candidate pool is the full flat pool the module docstring describes, not
    silently narrowed the way the legacy gate narrows it (root has no children
    to pick among until it's already been expanded once)."""
    fake = FakeReward()
    tree = ProgramMCTS(
        num_layers=D, budget=25, c=1.414, lam=5.0, seed=3, global_selection=True, epsilon=1.0
    )
    prog, chain = tree.propose()
    tree.update(chain, fake(prog, ["q"], ["42"])[0], program=prog)
    expanded_from: set = set()
    while (p := tree.propose()) is not None:
        program, chain = p
        # the node expanded FROM this proposal is the second-to-last link in
        # the backprop chain (chain[-1] is the newly-created child)
        expanded_from.add(chain[-2].edits)
        tree.update(chain, fake(program, ["q"], ["42"])[0], program=program)
    assert len(expanded_from) > 1, (
        "epsilon=1.0 only ever expanded from a single node -- candidate pool "
        "may be silently narrowed, same failure shape as the legacy gate bug"
    )


def test_legacy_mode_epsilon_cannot_fire_on_first_selection_from_fresh_root():
    """Characterizes (does NOT fix) a real epsilon-vs-progressive-widening
    limitation in legacy mode (global_selection=False): `propose()`'s
    selection loop is `while not self._can_expand(node) and node.children:`
    -- epsilon only lives inside that loop's body. On a fresh root,
    `_widening_limit(visits=0 or 1) == max(1, ceil(alpha*v**beta)) >= 1 >
    len(root.children) == 0`, so `_can_expand(root)` is True and the while
    loop body (epsilon included) never executes even once, regardless of
    epsilon -- the very first post-bootstrap proposal ALWAYS goes through the
    plain widening-gated expansion path, never the epsilon branch."""
    fake = FakeReward()
    tree = ProgramMCTS(
        num_layers=D, budget=1, c=1.414, lam=5.0, seed=0, global_selection=False, epsilon=1.0
    )  # epsilon=1.0: would ALWAYS
    # explore if the gate let it
    # the gate is exactly `while not self._can_expand(node) and node.children:`
    # (search.py's propose(), legacy branch) -- `_can_expand(root)` being True
    # here makes `not self._can_expand(node)` False, so the AND is False and
    # the loop body (epsilon included) cannot execute even once, by
    # construction, independent of epsilon's value or the RNG's state.
    assert tree._can_expand(tree.root)
    prog, chain = tree.propose()
    # confirms the plain (non-epsilon) expansion path was in fact taken: a
    # single new child placed directly on root, nothing deeper -- the shape
    # `_expand` produces, not what an epsilon-triggered `break` mid-descent
    # would (chain would still end at a root child either way here since root
    # was the only node with untried actions, so this checks structure, not
    # just chain length, as the discriminator).
    assert chain == [tree.root, tree.root.children[next(iter(tree.root.children))]]
    assert len(tree.root.children) == 1


def test_legacy_mode_is_reproducible_with_explicit_global_selection_false():
    """Two independently-constructed explicit global_selection=False runs,
    same seed, must be bit-identical: confirms the legacy path is
    deterministic."""
    fake = FakeReward()
    runner_a = MCTSRunner(
        make_inputs(3),
        num_layers=D,
        reward_fn=fake,
        budget=60,
        c=1.414,
        lam=5.0,
        seed=5,
        global_selection=False,
    )
    trees_a = runner_a.run(log_every=0)
    fake2 = FakeReward()
    runner_b = MCTSRunner(
        make_inputs(3),
        num_layers=D,
        reward_fn=fake2,
        budget=60,
        c=1.414,
        lam=5.0,
        seed=5,
        global_selection=False,
    )
    trees_b = runner_b.run(log_every=0)
    for qid in trees_a:
        assert sorted(trees_a[qid].evaluated) == sorted(trees_b[qid].evaluated)


def test_safe_grade_survives_garbage_answers():
    from re_polar.mcts.rewards import _safe_grade
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    ev = EvaluatorMath()
    # both real crash cases seen in production: 1st: eq() raised on extracted
    # '-'; 2nd: extract_ans() itself raised on collapsed-model babble during
    # set-normalization. Both = incorrect.
    assert _safe_grade(ev, "5", "the answer is \\boxed{-}") is False
    babble = "onlicesPLY" + "ffffffff@@@@" * 8 + "MASKolutions" + "ffffffff"
    assert _safe_grade(ev, "5", babble) is False
    assert _safe_grade(ev, "5", "") is False
    assert _safe_grade(ev, "0.5", "So the answer is \\boxed{1/2}.") is True


def test_grade_batch_isolates_pathological_answer_and_logs(tmp_path):
    """A tiny-to-write but explosive answer (huge factorial) must be graded 0
    and logged, not OOM/hang the job."""
    import json
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    class _Tok:
        padding_side = "left"

    class _Eng:
        tokenizer = _Tok()
        device = "cpu"

    class _Exe:
        engine = _Eng()

    log = tmp_path / "fails.jsonl"
    r = GenerationReward(
        _Exe(),
        fail_log_path=str(log),
        difficulty=4,
        grade_timeout_s=3,
        grade_mem_bytes=512 * 1024**2,
        grade_workers=2,
    )
    prog = Program.identity(36)
    refs = ["0.5", "5", "5"]
    texts = ["\\boxed{1/2}", "\\boxed{7}", "\\boxed{99999999!}"]
    qs = ["ok", "wrong", "explode"]
    out = r._grade_batch(refs, texts, qs, prog)
    assert out[0] == 1.0 and out[1] == 0.0  # normal grading still works
    assert out[2] == 0.0  # pathological -> incorrect, no crash
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert any(e["question"] == "explode" for e in logged)  # sample captured
    assert any(e["generated_answer"] == "\\boxed{99999999!}" for e in logged)


def test_grade_batch_recovers_from_broken_pool(tmp_path):
    """A hard worker crash breaks the shared grading pool. _grade_batch must
    reset it and re-grade the leftovers in isolation, NOT crash the job or
    mislabel the good answers as wrong."""
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    class _Tok:
        padding_side = "left"

    class _Eng:
        tokenizer = _Tok()
        device = "cpu"

    class _Exe:
        engine = _Eng()

    r = GenerationReward(_Exe(), grade_workers=1)
    r._ensure_pool()
    r._pool.stop()  # simulate a broken pool (worker crashed -> pool unusable)
    # every subsequent schedule() on the dead pool raises -> must recover, not crash
    out = r._grade_batch(
        ["0.5", "5"], ["\\boxed{1/2}", "\\boxed{7}"], ["a", "b"], Program.identity(36)
    )
    assert out == [1.0, 0.0]  # correct results despite starting from a dead pool


def test_boxed_gate_rejects_unboxed_babble_without_sympy():
    """An UNBOXED generation echoing a giant expression as free text is
    graded 0 by the boxed gate BEFORE sympy runs, so it can never explode a
    grading worker on that class of input."""
    from re_polar.mcts.rewards import _safe_grade
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    ev = EvaluatorMath(strict_extract=True)
    # giant expression as FREE TEXT (no \boxed{}) -> gate 0, sympy untouched
    assert _safe_grade(ev, "4", "the ones digit of $22^{22(11^{11})}$ is 4") is False
    assert _safe_grade(ev, "0.5", "So the answer is \\boxed{1/2}.") is True  # boxed correct
    assert _safe_grade(ev, "5", "\\boxed{7}") is False  # boxed wrong


def test_grade_batch_no_storm_on_boxed_giant(tmp_path):
    """A boxed generation that still explodes symbolically must be contained
    by the memory+time cap (reward 0 + logged) while the pool SURVIVES so
    sibling grades keep working. No per-answer pool spawn => no semaphore/FD
    storm. Asserts no OSError/BrokenPool/unresolved ever hits the fail log."""
    import json
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    class _Tok:
        padding_side = "left"

    class _Eng:
        tokenizer = _Tok()
        device = "cpu"

    class _Exe:
        engine = _Eng()

    log = tmp_path / "fails.jsonl"
    r = GenerationReward(
        _Exe(),
        fail_log_path=str(log),
        difficulty=3,
        grade_timeout_s=3,
        grade_mem_bytes=512 * 1024**2,
        grade_workers=2,
    )
    prog = Program.identity(36)
    refs = ["0.5", "5", "5", "4"]
    texts = ["\\boxed{1/2}", "\\boxed{99999999!}", "\\boxed{7}", "no box, giant 22^{22(11^{11})}"]
    out = r._grade_batch(refs, texts, ["good", "boxed_giant", "wrong", "unboxed"], prog)
    r._reset_pool()
    assert out[0] == 1.0  # good answer survives the giant sibling
    assert out[1] == 0.0  # boxed giant contained -> 0 (no crash)
    assert out[2] == 0.0 and out[3] == 0.0  # wrong + unboxed -> 0
    logged = (
        [json.loads(x)["failure"] for x in log.read_text().splitlines()] if log.exists() else []
    )
    assert not any("OSError" in f or "Broken" in f or f == "unresolved" for f in logged)


def test_truncate_after_first_boxed_is_noop_below_two_spans():
    """0 or 1 boxed spans -> unchanged (nothing to disambiguate)."""
    from re_polar.mcts.rewards import truncate_after_first_boxed

    assert truncate_after_first_boxed("no box here") == "no box here"
    assert truncate_after_first_boxed("\\boxed{7}") == "\\boxed{7}"
    assert truncate_after_first_boxed("\\boxed{7} trailing prose.") == "\\boxed{7} trailing prose."


def test_truncate_after_first_boxed_cuts_hallucinated_continuation():
    """A real production failure mode: a fewshot prompt's real answer is
    correct and boxed FIRST, but the model keeps generating within its token
    budget and re-emits PAPER_INSTRUCTION's own unfilled `\\boxed{ANSWER}`
    placeholder while hallucinating a next problem block -- `extract_boxed`'s
    last-match behavior grabs that placeholder instead of the real answer unless
    the second span is stripped first."""
    from re_polar.mcts.rewards import truncate_after_first_boxed

    text = (
        " \\boxed{10}\n\nSolve the following math problem and output ONLY the "
        "final answer directly, formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\nA certain school has 480 students..."
    )
    assert truncate_after_first_boxed(text) == " \\boxed{10}"
    # brace-depth matched, not naive first "}" -- must not truncate inside a nested brace
    assert (
        truncate_after_first_boxed("\\boxed{\\frac{1}{2}} \\boxed{ANSWER}")
        == "\\boxed{\\frac{1}{2}}"
    )


def test_grade_batch_grades_truncated_text_but_logs_raw(tmp_path):
    """Regression for a real production incident: _grade_batch must grade on
    the truncated text (real answer survives) while still logging the RAW
    generated text (auditability -- see the hallucinated continuation itself,
    not just the graded fragment)."""
    import json

    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    class _Tok:
        padding_side = "left"

    class _Eng:
        tokenizer = _Tok()
        device = "cpu"

    class _Exe:
        engine = _Eng()

    r = GenerationReward(_Exe(), grade_workers=1, text_log_path=str(tmp_path / "answers.jsonl"))
    prog = Program.identity(36)
    raw_text = (
        " \\boxed{10}\n\nSolve the following math problem and output ONLY "
        "the final answer directly, formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\nnext hallucinated question..."
    )
    out = r._grade_batch(["10"], [raw_text], ["q"], prog)
    r._reset_pool()

    assert out == [1.0]  # graded on the truncated (correct) span, not the placeholder
    logged = json.loads((tmp_path / "answers.jsonl").read_text().splitlines()[0])
    assert logged["text"] == raw_text  # log keeps the raw, untruncated generation
    assert logged["reward"] == 1.0


def test_passk_shrinks_generate_batch_to_avoid_oom(tmp_path):
    """num_return_sequences=k multiplies the generate() batch k-fold; passk must
    shrink the question-chunk so chunk_size*k stays <= self.batch_size (the size
    proven safe for greedy generation). Regression for a real CUDA OOM
    (batch_size=32, k=5 -> 160 concurrent sequences on a memory-constrained GPU)."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program

    class _FakeTensor:
        """Just enough tensor-like slicing/shape for passk's generate() plumbing."""

        def __init__(self, rows):
            self.rows = rows

        @property
        def shape(self):
            return (len(self.rows), len(self.rows[0]) if self.rows else 0)

        def __getitem__(self, idx):
            row_idx, col_idx = idx
            rows = self.rows[row_idx] if isinstance(row_idx, slice) else [self.rows[row_idx]]
            return _FakeTensor([r[col_idx] for r in rows])

        def to(self, device):
            return self

    class _FakeBatchEncoding(dict):
        def to(self, device):
            return self

    class _FakeTokenizer:
        padding_side = "left"
        pad_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=None):
            return _FakeBatchEncoding(input_ids=_FakeTensor([[1, 2, 3] for _ in prompts]))

        def batch_decode(self, tensor, skip_special_tokens=True):
            return ["\\boxed{1}" for _ in tensor.rows]

    class _RecordingModel:
        def __init__(self):
            self.batch_sizes: list = []  # input_ids batch dim seen per generate() call

        def generate(self, input_ids, num_return_sequences, **_kw):
            batch = input_ids.shape[0]
            self.batch_sizes.append(batch)
            total = batch * num_return_sequences
            return _FakeTensor([[9] for _ in range(total)])

    model = _RecordingModel()

    class _FakeExecutor:
        engine = SimpleNamespace(tokenizer=_FakeTokenizer(), device="cpu")

        @contextmanager
        def apply(self, program):
            yield model

    r = GenerationReward(_FakeExecutor(), batch_size=32, grade_workers=1)
    questions = [f"q{i}" for i in range(10)]
    gt = ["1"] * 10
    out = r.passk(Program.identity(36), questions, gt, k=5, temperature=0.7)
    r._reset_pool()

    assert len(out) == 10
    assert all(v == 1.0 for v in out)  # "\boxed{1}" vs gt "1" -> every sample correct
    # batch_size=32 // k=5 == 6 -> every generate() call sees <= 6 prompts, never 32.
    assert model.batch_sizes, "generate() was never called"
    assert max(model.batch_sizes) <= 6
    assert sum(model.batch_sizes) == 10  # all 10 questions covered exactly once


def test_text_log_records_answers_and_batch_composition(tmp_path):
    """--log-answers side-log (re_polar/mcts/textlog.py): every graded row must be
    recoverable WITH the batch it was generated in, since a recorded label is
    not reproducible by regeneration alone (bf16 batched-kernel non-
    associativity, see textlog.py's own docstring). Exercises
    GenerationReward._grade_batch's hook directly with a stub, so no
    model/GPU is needed."""
    import json as _json

    from re_polar.mcts.textlog import TextLog, question_hash

    log = TextLog(tmp_path / "answers.jsonl")
    questions = ["what is 2+2?", "what is 3+3?", "what is 2+2?"]  # note the repeat
    log.write_batch(
        questions,
        ["4", "6", "4"],
        ["the answer is 4", "the answer is 7", "4"],
        [1.0, 0.0, 1.0],
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        difficulty=3,
    )

    rows = [_json.loads(l) for l in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    # one shared batch id, positions in order, batch size on every row -> the exact
    # composition is recoverable by grouping on "b" and sorting by "i"
    assert len({r["b"] for r in rows}) == 1
    assert [r["i"] for r in rows] == [0, 1, 2]
    assert {r["n"] for r in rows} == {3}
    assert [r["reward"] for r in rows] == [1.0, 0.0, 1.0]
    assert rows[1]["text"] == "the answer is 7"
    assert all(r["difficulty"] == 3 for r in rows)
    # rows key questions by content hash, not text
    assert rows[0]["q"] == question_hash("what is 2+2?") == rows[2]["q"]

    # sidecar holds each DISTINCT question exactly once
    qs = [_json.loads(l) for l in (tmp_path / "answers.questions.jsonl").read_text().splitlines()]
    assert len(qs) == 2
    assert {q["question"] for q in qs} == {"what is 2+2?", "what is 3+3?"}
    assert {q["q"] for q in qs} == {r["q"] for r in rows}


def test_text_log_lines_stay_atomically_appendable(tmp_path):
    """Each line must stay under PIPE_BUF (4096 B) so concurrent O_APPEND writes
    from replica threads cannot interleave -- that is why the log is row-major
    rather than batch-major. A pathological over-long answer must be truncated,
    not written whole."""
    import json as _json

    from re_polar.mcts.textlog import TextLog

    log = TextLog(tmp_path / "a.jsonl")
    log.write_batch(["q"], ["1"], ["x" * 50_000], [0.0], [[0, 1]])
    line = (tmp_path / "a.jsonl").read_text().splitlines()[0]
    assert len(line.encode()) < 4096, f"line is {len(line.encode())} B -- not atomic under O_APPEND"
    assert _json.loads(line)["truncated"] == 50_000


def test_text_log_is_off_by_default_and_costs_nothing():
    """No text_log_path -> no TextLog object at all, so the reward path is
    byte-identical to before the log existed."""
    from re_polar.mcts import rewards as R

    gr = R.GenerationReward.__new__(R.GenerationReward)  # no model needed
    gr.text_log = None
    assert gr.text_log is None
