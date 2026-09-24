"""Program-major batched MCTS scheduler + on-disk eval cache.

The expensive operation is evaluating (program, input) via generation. Running
one tree at a time wastes the GPU on batch-of-1 calls; instead all trees run
in lockstep rounds:

  round: every non-exhausted tree proposes one complete program
         -> proposals grouped by identical program (path tuple)
         -> per distinct program: ONE executor.apply + batched generation
            over all its inputs (minus cache hits)
         -> rewards distributed, every tree backprops

Early rounds collide heavily on near-identity programs (big batches); later
rounds diverge (smaller groups), still strictly better than per-tree eval,
and the cross-round (query_id, path) cache removes all repeat evaluations.

The identity program is precomputed for every input first: it yields
initial_transition_metric for the output schema and seeds the cache.

PER-TREE SEED (on by default): every tree gets its own deterministic seed
derived from query_id (`derive_tree_seed`, sha256-based, NOT Python's
built-in hash(), which is randomized per-process unless PYTHONHASHSEED is
fixed). Pass `per_tree_seed=False` to instead have all trees share one
seed (still fully supported; see
test_program_major_batching_shares_evals_shared_seed_mode).

Why the default is per-tree, not shared: sharing one seed across all trees
is what makes program-major batching effective (identical root-action
shuffles collide into big batches), but it also means every tree's shuffle
is IDENTICAL, so early rounds propose the same program for every input
before rewards have any chance to diverge them. A per-seed pilot confirmed
this is a real confound, not just a theoretical one: on diff1, shared-seed's
top program covered 51% of inputs vs. per-tree-seed's 4% (top-10 coverage
78% vs 29%, 77% singletons under per-tree-seed) -- the shared seed was
manufacturing an artificial "menu" of common programs rather than reflecting
genuine per-input diversity.

Cost of the per-tree default: it collapses program-major batching (see
test_per_tree_seed_breaks_program_major_batching), empirically ~4.7x more
reward_fn calls than shared-seed for the same budget. The REWARD-FN REPLICA
POOL below exists specifically to recoup this batching loss; our own
cluster benchmarks across GPU architectures show NO real speedup from more
replicas (N=2: ~1.0x on one architecture; on another, N=2: 0.96x, N=4:
0.70x, degrading, not improving), so per-tree seeding is a real,
currently-uncompensated cost increase for any full-scale run, not a free
correctness fix -- measure directly before assuming the pool will absorb
this.

REWARD-FN REPLICA POOL (off by default): the CoLa authors note on
the HF paper page that per-sample searches are independent and "can be
parallelized". `MCTSRunner(..., reward_fns=[rf0, rf1, ...])` takes N independent,
*equivalent* reward callables, each backed by its OWN model copy, and dispatches
a round's distinct-program evaluations across them (one program per replica at a
time; run()/_evaluate_jobs). This fills the GPU when program-major batches are
small (late rounds, and every round under per_tree_seed). It is RESULT-PRESERVING:
each (program, input) is scored by exactly one replica and replicas are
equivalent, so trees/rewards do not depend on the pool size (test_mcts.py::
test_reward_pool_parallel_matches_serial_*). Default (reward_fns=None) wraps the
single `reward_fn` in a 1-element pool -> bit-identical to the pre-pool scheduler.
The pool is backend-agnostic (threads on one GPU today; processes / multi-GPU
later), the scheduler only needs N callables. OOM avoidance is the pool
BUILDER's job (re_polar.mcts.replica_pool.safe_replica_count), not the scheduler's.

PER-REPLICA CUDA STREAMS: without this, all replicas' kernels land
on PyTorch's single default CUDA stream regardless of which thread submitted
them, so they execute strictly in issue order -- concurrent *threads* do not
mean concurrent *GPU work*. `_generate_parallel` now gives each replica its own
`torch.cuda.Stream()` (created lazily, once, cached on the runner instance) so
the GPU can genuinely schedule their kernels concurrently -- this is a native
CUDA capability (concurrent kernel execution across streams within one
process), NOT MPS or multiprocessing, so it costs nothing extra: no new
processes, no duplicate model loads. Measured on a controlled fixed workload
(2 replicas, 32Q each, on a fixed GPU): plain threading = 1.25x over serial,
streams = 1.35x, separate processes (GIL fully removed) = 1.47x -- streams
recovers about half the threads-vs-processes gap for free. None reach ~2x:
Qwen3-8B batch=16 decode is memory-bandwidth-bound (streaming ~16GB of
weights/replica/step dominates over FLOPs), so replicas fundamentally
contend for one shared HBM bus -- a hardware ceiling no dispatch-mechanism
fix removes. No-op on CPU (`torch.cuda.
is_available()` guard) and result-preserving (same tests as the base pool:
`test_reward_pool_parallel_matches_serial_*` do not depend on execution
being GPU-concurrent, only on per-replica exclusivity, which streams don't
change).

ASYNC SCHEDULER (off by default -- `run(async_scheduler=True)`):
the replica pool + CUDA streams above both help WHEN there's concurrent work,
but measured before/after on the real tiny-config bench (8 inputs, budget=12)
neither moved the numbers on a pool-heavy GPU (N=2 0.98x, N=4 regression
0.64x, both unchanged) -- because the lockstep round barrier is the actual
bottleneck there, not dispatch mechanism. Every round requires ALL non-exhausted trees
to propose, then ALL of that round's distinct-program groups to be evaluated,
before ANY tree may propose its next round -- so if a round's proposals
collide onto few distinct paths (the docstring above already notes early
rounds do this, and at budget=12 "early" is most of the run), the pool sits
mostly idle for that whole round regardless of pool size, and a tree that
would be ready to diverge into new territory cannot do so until the group's
slowest sibling group in the SAME round finishes.
`_run_async` replaces the two-phase "gather ALL trees -> evaluate ALL ->
backprop ALL" cadence with a continuous frontier: each tree re-enters a
`ready` queue for its next propose() as soon as ITS OWN reward (cached or
freshly generated) is applied, independent of any other tree's progress. This
lets trees interleave at different logical depths -- a tree whose group
resolved early (e.g., a cache hit, or a fast group) can push into
later-round, more-diverged territory that a still-catching-up tree hasn't
reached yet, mixing "early" and "late" proposals across trees in the SAME
wall-clock window instead of the lockstep scheduler artificially
synchronizing everyone onto the same (possibly collision-heavy) round.
Falls back to `_run_lockstep` for a 1-replica pool (nothing to decouple).
RESULT-PRESERVING (see `_run_async`'s own docstring for the full argument):
`ProgramMCTS.propose`/`.update` are pure per-tree state, verified by reading
search.py, not assumed from this docstring alone -- a tree's own propose ->
reward -> update sequence is identical regardless of scheduling order, and
the shared EvalCache is content-addressed so insertion order doesn't affect
its values. Proven via test_async_scheduler_matches_lockstep_{shared_seed,
per_tree_seed}: byte-identical `evaluated` dicts vs. `_run_lockstep`, plus
test_async_scheduler_replica_exclusivity_and_real_parallelism (mirrors the
pool's own exclusivity/concurrency test). Whether it actually fixes the
N=4 regression / tiny-config starvation on a real multi-day run is a
separate, larger validation not covered by these unit-level checks.

MASKED-BATCH SCHEDULER (off by default -- `MCTSRunner(
masked_batch_reward_fn=...)`): a third, independent opt-in axis alongside the
replica pool and async scheduler, attacking a different bottleneck. The
replica pool parallelizes ACROSS distinct programs by giving each its own
model replica (capped ~1.2-1.5x by shared HBM bandwidth -- N replicas still
each read the full model's weights). Masked-batching instead runs ALL of a
round's distinct programs through ONE shared forward pass -- gather (per
underlying layer) only the rows that still need that layer, run one batched
matmul, scatter back -- so a layer's weights are read from HBM ONCE per round
regardless of how many distinct programs need it, not once per program.
Validated standalone for correctness (bit-identical to an uncached control)
and benchmarked separately: speedup scales with round batch width (0.79x @
N=4 synthetic -> 1.95x @ N=128 synthetic; 2.7-3.0x @ N=256/512 on REAL
search-derived programs) but at a measured **8.6-8.8% verdict-flip-
rate cost vs. the cached serial path on real programs** -- see `rewards.py::
GenerationReward.masked_batch_call`'s docstring for the mechanism (`use_cache
=False` full-prefix recompute, diverges from the `use_cache=True` baseline at
the same bf16-non-associativity level that makes greedy decoding diverge
across GPU architectures).
NOT proven result-preserving the way the replica pool / async scheduler are --
this is a genuine speed/fidelity tradeoff, opt-in and clearly labeled as such
everywhere it's wired in. `_evaluate_jobs` routes a round through
`_generate_masked_batch` instead of `_generate_parallel`/serial whenever
`masked_batch_reward_fn` is set AND the round has >1 distinct pending program
(a single-program round has nothing to batch across).
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from re_polar.core import Program

from .search import DEFAULT_MAX_REPEAT_TIMES, ProgramMCTS


def derive_tree_seed(base_seed: int, query_id: str) -> int:
    """Deterministic per-query-id seed for `per_tree_seed=True`.

    sha256 rather than Python's `hash()`: str hashing is randomized per
    process (PYTHONHASHSEED) unless explicitly disabled, which would make
    per-tree-seed runs non-reproducible across processes/machines.
    """
    digest = hashlib.sha256(f"{base_seed}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class EvalCache:
    """(query_id, path tuple) -> reward; JSONL-backed so cluster reruns resume free."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._d: Dict[Tuple[str, Tuple[int, ...]], float] = {}
        if self.path and self.path.exists():
            with open(self.path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._d[(rec["query_id"], tuple(rec["path"]))] = rec["reward"]
            print(f"EvalCache: loaded {len(self._d)} entries from {self.path}")

    def get(self, query_id: str, path: Tuple[int, ...]) -> Optional[float]:
        return self._d.get((query_id, path))

    def put(self, query_id: str, path: Tuple[int, ...], reward: float):
        self._d[(query_id, path)] = reward
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"query_id": query_id, "path": list(path), "reward": reward}) + "\n")


class MCTSRunner:
    """Drives one tree per input, program-major batched, against a reward fn.

    reward_fn(program, questions, gt_answers) -> list[float]  (see rewards.py)
    inputs: [{"query_id", "question", "gt_ans", ...}], extra keys pass through.
    """

    def __init__(self, inputs: List[dict], num_layers: int, reward_fn, budget: int,
                 c: float, lam: float, seed: int = 42, cache: Optional[EvalCache] = None,
                 max_repeat_times: int = DEFAULT_MAX_REPEAT_TIMES,
                 per_tree_seed: bool = True,
                 reward_fns: Optional[List] = None, epsilon: float = 0.0,
                 masked_batch_reward_fn=None,
                 global_selection: bool = False,
                 ucb_global_v: bool = True,
                 alpha: float = 2.0, beta: float = 0.5):
        self.inputs = inputs
        # MASKED-BATCH SCHEDULER (opt-in, see module docstring below):
        # None (default) -> zero behavior change, _evaluate_jobs takes the exact
        # same branches as before this param existed. When set, a round with >1
        # distinct pending program routes through ONE shared cross-program forward
        # pass (`masked_batch_reward_fn.masked_batch_call`, see rewards.py) instead
        # of one reward_fn call per distinct program (serial or replica-parallel).
        # Measured tradeoff, not a strict improvement -- see
        # `GenerationReward.masked_batch_call`'s own docstring for the
        # ~8.6-8.8% verdict-flip-rate cost this carries on real search-derived
        # programs. Independent of `reward_fns`/the replica pool -- the two opt-ins
        # are mutually exclusive in practice (masked_batch_reward_fn, if set, takes
        # priority in `_evaluate_jobs` whenever a round has >1 distinct job).
        self._masked_batch_reward = masked_batch_reward_fn
        # Reward-fn REPLICA POOL for cross-sample parallelism (module docstring).
        # None -> a 1-element pool wrapping `reward_fn` == the pre-pool scheduler.
        # `reward_fn` stays the primary (identity precompute + the serial path).
        self._reward_pool = list(reward_fns) if reward_fns else [reward_fn]
        self.reward_fn = self._reward_pool[0]
        # lazily built in _generate_parallel: {id(rf): torch.cuda.Stream()},
        # None on CPU / before first parallel round (module docstring).
        self._replica_streams: Optional[dict] = None
        self.cache = cache or EvalCache()
        self.per_tree_seed = per_tree_seed
        # Default (per_tree_seed=True): every tree gets its
        # own deterministic seed (derive_tree_seed, module docstring) -- avoids
        # the confirmed shared-seed menu-collapse confound, at the cost of
        # breaking program-major batching by design (see
        # test_per_tree_seed_breaks_program_major_batching). Pass
        # per_tree_seed=False for the OLD mode: ALL trees share one seed ->
        # identical exploration schedules that only diverge where rewards
        # diverge, which is what makes program-major batching effective (early
        # rounds propose one program for all inputs -- verified by
        # tests/test_mcts.py::test_program_major_batching_shares_evals_
        # shared_seed_mode).
        # global_selection (default False -- was tried as the default for a
        # while, then reverted based on a real-data finding; pass True to opt
        # back into the flat/global mode): forwarded straight through to
        # every tree, see ProgramMCTS's "GLOBAL SELECTION MODE" block
        # comment (re_polar/mcts/search.py) for the mechanism and the finding
        # that reverted the default. Note this scheduler's own
        # identity precompute below (`_evaluate_group(self.identity, ...)`)
        # still runs unconditionally and still writes
        # `self.trees[qid].evaluated[...]` directly, same as every other mode
        # -- it does NOT set the tree's root.visits/total_reward. Under this
        # mode each tree separately bootstraps root's own v/Q on its first
        # propose() call (ProgramMCTS._propose_global's docstring), which
        # re-proposes the identity program and gets served straight from
        # `self.cache` (no second real reward_fn call, just a redundant cache
        # lookup) -- harmless, not optimized away, since ProgramMCTS itself
        # has no dependency on the scheduler and must stay usable standalone.
        self.global_selection = global_selection
        # ucb_global_v (default True, matching PoLar's own
        # stated UCB formula -- pass False for the textbook-UCT local-V
        # reading): forwarded straight through to every tree -- see
        # ProgramMCTS.__init__'s comment (re_polar/mcts/search.py) for what this
        # changes and why.
        self.ucb_global_v = ucb_global_v
        # alpha/beta (progressive-widening scale/exponent, ProgramMCTS's own
        # defaults 2.0/0.5 -- see re_polar/mcts/search.py's _widening_limit):
        # exposed here so a caller can tune or (via a very large alpha, so the
        # cap is never reached within any realistic budget) effectively
        # disable widening without touching ProgramMCTS directly.
        self.alpha, self.beta = alpha, beta
        self.trees = {
            inp["query_id"]: ProgramMCTS(
                num_layers, budget=budget, c=c, lam=lam,
                seed=derive_tree_seed(seed, inp["query_id"]) if per_tree_seed else seed,
                max_repeat_times=max_repeat_times,
                epsilon=epsilon, global_selection=global_selection,
                ucb_global_v=ucb_global_v, alpha=alpha, beta=beta)
            for inp in inputs
        }
        self.by_id = {inp["query_id"]: inp for inp in inputs}
        self.identity = Program.identity(num_layers)
        self.initial_metric: Dict[str, float] = {}

    def _cache_split(self, program: Program, query_ids: List[str]):
        """(path, {qid: cached reward}, [missing qids]), the only cache READ."""
        path = tuple(program.to_layer_path())
        cached, missing = {}, []
        for qid in query_ids:
            hit = self.cache.get(qid, path)
            if hit is None:
                missing.append(qid)
            else:
                cached[qid] = hit
        return path, cached, missing

    def _generate(self, reward_fn, program: Program, missing: List[str]) -> Dict[str, float]:
        """One replica's GPU work: fresh rewards for `missing`. Touches NO shared
        state (no cache, no trees), so replicas may run it concurrently from
        different threads without a lock."""
        if not missing:
            return {}
        fresh = reward_fn(
            program,
            [self.by_id[q]["question"] for q in missing],
            [self.by_id[q]["gt_ans"] for q in missing],
        )
        return dict(zip(missing, fresh))

    def _store(self, path: Tuple[int, ...], fresh: Dict[str, float]):
        """The only cache WRITE; always called serially, after generation."""
        for qid, r in fresh.items():
            self.cache.put(qid, path, r)

    def _evaluate_group(self, program: Program, query_ids: List[str],
                        reward_fn=None) -> Dict[str, float]:
        """Cache-aware evaluation of one program (serial). Used for the identity
        precompute; the per-round path goes through `_evaluate_jobs`."""
        path, cached, missing = self._cache_split(program, query_ids)
        fresh = self._generate(reward_fn or self._reward_pool[0], program, missing)
        self._store(path, fresh)
        return {**cached, **fresh}

    def _evaluate_jobs(self, jobs) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Evaluate a round's distinct-program jobs -> {path: {qid: reward}}.

        `jobs`: [(path, program, [query_ids]), ...], one per distinct program this
        round (paths are unique). Cache read (phase 1) and write (phase 3) stay
        SERIAL; only the reward-fn generation (phase 2) fans out across the
        replica pool. A 1-replica pool takes the serial branch and is
        bit-identical to calling `_evaluate_group` per job."""
        prepared = [(*self._cache_split(program, qids), program)
                    for _path, program, qids in jobs]  # (path, cached, missing, program)
        gen_jobs = [(path, program, missing)
                    for path, _cached, missing, program in prepared if missing]
        if self._masked_batch_reward is not None and len(gen_jobs) > 1:
            fresh_by_path = self._generate_masked_batch(gen_jobs)
        elif len(self._reward_pool) > 1 and len(gen_jobs) > 1:
            fresh_by_path = self._generate_parallel(gen_jobs)
        else:
            rf = self._reward_pool[0]
            fresh_by_path = {path: self._generate(rf, program, missing)
                             for path, program, missing in gen_jobs}
        results = {}
        for path, cached, _missing, _program in prepared:
            fresh = fresh_by_path.get(path, {})
            self._store(path, fresh)
            results[path] = {**cached, **fresh}
        return results

    def _generate_parallel(self, gen_jobs) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Run `gen_jobs` across the replica pool, <=1 job per replica at a time.

        Two programs must never share one replica concurrently (they'd clobber
        each other's layer rerouting). A blocking queue of replicas enforces it: a
        task checks a replica out for its duration and returns it. max_workers ==
        pool size, so exactly one distinct replica is live per running task.

        Each replica also gets its own `torch.cuda.Stream()` (module docstring,
        "PER-REPLICA CUDA STREAMS") so concurrent threads translate into
        concurrent GPU kernel execution instead of all queuing on the default
        stream. Built once, lazily, and cached on the instance -- cheap but no
        reason to recreate every round. No-op on CPU."""
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue

        import torch

        if torch.cuda.is_available():
            if self._replica_streams is None:
                self._replica_streams = {id(rf): torch.cuda.Stream() for rf in self._reward_pool}
            streams_by_id = self._replica_streams
        else:
            streams_by_id = {}

        available: "Queue" = Queue()
        for rf in self._reward_pool:
            available.put(rf)

        def _task(path, program, missing):
            rf = available.get()
            try:
                stream = streams_by_id.get(id(rf))
                if stream is not None:
                    # whole call (through GenerationReward's implicit CPU sync at
                    # tokenizer.batch_decode) runs on this replica's own stream --
                    # returning here already waited for that stream's work, no
                    # separate synchronize() needed (see diagnose_replica_
                    # overlap.py::_run_one for the same reasoning, tested there).
                    with torch.cuda.stream(stream):
                        return path, self._generate(rf, program, missing)
                return path, self._generate(rf, program, missing)
            finally:
                available.put(rf)

        out: Dict[Tuple[int, ...], Dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=len(self._reward_pool)) as ex:
            for fut in [ex.submit(_task, *gj) for gj in gen_jobs]:
                path, fresh = fut.result()
                out[path] = fresh
        return out

    def _generate_masked_batch(self, gen_jobs) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """Cross-program masked/gathered batching for one round (module docstring
        "MASKED-BATCH SCHEDULER"). Flattens ALL (path, qid) rows across ALL of this
        round's distinct-program jobs into ONE call to `masked_batch_reward_fn.
        masked_batch_call` -- a single shared forward pass covering every row,
        instead of one reward-fn call per distinct program. Reshapes the flat
        per-row reward list back into the same `{path: {qid: reward}}` shape
        `_generate_parallel`/the serial branch return, so callers (`_evaluate_jobs`)
        don't need to know which branch ran."""
        rows: List[Tuple[Tuple[int, ...], str, Program, str, str]] = []
        for path, program, missing in gen_jobs:
            for qid in missing:
                inp = self.by_id[qid]
                rows.append((path, qid, program, inp["question"], inp["gt_ans"]))
        if not rows:
            return {}
        programs = [r[2] for r in rows]
        questions = [r[3] for r in rows]
        gt_answers = [r[4] for r in rows]
        rewards = self._masked_batch_reward.masked_batch_call(programs, questions, gt_answers)
        out: Dict[Tuple[int, ...], Dict[str, float]] = {}
        for (path, qid, _program, _q, _g), r in zip(rows, rewards):
            out.setdefault(path, {})[qid] = r
        return out

    def run(self, log_every: int = 1, async_scheduler: bool = False) -> Dict[str, ProgramMCTS]:
        """async_scheduler=False (default): the original lockstep round loop,
        completely unchanged -- zero behavior/risk change for any existing
        caller. async_scheduler=True: drop the round barrier (module docstring
        "ASYNC SCHEDULER"); only takes effect with a >1-replica pool (falls
        back to lockstep otherwise -- nothing to decouple with one replica)."""
        if async_scheduler and len(self._reward_pool) > 1:
            return self._run_async(log_every)
        return self._run_lockstep(log_every)

    def _run_lockstep(self, log_every: int = 1) -> Dict[str, ProgramMCTS]:
        # identity precompute: initial_transition_metric + cache seed + tree seed
        identity_rewards = self._evaluate_group(self.identity, list(self.trees))
        for qid, r in identity_rewards.items():
            self.initial_metric[qid] = r
            self.trees[qid].evaluated[tuple(self.identity.to_layer_path())] = r

        rounds = 0
        while True:
            pending = defaultdict(list)  # path tuple -> [(qid, program, chain)]
            for qid, tree in self.trees.items():
                proposal = tree.propose()
                if proposal is not None:
                    program, chain = proposal
                    pending[tuple(program.to_layer_path())].append((qid, program, chain))
            if not pending:
                break
            rounds += 1
            jobs = [(path, entries[0][1], [qid for qid, _, _ in entries])
                    for path, entries in pending.items()]
            results = self._evaluate_jobs(jobs)
            for path, entries in pending.items():
                rewards = results[path]
                for qid, prog, chain in entries:
                    self.trees[qid].update(chain, rewards[qid], program=prog)
            if log_every and rounds % log_every == 0:
                done = sum(t.exhausted for t in self.trees.values())
                solved = sum(bool(t.valid_paths()) for t in self.trees.values())
                print(f"round {rounds}: distinct programs={len(pending)}, "
                      f"trees done={done}/{len(self.trees)}, solved={solved}, "
                      f"replicas={len(self._reward_pool)}")
        return self.trees

    def _run_async(self, log_every: int = 1) -> Dict[str, ProgramMCTS]:
        """No round barrier: each tree proposes/evaluates/backprops on its own
        pace instead of every tree waiting for the round's slowest group.

        WHY THIS IS RESULT-PRESERVING (see also module docstring "ASYNC
        SCHEDULER"): `ProgramMCTS.propose`/`.update` touch ONLY that tree's own
        state (rng, tree nodes, proposals counter, evaluated dict) -- nothing
        cross-tree, nothing global (verified by reading search.py, not just
        assumed). A tree's trajectory is therefore identical to the lockstep
        scheduler's REGARDLESS of what other trees are doing concurrently, AS
        LONG AS that tree is never asked to propose again before its current
        proposal's reward has been applied via update(). This function
        enforces exactly that: a qid only re-enters `ready` after its reward
        (cached or freshly generated) has been backpropped. The shared
        EvalCache is content-addressed and rewards are a deterministic
        function of (program, question, gt_answer) (greedy decoding), so cache
        VALUES don't depend on insertion order -- only WHICH physical
        generate() call computed a given reward changes, which is not
        externally visible. Proven, not just argued: test_async_scheduler_
        matches_lockstep_* assert byte-identical `evaluated` dicts across both
        per_tree_seed settings.

        Cache reads/writes and every tree.update() call happen on the driving
        thread only (inside this loop, after a future resolves or on the
        immediate cache-hit path below) -- worker threads (_gpu_task) touch no
        shared state, matching the "cache stays serial" invariant the lockstep
        path also relies on (_evaluate_jobs docstring)."""
        identity_rewards = self._evaluate_group(self.identity, list(self.trees))
        for qid, r in identity_rewards.items():
            self.initial_metric[qid] = r
            self.trees[qid].evaluated[tuple(self.identity.to_layer_path())] = r

        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from queue import Queue

        import torch

        pool = self._reward_pool
        if torch.cuda.is_available():
            if self._replica_streams is None:
                self._replica_streams = {id(rf): torch.cuda.Stream() for rf in pool}
            streams_by_id = self._replica_streams
        else:
            streams_by_id = {}
        available: "Queue" = Queue()
        for rf in pool:
            available.put(rf)

        def _gpu_task(path, program, entries):
            rf = available.get()
            try:
                missing_qids = [qid for qid, _prog, _chain in entries]
                stream = streams_by_id.get(id(rf))
                if stream is not None:
                    with torch.cuda.stream(stream):
                        fresh = self._generate(rf, program, missing_qids)
                else:
                    fresh = self._generate(rf, program, missing_qids)
                return path, program, entries, fresh
            finally:
                available.put(rf)

        ready = list(self.trees)      # qids due for a fresh propose()
        frontier: Dict[Tuple[int, ...], list] = defaultdict(list)  # path -> pending entries
        rounds = 0

        def _propose_ready_into_frontier():
            nonlocal ready
            for qid in ready:
                proposal = self.trees[qid].propose()
                if proposal is None:
                    continue  # exhausted: dropped, never re-enters `ready`
                program, chain = proposal
                frontier[tuple(program.to_layer_path())].append((qid, program, chain))
            ready = []

        def _dispatch_frontier(in_flight: dict) -> list:
            """Drain `frontier`: resolve cache hits immediately (no GPU wait,
            backprop right away), submit cache misses as GPU tasks. Returns
            qids ready to propose again from the immediate cache-hit path."""
            nonlocal rounds
            newly_ready = []
            for path in list(frontier.keys()):
                entries = frontier.pop(path)
                program = entries[0][1]
                qids = [e[0] for e in entries]
                _path, cached, missing = self._cache_split(program, qids)
                for qid, _prog, chain in entries:
                    if qid in cached:
                        self.trees[qid].update(chain, cached[qid], program=program)
                        newly_ready.append(qid)
                remaining = [e for e in entries if e[0] in missing]
                if remaining:
                    rounds += 1
                    fut = ex.submit(_gpu_task, path, program, remaining)
                    in_flight[fut] = path
            return newly_ready

        with ThreadPoolExecutor(max_workers=len(pool)) as ex:
            in_flight: dict = {}
            while in_flight or ready or frontier:
                if ready:
                    _propose_ready_into_frontier()
                if frontier:
                    ready = _dispatch_frontier(in_flight)
                    continue  # keep feeding the pool before blocking on a future
                if not in_flight:
                    break
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                newly_ready = []
                for fut in done:
                    path = in_flight.pop(fut)
                    _path, program, entries, fresh = fut.result()
                    self._store(path, fresh)
                    for qid, _prog, chain in entries:
                        self.trees[qid].update(chain, fresh[qid], program=program)
                        newly_ready.append(qid)
                ready = newly_ready
                if log_every and rounds % log_every == 0:
                    done_ct = sum(t.exhausted for t in self.trees.values())
                    solved = sum(bool(t.valid_paths()) for t in self.trees.values())
                    print(f"async: dispatched={rounds}, in_flight={len(in_flight)}, "
                          f"trees done={done_ct}/{len(self.trees)}, solved={solved}, "
                          f"replicas={len(pool)}")
        return self.trees
