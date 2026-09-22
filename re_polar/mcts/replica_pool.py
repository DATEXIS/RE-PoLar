"""Build a GPU-memory-budgeted pool of independent GenerationReward replicas for
`MCTSRunner(reward_fns=...)`, so cross-sample MCTS parallelism (see the scheduler
module docstring) can fill the GPU WITHOUT running into OOM.

Each replica is its own `LayerEngine` (== its own model copy) -> `ProgramExecutor`
-> `GenerationReward`, all on the SAME device. Same GPU architecture => identical
greedy numerics => result-preserving; do NOT spread replicas across mixed GPU
architectures (greedy decoding diverges across architectures, bf16 kernel
non-associativity).

`torch` / engine imports are LAZY (inside the builders) so the torch-free
scheduler and CPU tests never pull them in. `safe_replica_count` is pure
arithmetic and stays importable everywhere.
"""
from typing import List, Optional, Tuple


def safe_replica_count(free_bytes: int, model_bytes: int, activation_bytes: int,
                       max_replicas: int, safety: float = 0.9) -> int:
    """Largest N with ``N*(model_bytes + activation_bytes) <= safety*free_bytes``,
    clamped to ``[1, max_replicas]``.

    Each replica holds one model copy (``model_bytes``) AND, since all N generate
    concurrently, one in-flight generation batch (``activation_bytes`` = the
    worst-case peak for a single replica's batch: KV cache + activations).
    Budgeting ``N*(model + activation)`` up front reserves room for that concurrent
    peak so the pool never OOMs mid-round. ``safety`` leaves headroom for allocator
    fragmentation and fixed CUDA context overhead. Floor 1: one replica == today's
    single-model job, which already fits by assumption; degenerate inputs -> 1.
    """
    per = model_bytes + activation_bytes
    if per <= 0 or free_bytes <= 0:
        return 1
    n = int((safety * free_bytes) // per)
    return max(1, min(max_replicas, n))


def estimate_model_bytes(model) -> int:
    """Weight footprint of a loaded model: parameter + buffer bytes."""
    params = sum(t.numel() * t.element_size() for t in model.parameters())
    buffers = sum(t.numel() * t.element_size() for t in model.buffers())
    return params + buffers


def estimate_activation_bytes(num_layers: int, hidden_size: int, batch_size: int,
                              seq_len: int, dtype_bytes: int = 2,
                              fudge: float = 2.0) -> int:
    """Conservative worst-case peak activation/KV memory for ONE replica's
    generation batch. The KV cache dominates: ``2 (K+V) * layers * batch * seq *
    hidden * dtype``; ``fudge`` covers attention/temporary activations on top.
    Deliberately over-estimates so ``safe_replica_count`` under-counts N rather
    than OOMs.
    """
    kv = 2 * num_layers * batch_size * seq_len * hidden_size * dtype_bytes
    return int(kv * fudge)


def build_generation_replicas(model_id: str, n: int, *, trust_remote_code: bool = True,
                              reward_kwargs: Optional[dict] = None) -> List:
    """Create ``n`` independent `GenerationReward` replicas (one model copy each)
    on the same device. ``reward_kwargs`` pass through to `GenerationReward`
    (``max_new_tokens``, ``batch_size``, ``difficulty``, ``prompt_style``,
    ``fail_log_path``, ...). ``n >= 1``; ``n == 1`` yields a normal single reward
    that `MCTSRunner` treats exactly as the pre-pool path.
    """
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward

    reward_kwargs = dict(reward_kwargs or {})
    replicas = []
    for _ in range(max(1, n)):
        engine = LayerEngine(model_id, trust_remote_code=trust_remote_code)
        replicas.append(GenerationReward(ProgramExecutor(engine), **reward_kwargs))
    return replicas


def build_budgeted_replicas(model_id: str, *, max_replicas: int, batch_size: int,
                            seq_len: int, trust_remote_code: bool = True,
                            reward_kwargs: Optional[dict] = None, safety: float = 0.9,
                            activation_fudge: float = 2.0) -> Tuple[List, int, dict]:
    """Build one replica, measure its weight footprint and the current free GPU
    memory, compute a safe N with `safe_replica_count`, then build the remaining
    N-1 identical replicas. Returns ``(replicas, n, diag)``; ``diag`` records the
    numbers used, for logging. On CPU / no CUDA, returns a single replica.

    Budget note: after the first replica is built, ``mem_get_info`` free already
    excludes its weights, so the total capacity for the whole pool is taken as
    ``free + model_bytes`` (add the first replica's weights back), and the pool is
    sized as N copies each needing weights + one concurrent generation batch.
    """
    reward_kwargs = dict(reward_kwargs or {})
    reward_kwargs.setdefault("batch_size", batch_size)

    first = build_generation_replicas(model_id, 1, trust_remote_code=trust_remote_code,
                                      reward_kwargs=reward_kwargs)[0]
    engine = first.executor.engine
    model_bytes = estimate_model_bytes(engine.model)

    import torch
    if not torch.cuda.is_available():
        diag = {"n": 1, "reason": "no cuda", "model_bytes": model_bytes}
        return [first], 1, diag

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    hidden = int(getattr(engine.model.config, "hidden_size", 4096))
    act = estimate_activation_bytes(engine.num_layers, hidden, batch_size, seq_len,
                                    fudge=activation_fudge)
    budget_free = free_bytes + model_bytes  # count the already-built replica's weights
    n = safe_replica_count(budget_free, model_bytes, act, max_replicas, safety)

    replicas = [first]
    if n > 1:
        replicas += build_generation_replicas(model_id, n - 1,
                                               trust_remote_code=trust_remote_code,
                                               reward_kwargs=reward_kwargs)
    diag = {"n": n, "model_bytes": model_bytes, "free_bytes": free_bytes,
            "total_bytes": total_bytes, "activation_bytes": act,
            "max_replicas": max_replicas, "safety": safety, "batch_size": batch_size,
            "seq_len": seq_len}
    return replicas, n, diag
