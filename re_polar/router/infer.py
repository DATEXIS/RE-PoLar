"""Router INFERENCE + EVALUATION over the DART-Math test splits.

Loads a trained :class:`~re_polar.router.model.PolarRouter` checkpoint, predicts one
layer-execution program per test question, executes+grades those programs with
the *same* :class:`~re_polar.mcts.rewards.GenerationReward` used for MCTS discovery,
and compares router accuracy against the identity-program baseline on TEST:
the router's success criterion is beating that baseline (router_acc >= identity_acc).

Reimplemented from the paper ("Skip a Layer or Loop It? Learning
Program-of-Layers in LLMs", arXiv:2606.06574), no PoLar code is copied or
imported (see NOTICE.md).

Run::

    python -m re_polar.router.infer \\
        --checkpoint ./results/router/router_qwen3_8b.pt \\
        --model qwen3_8b --data-dir ./data/dart_math \\
        --difficulty all --output ./results/router/infer_qwen3_8b.json

CRITICAL STRUCTURE CONSTRAINT: GenerationReward grades answers in a
``spawn`` multiprocessing pool whose workers
RE-IMPORT this module (``__mp_main__``). So this module's TOP LEVEL is kept
TORCH-FREE, only argparse/json/pathlib/typing and
``from re_polar.models import MODEL_REGISTRY`` live here. Every heavy import (torch,
re_polar.core, re_polar.mcts.rewards, re_polar.router.train)
lives INSIDE ``main`` or a helper, so grading workers stay lightweight. Verify
with ``grep -n '^import torch' re_polar/router/infer.py`` (must be empty).
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# NOTE: keep this module's top level TORCH-FREE (see module docstring).
# `models` is torch-free.
from re_polar.models import MODEL_REGISTRY

__all__ = [
    "resolve_difficulties",
    "load_test_split",
    "group_by_program",
    "grade_router",
    "summarize",
    "assert_roundtrips",
    "decode_programs",
    "predict_programs",
    "predict_programs_topk",
    "grade_router_topk",
    "main",
]

DIFFICULTIES = [1, 2, 3, 4, 5]

# reward_fn(program, questions, gt_answers) -> [0/1, ...]; GenerationReward is one.
RewardFn = Callable[[object, List[str], List[str]], List[float]]


# --------------------------------------------------------------------------- #
# pure helpers (torch-free, imported by the CPU-only tests)
# --------------------------------------------------------------------------- #
def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def resolve_difficulties(raw: Optional[Sequence[str]]) -> List[int]:
    """CLI ``--difficulty`` values -> sorted difficulty ints.

    ``None`` / empty (flag omitted) or any literal ``"all"`` -> all of 1..5.
    """
    if not raw:
        return list(DIFFICULTIES)
    if any(str(x).strip().lower() == "all" for x in raw):
        return list(DIFFICULTIES)
    return sorted({int(x) for x in raw})


def load_test_split(data_dir, difficulty: int, limit: Optional[int] = None) -> List[dict]:
    """Load ``<data_dir>/diff{N}/test.json`` (a list of {query_id, question, gt_ans}).

    ``limit`` keeps only the first N questions (smoke runs).
    """
    path = Path(data_dir) / f"diff{difficulty}" / "test.json"
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of questions, got {type(data)}")
    if limit is not None:
        data = data[:limit]
    return data


def group_by_program(programs: Sequence[object]) -> Dict[tuple, List[int]]:
    """Group question indices by executed layer-path (the MCTS program-major idea).

    Two predictions with the same ``to_layer_path()`` execute identically, so they
    share one (batched) generation+grading call. Key = ``tuple(to_layer_path())``.
    """
    groups: Dict[tuple, List[int]] = {}
    for i, program in enumerate(programs):
        key = tuple(program.to_layer_path())
        groups.setdefault(key, []).append(i)
    return groups


def grade_router_topk(
    topk_programs: Sequence[Sequence[object]],
    questions: Sequence[str],
    gt_answers: Sequence[str],
    reward_fn: RewardFn,
) -> tuple:
    """Grade top-k programs per question -> (rewards_at_k, rewards_at_1, chosen).

    ``topk_programs[i]`` is question i's decoded candidates (best-first). Every
    DISTINCT (executed layer-path) is graded ONCE across all questions that hold
    it (same program-major batching as :func:`grade_router`), so pass@k costs at
    most k× the distinct-program generations, never n×k. Then:

    * ``rewards_at_k[i]`` = 1 if ANY of question i's candidates solves (pass@k);
    * ``rewards_at_1[i]`` = reward of the top-1 candidate (pass@1, == the current
      metric, top-1 is exactly ``decode``'s program);
    * ``chosen[i]`` = the shortest solving candidate (for reporting) else top-1.
    """
    n = len(topk_programs)
    if not (len(questions) == len(gt_answers) == n):
        raise ValueError("topk_programs, questions and gt_answers must be the same length")
    # distinct path -> (representative program, question indices that hold it)
    tasks: Dict[tuple, dict] = {}
    for i, cands in enumerate(topk_programs):
        for p in cands:
            key = tuple(p.to_layer_path())
            t = tasks.get(key)
            if t is None:
                tasks[key] = {"program": p, "idxs": [i]}
            elif t["idxs"][-1] != i:  # a question lists a path at most once (decode_topk dedups)
                t["idxs"].append(i)
    pair_reward: Dict[tuple, float] = {}
    for key, t in tasks.items():
        idxs = t["idxs"]
        rs = reward_fn(t["program"], [questions[i] for i in idxs], [gt_answers[i] for i in idxs])
        if len(rs) != len(idxs):
            raise ValueError(f"reward_fn returned {len(rs)} rewards for {len(idxs)} questions")
        for j, i in enumerate(idxs):
            pair_reward[(key, i)] = float(rs[j])
    rewards_at_k: List[float] = [0.0] * n
    rewards_at_1: List[float] = [0.0] * n
    chosen: List[object] = [None] * n
    for i, cands in enumerate(topk_programs):
        best = None
        for j, p in enumerate(cands):
            r = pair_reward.get((tuple(p.to_layer_path()), i), 0.0)
            if j == 0:
                rewards_at_1[i] = r
            if r >= 1.0 and (best is None or len(p.to_layer_path()) < len(best.to_layer_path())):
                best = p
        rewards_at_k[i] = 1.0 if best is not None else 0.0
        chosen[i] = best if best is not None else (cands[0] if cands else None)
    return rewards_at_k, rewards_at_1, chosen


def grade_router(
    programs: Sequence[object],
    questions: Sequence[str],
    gt_answers: Sequence[str],
    reward_fn: RewardFn,
) -> List[float]:
    """Per-question 0/1 reward for the router's predicted programs.

    Groups identical programs (:func:`group_by_program`) and calls ``reward_fn``
    ONCE per distinct program on that group's questions, then scatters the rewards
    back to their original positions. ``reward_fn`` is injected so tests can stub
    it (and to reuse GenerationReward's batched greedy generation verbatim).
    """
    n = len(programs)
    if not (len(questions) == len(gt_answers) == n):
        raise ValueError("programs, questions and gt_answers must be the same length")
    rewards: List[float] = [0.0] * n
    for _key, idxs in group_by_program(programs).items():
        representative = programs[idxs[0]]  # same path => same execution
        group_rewards = reward_fn(
            representative,
            [questions[i] for i in idxs],
            [gt_answers[i] for i in idxs],
        )
        if len(group_rewards) != len(idxs):
            raise ValueError(
                f"reward_fn returned {len(group_rewards)} rewards for {len(idxs)} questions"
            )
        for j, i in enumerate(idxs):
            rewards[i] = float(group_rewards[j])
    return rewards


def summarize(
    rewards: Sequence[float],
    programs: Sequence[object],
    identity_rewards: Sequence[float],
    num_layers: int,
) -> dict:
    """Per-difficulty metrics from graded router + identity rewards.

    Router accuracy vs identity accuracy (the gate), plus program-shape stats:
    mean executed length, and the SKIP / REPEAT fractions (``recurrence_rate`` =
    fraction of programs with >= 1 REPEAT segment).
    """
    from re_polar.core import Op  # torch-free; imported here to keep the top level clean

    router_acc = _mean(rewards)
    identity_acc = _mean(identity_rewards)
    exec_lens = [len(p.to_layer_path()) for p in programs]
    has_skip = [any(s.op is Op.SKIP for s in p.segments) for p in programs]
    has_repeat = [any(s.op is Op.REPEAT for s in p.segments) for p in programs]
    frac_repeat = _mean([1.0 if f else 0.0 for f in has_repeat])
    return {
        "n": len(rewards),
        "router_acc": router_acc,
        "identity_acc": identity_acc,
        "delta": router_acc - identity_acc,
        "mean_executed_len": _mean(exec_lens),
        "mean_identity_len": float(num_layers),
        "frac_programs_with_skip": _mean([1.0 if f else 0.0 for f in has_skip]),
        "frac_programs_with_repeat": frac_repeat,
        "recurrence_rate": frac_repeat,
    }


def assert_roundtrips(programs: Sequence[object], num_layers: int) -> None:
    """Standing invariant: every predicted program re-parses to the same path.

    ``program_from_layer_path`` already re-executes+asserts internally; the extra
    equality check documents the invariant and crashes loudly on any drift.
    """
    from re_polar.router.train import program_from_layer_path

    for program in programs:
        path = program.to_layer_path()
        reparsed = program_from_layer_path(path, num_layers)
        if reparsed.to_layer_path() != path:
            raise AssertionError(f"predicted program did not round-trip: {path}")


# --------------------------------------------------------------------------- #
# router forward + decode (torch inside)
# --------------------------------------------------------------------------- #
def decode_programs(router, seg_logits, op_logits) -> List[object]:
    """One :class:`~re_polar.core.Program` per row of a batched forward output.

    ``decode`` takes a single example, so we split the (B,D) / (B,D,n_ops) batch
    row-by-row. Deterministic (``decode`` is), and each result is a valid Program.
    """
    return [router.decode(seg_logits[i], op_logits[i]) for i in range(seg_logits.shape[0])]


def predict_programs(router, questions: Sequence[str], *, batch_size: int = 64) -> List[object]:
    """Predict one program per question: batched router forward -> per-row decode."""
    import torch

    programs: List[object] = []
    router.eval()
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            chunk = list(questions[start:start + batch_size])
            seg_logits, op_logits = router(questions=chunk)
            programs.extend(decode_programs(router, seg_logits, op_logits))
    return programs


def predict_programs_topk(
    router, questions: Sequence[str], *, k: int = 5, batch_size: int = 64
) -> List[List[object]]:
    """Predict up to ``k`` distinct valid programs per question (best-first).

    Same batched forward as :func:`predict_programs`, but each row is decoded with
    ``router.decode_topk`` (the paper's top-k beam). ``[cands[0] for cands in ...]``
    is exactly :func:`predict_programs` (the top-1), so pass@1 is unchanged.
    """
    import torch

    out: List[List[object]] = []
    router.eval()
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            chunk = list(questions[start:start + batch_size])
            seg_logits, op_logits = router(questions=chunk)
            for i in range(seg_logits.shape[0]):
                out.append(router.decode_topk(seg_logits[i], op_logits[i], k=k))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Router inference + evaluation on TEST.")
    p.add_argument("--checkpoint", required=True, help="router .pt from re_polar.router.train")
    p.add_argument("--model", default="qwen3_8b", choices=sorted(MODEL_REGISTRY),
                   help="target model in MODEL_REGISTRY (sets router D); default qwen3_8b")
    p.add_argument("--data-dir", required=True, help="dir containing diff{N}/test.json")
    p.add_argument("--difficulty", action="append", default=None,
                   help="difficulty split: int (repeatable) or 'all' (default: all 1..5)")
    p.add_argument("--output", required=True, help="path to write the results JSON")
    p.add_argument("--batch-size", type=int, default=64,
                   help="generation batch for GenerationReward (also router forward batch)")
    p.add_argument("--top-k-paths", type=int, default=1,
                   help="decode top-k programs per question and score pass@k (paper's "
                        "top-k beam eval); default 1 == top-1 pass@1 (unchanged behaviour)")
    p.add_argument("--limit", type=int, default=None,
                   help="first N test questions per difficulty (smoke)")
    p.add_argument("--device", default=None, help="cpu / cuda / mps for the router (default: auto)")
    p.add_argument("--prompt-style", default="paper_minimal_fewshot",
                   help="passed through to GenerationReward -- default matches the MCTS "
                        "search's own canonical choice. "
                        "This CLI previously silently defaulted to GenerationReward's OWN "
                        "class default ('raw', no fewshot demo) with no way to override -- "
                        "fixed to make it explicit. Note: grade_router_topk/predict_programs_topk "
                        "themselves take an externally-built reward_fn, so any caller that builds "
                        "its OWN GenerationReward directly is unaffected by this CLI's default.")
    return p


def _resolve_device(name: Optional[str]):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: Optional[Sequence[str]] = None) -> Path:
    # Heavy imports live INSIDE main() so the spawn grading workers that re-import
    # this module stay torch-free (see the module docstring).
    from re_polar.core.layer_engine import LayerEngine
    from re_polar.core import ProgramExecutor
    from re_polar.mcts.rewards import GenerationReward
    from re_polar.core import Program
    from re_polar.router.train import load_checkpoint

    args = _build_arg_parser().parse_args(argv)
    difficulties = resolve_difficulties(args.difficulty)
    device = _resolve_device(args.device)

    cfg = MODEL_REGISTRY[args.model]
    engine = LayerEngine(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    executor = ProgramExecutor(engine)
    D = engine.num_layers
    # Reuse the MCTS reward verbatim: apply-program + batched greedy gen + capped grading.
    reward_fn = GenerationReward(executor, batch_size=args.batch_size, prompt_style=args.prompt_style)

    router = load_checkpoint(args.checkpoint)  # offline rebuild (embed_dim path, no download yet)
    if router.num_layers != D:
        raise SystemExit(
            f"Router D={router.num_layers} != target model {args.model} D={D}; "
            "the checkpoint was trained for a different model."
        )
    router.eval()
    # Materialize the lazy frozen encoder (public trigger), THEN co-locate the
    # whole router (head + encoder submodule) on `device` so encode+head agree.
    router.encode_questions(["warmup"])
    router.to(device)

    per_difficulty: List[dict] = []
    programs_out: Dict[str, List[dict]] = {}
    identity_program = Program.identity(D)

    for diff in difficulties:
        data = load_test_split(args.data_dir, diff, args.limit)
        query_ids = [d["query_id"] for d in data]
        questions = [d["question"] for d in data]
        gt_answers = [d["gt_ans"] for d in data]
        k = max(1, args.top_k_paths)
        print(f"[diff {diff}] {len(questions)} test questions, predicting programs"
              f"{f' (top-{k})' if k > 1 else ''}...")

        identity_rewards = reward_fn(identity_program, questions, gt_answers)
        if k == 1:
            # default path, top-1 pass@1, byte-identical to the original behaviour.
            programs = predict_programs(router, questions, batch_size=args.batch_size)
            assert_roundtrips(programs, D)
            router_rewards = grade_router(programs, questions, gt_answers, reward_fn)
            summary = summarize(router_rewards, programs, identity_rewards, D)
        else:
            # opt-in top-k: report pass@1 (top-1, shape stats) AND pass@k (any-solve).
            topk = predict_programs_topk(router, questions, k=k, batch_size=args.batch_size)
            programs = [cands[0] for cands in topk]  # top-1 == predict_programs
            assert_roundtrips(programs, D)
            rewards_at_k, rewards_at_1, _chosen = grade_router_topk(
                topk, questions, gt_answers, reward_fn)
            summary = summarize(rewards_at_1, programs, identity_rewards, D)
            summary["top_k"] = k
            summary["router_acc_at_k"] = _mean(rewards_at_k)
            summary["delta_at_k"] = summary["router_acc_at_k"] - summary["identity_acc"]
            summary["gate1_at_k_pass"] = summary["router_acc_at_k"] >= summary["identity_acc"]
            summary["mean_candidates"] = _mean([float(len(c)) for c in topk])
        summary["difficulty"] = diff
        summary["gate1_pass"] = summary["router_acc"] >= summary["identity_acc"]
        per_difficulty.append(summary)
        programs_out[str(diff)] = [
            {"query_id": qid, "layer_path": p.to_layer_path()}
            for qid, p in zip(query_ids, programs)
        ]

    overall = {
        "router_acc": _mean([s["router_acc"] for s in per_difficulty]),
        "identity_acc": _mean([s["identity_acc"] for s in per_difficulty]),
        "gate1_pass": all(s["gate1_pass"] for s in per_difficulty),
    }
    if any("router_acc_at_k" in s for s in per_difficulty):
        overall["router_acc_at_k"] = _mean([s.get("router_acc_at_k", s["router_acc"])
                                            for s in per_difficulty])
        overall["gate1_at_k_pass"] = all(s.get("gate1_at_k_pass", s["gate1_pass"])
                                         for s in per_difficulty)

    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "per_difficulty": per_difficulty,
        "overall": overall,
        "programs": programs_out,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    _print_report(args.model, D, args.checkpoint, per_difficulty, overall)
    print(f"\nWrote results -> {out_path}")

    return out_path


def _print_report(model: str, D: int, checkpoint, per_difficulty: List[dict], overall: dict) -> None:
    print(f"\nRouter inference, {model} (D={D}), checkpoint {checkpoint}")
    header = (f"{'diff':>4} {'n':>6} {'router':>8} {'ident':>8} {'delta':>8} "
              f"{'execlen':>8} {'skip%':>6} {'rep%':>6}  gate1")
    print(header)
    has_k = any("router_acc_at_k" in s for s in per_difficulty)
    for s in per_difficulty:
        line = (f"{s['difficulty']:>4} {s['n']:>6} {s['router_acc']:>8.4f} {s['identity_acc']:>8.4f} "
                f"{s['delta']:>+8.4f} {s['mean_executed_len']:>8.2f} "
                f"{s['frac_programs_with_skip'] * 100:>5.1f}% {s['frac_programs_with_repeat'] * 100:>5.1f}%  "
                f"{'PASS' if s['gate1_pass'] else 'FAIL'}")
        if has_k and "router_acc_at_k" in s:
            line += (f"   pass@{s['top_k']} {s['router_acc_at_k']:>8.4f} "
                     f"({s['delta_at_k']:>+.4f} {'PASS' if s['gate1_at_k_pass'] else 'FAIL'})")
        print(line)
    for s in per_difficulty:
        print(f"GATE 1 (router_acc >= identity_acc) diff {s['difficulty']}: "
              f"{'PASS' if s['gate1_pass'] else 'FAIL'} "
              f"({s['router_acc']:.4f} vs {s['identity_acc']:.4f})")
    print(f"\nOVERALL GATE 1: {'PASS' if overall['gate1_pass'] else 'FAIL'} "
          f"(mean router {overall['router_acc']:.4f} vs identity {overall['identity_acc']:.4f})")
    if "router_acc_at_k" in overall:
        print(f"OVERALL pass@k: {'PASS' if overall['gate1_at_k_pass'] else 'FAIL'} "
              f"(mean router_acc_at_k {overall['router_acc_at_k']:.4f} vs identity {overall['identity_acc']:.4f})")


if __name__ == "__main__":
    main()
