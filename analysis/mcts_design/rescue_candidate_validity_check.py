"""When identity is excluded from the router's top-k, are the remaining
candidates actually MCTS-verified-CORRECT programs for that question, or
just near-identity noise?

Motivation: for the vast majority of questions where identity solves the
problem, MCTS also found a different, non-identity program that solves it
(final_valid_transitions almost always has >1 entry when identity is one of
them). Yet identity-excluded pass@1 drops hard. Those two facts are only
consistent if the router's own top-k beam, once identity is excluded, mostly
does NOT contain one of the MCTS-verified-correct alternatives -- i.e.
decode_topk's rank-2..6 candidates are near-identity perturbations of a
KEEP-biased collapsed distribution, not real draws from the wide space MCTS
actually searched.

This checks that directly: for every question, does the router's
identity-excluded top-k candidate at each rank belong to that question's own
`final_valid_transitions` set (MCTS's own record of which programs it
verified correct)? No online generation/grading needed -- membership is a
pure layer-path comparison against already-computed MCTS labels, so this is
much cheaper than an online-execution rescue evaluation.

Reports membership split by identity-WRONG vs identity-RIGHT subset, not
just pooled over the whole test slice: the pooled cumulative-through-rank-5
membership rate looks far too low to explain real online-execution rescue
rates on the identity-wrong subset -- but MCTS membership is a strict LOWER
BOUND on real correctness (MCTS's own search is not exhaustive; a program
can execute correctly without ever being recorded in
final_valid_transitions), so the two numbers are never directly comparable
pooled across the whole slice. The identity-wrong-only membership rate is
the one actually comparable to a real rescue-only@k number.

Usage:
    python -m analysis.mcts_design.rescue_candidate_validity_check \
        --samples ./data/mcts/qwen3_8b/dart-math-diff-1/merged_mcts_samples.json \
        --checkpoints strict_ce=./results/router_qwen3_8b_strict_ce.pt \
        --model qwen3_8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_test_set_with_valid_sets(samples_path, D, start, end):
    """Full test slice's questions + each one's MCTS-verified-correct program
    set (final_valid_transitions, as layer-path tuples)."""
    data = json.loads(Path(samples_path).read_text())
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    test = [s for s in samples[start:end] if s.get("gt_ans")]
    questions = [s["question"] for s in test]
    valid_sets = [
        {tuple(int(x) for x in p) for p in (s.get("final_valid_transitions") or []) if p}
        for s in test
    ]
    return questions, valid_sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--start", type=int, default=1500)
    ap.add_argument("--end", type=int, default=1750)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoModel
    from re_polar.models import MODEL_REGISTRY
    from re_polar.router.infer import predict_programs_topk
    from re_polar.router.train import load_checkpoint, _resolve_device

    cfg = MODEL_REGISTRY[args.model]
    D = cfg["num_layers"]
    identity = tuple(range(D))
    device = _resolve_device(None)

    questions, valid_sets = load_test_set_with_valid_sets(args.samples, D, args.start, args.end)
    n = len(questions)
    n_multi_alt = sum(1 for vs in valid_sets if identity in vs and len(vs - {identity}) > 0)
    n_id_valid = sum(1 for vs in valid_sets if identity in vs)
    id_wrong_mask = [identity not in vs for vs in valid_sets]
    n_wrong = sum(id_wrong_mask)
    print(f"n={n} test questions | device={device} | identity-wrong subset: {n_wrong} | "
          f"of {n_id_valid} where identity is MCTS-verified-correct, {n_multi_alt} "
          f"({100 * n_multi_alt / max(n_id_valid, 1):.1f}%) also have >=1 non-identity "
          f"MCTS-verified-correct alternative", flush=True)

    for pair in args.checkpoints:
        label, path = pair.split("=", 1)
        ckpt_meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
        encoder = AutoModel.from_pretrained(ckpt_meta["embedding_model_name"]).to(device)
        router = load_checkpoint(path, encoder=encoder).to(device)
        router.eval()

        tk6 = predict_programs_topk(router, questions, k=6, batch_size=args.gen_batch_size)

        # identity-excluded candidate list per question. Tracked POOLED (the
        # whole slice) and split by identity-wrong subset (the subset that's
        # actually comparable to rescue-only@k from real execution).
        def _new_counters():
            return [0] * 5, [0] * 5, [0] * 5  # rank_membership, rank_present, cum_membership_by_k

        pooled = _new_counters()
        wrong_only = _new_counters()
        right_only = _new_counters()

        for cands, vs, is_wrong in zip(tk6, valid_sets, id_wrong_mask):
            filtered = [c for c in cands if tuple(c.to_layer_path()) != identity]
            for bucket in (pooled, wrong_only if is_wrong else right_only):
                rank_membership, rank_present, cum_membership_by_k = bucket
                hit_so_far = False
                for i in range(5):
                    if i < len(filtered):
                        rank_present[i] += 1
                        is_member = tuple(filtered[i].to_layer_path()) in vs
                        if is_member:
                            rank_membership[i] += 1
                            hit_so_far = True
                    if hit_so_far:
                        cum_membership_by_k[i] += 1

        def _report(name, bucket, denom):
            rank_membership, rank_present, cum_membership_by_k = bucket
            r1 = rank_membership[0] / max(rank_present[0], 1)
            c5 = cum_membership_by_k[4] / max(denom, 1)
            print(f"  {label} [{name}, n={denom}]: rank-1 MCTS-verified-correct "
                  f"{rank_membership[0]}/{rank_present[0]} ({100 * r1:.1f}%) | "
                  f"cumulative through rank-5: {cum_membership_by_k[4]}/{denom} ({100 * c5:.1f}%)",
                  flush=True)
            print(f"    per-rank membership: " +
                  ", ".join(f"@{i+1}={rank_membership[i]}/{rank_present[i]}" for i in range(5)),
                  flush=True)

        _report("POOLED", pooled, n)
        _report("identity-WRONG subset (comparable to rescue-only@k)", wrong_only, n_wrong)
        _report("identity-RIGHT subset", right_only, n - n_wrong)

        del router, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone. Low rank-1 membership despite the near-universal existence of a "
          "non-identity MCTS-verified alternative (see the n_multi_alt line above) would "
          "confirm: the router's identity-excluded guesses are near-identity noise from a "
          "collapsed/KEEP-biased beam, not real draws from the space MCTS actually "
          "searched -- explaining the sharp identity-excluded pass@1 drop.", flush=True)


if __name__ == "__main__":
    main()
