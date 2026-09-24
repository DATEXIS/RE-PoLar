"""Write out just the RESCUE population (query_id/path/num_options/
identity_correct) from a mmlu MCTS --log-probs sidecar (probs_<model>_
mmlu.jsonl), with NO GPU work -- reuses `select_rescue_best_programs` from
letter_bias_probe_best_program_run.py (pure stdlib) but skips that script's
slow part (the per-query letter-bias GPU probe pass, an exploratory
diagnostic not needed here -- see that file's own docstring).

Exists to unblock letter_bias_shuffle_robustness.py's --rescue-jsonl input
(it only ever reads query_id/path/num_options per line -- see its
load_rescue()) without needing the full best-program-run job's other output.

--include-identity-correct: broadens the population from RESCUE (identity
wrong, program fixes it) to ANY sample with a valid non-identity program,
regardless of identity's own score -- tests whether reshuffle-fragility is
specific to the rescue mechanism or general to any MCTS-discovered program.
Off by default (RESCUE-only, unchanged behavior).

    python -m analysis.mmlu_pro_track.select_rescue_population \\
        --probs-jsonl probs_qwen3_8b_mmlu.jsonl \\
        --output rescue_population_qwen3_8b.jsonl
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_mod_path = Path(__file__).resolve().parent / "letter_bias_probe_best_program_run.py"
_spec = importlib.util.spec_from_file_location("letter_bias_probe_best_program_run", _mod_path)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
select_rescue_best_programs = _probe.select_rescue_best_programs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--probs-jsonl", required=True)
    parser.add_argument(
        "--include-identity-correct",
        action="store_true",
        help="broaden beyond RESCUE (identity wrong) to any sample with a "
        "valid non-identity program, regardless of identity's own score",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    by_query: Dict[str, List[dict]] = defaultdict(list)
    with open(args.probs_jsonl) as f:
        for line in f:
            r = json.loads(line)
            by_query[r["query_id"]].append(r)
    print(
        f"Loaded {sum(len(v) for v in by_query.values())} records across "
        f"{len(by_query)} distinct query_ids",
        flush=True,
    )

    rescue = select_rescue_best_programs(
        by_query, require_identity_wrong=not args.include_identity_correct
    )
    n_identity_correct = sum(1 for info in rescue.values() if info["identity_correct"])
    print(
        f"{'Any-valid-program' if args.include_identity_correct else 'RESCUE'} queries: "
        f"{len(rescue)} ({n_identity_correct} with identity already correct, "
        f"{len(rescue) - n_identity_correct} identity-wrong/RESCUE)",
        flush=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for qid, info in rescue.items():
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "path": info["path"],
                        "num_options": info["num_options"],
                        "identity_correct": info["identity_correct"],
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(rescue)} records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
