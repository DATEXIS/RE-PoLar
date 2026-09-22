"""Rebuilds identity_generated_text/program_generated_text/identity_correct/
program_correct DIRECTLY from the real MCTS text log (re_polar/mcts/textlog.py's
answers_{model}_diff{N}.jsonl, written by --log-answers during search) --
NO regeneration, NO grading call, NO GPU/torch needed at all.

Why this exists instead of just regenerating the text: `select_and_generate.
py` regenerates text on a GPU and regrades it, which does NOT perfectly
reproduce the ORIGINAL MCTS-time verdict for programs evaluated via
`GenerationReward.masked_batch_call` -- a real, separately-documented
reproducibility gap (see that method's own docstring: re-executing a
recorded-valid program in ISOLATION mismatches its recorded label ~13% of the
time). That's moot if you never regenerate: `answers_{model}_diff{N}.jsonl`
already has the EXACT text that produced the EXACT reward MCTS used to label
a path RESCUE in the first place -- `identity_correct`/`program_correct`
become a straight read of that reward, not a re-derived one.

Validated end-to-end before trusting it (qwen3_8b diff1): 1028/1028 identity
paths found in the log with reward matching the existing identity_correct
exactly; 805/805 RESCUE program paths found with reward==1.0 exactly (as
required by definition of RESCUE). This script enforces that same 100%
coverage as a hard requirement (crashes loudly on any miss -- a silent skip
would hide a real coverage gap in some other combo's log).

Join key: (question_hash(question), tuple(exact layer path)) -> (reward,
text), LAST occurrence wins on a duplicate key -- matches
`ProgramMCTS.evaluated[path] = reward`'s own overwrite-on-revisit semantics
(re_polar/mcts/search.py), so a path's joined verdict is whichever evaluation
was authoritative for `final_valid_transitions` at the end of the search.

`--records` is a per-question record list carrying `identity_path`/
`program_path`/`sample_type` (RESCUE/UNRESCUABLE) for every question to
rebuild -- built by `select_and_generate.py`'s `select_rescue`/
`select_base_wrong`/`pick_program` functions applied over a full
`merged_mcts_samples.json` file (not just its own pilot-scale `main()`).

    python -m analysis.error_analysis.build_error_analysis_from_textlog \\
        --records records_qwen3_8b_diff1.jsonl \\
        --answers answers_qwen3_8b_diff1.jsonl \\
        --output error_analysis_qwen3_8b_diff1.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path


def question_hash(question: str) -> str:
    """MUST match re_polar/mcts/textlog.py::question_hash exactly -- reimplemented
    here (not imported) to keep this script torch-free, same situation as
    tag_with_llm.py's own _truncate_after_first_boxed reimplementation."""
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def build_index(answers_path: str) -> dict:
    """{(q_hash, tuple(path)): (reward, text)}, last line wins on a duplicate key."""
    index = {}
    n_lines = 0
    with open(answers_path) as f:
        for line in f:
            n_lines += 1
            row = json.loads(line)
            index[(row["q"], tuple(row["path"]))] = (row["reward"], row["text"])
            if n_lines % 2_000_000 == 0:
                print(f"  ...indexed {n_lines} log lines, {len(index)} unique keys so far",
                      flush=True)
    print(f"Indexed {n_lines} log lines -> {len(index)} unique (q_hash,path) keys", flush=True)
    return index


def rebuild_one(r: dict, index: dict) -> dict:
    out = dict(r)
    qh = question_hash(r["question"])

    id_key = (qh, tuple(r["identity_path"]))
    if id_key not in index:
        raise KeyError(f"identity path not found in text log for query_id={r['query_id']!r} "
                        f"(question_hash={qh}, path={r['identity_path']})")
    id_reward, id_text = index[id_key]
    out["identity_generated_text"] = id_text
    out["identity_correct"] = bool(id_reward)

    if r["sample_type"] == "RESCUE":
        prog_key = (qh, tuple(r["program_path"]))
        if prog_key not in index:
            raise KeyError(f"RESCUE program path not found in text log for "
                            f"query_id={r['query_id']!r} (question_hash={qh}, "
                            f"path={r['program_path']})")
        prog_reward, prog_text = index[prog_key]
        if prog_reward < 1.0:
            raise ValueError(f"query_id={r['query_id']!r} is labeled RESCUE but its "
                              f"program_path's logged reward is {prog_reward}, not 1.0 -- "
                              f"contradicts final_valid_transitions, investigate before trusting "
                              f"this combo's data")
        out["program_generated_text"] = prog_text
        out["program_correct"] = True
    else:
        out["program_generated_text"] = None
        out["program_correct"] = None

    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                    help="existing records_{model}_diff{N}.jsonl (record list + "
                         "identity_path/program_path/sample_type, reused as-is)")
    p.add_argument("--answers", required=True,
                    help="MCTS text log, answers_{model}_diff{N}.jsonl (re_polar/mcts/textlog.py)")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    records = [json.loads(line) for line in Path(args.records).open()]
    print(f"{len(records)} records to rebuild from the real MCTS text log", flush=True)

    index = build_index(args.answers)
    rebuilt = [rebuild_one(r, index) for r in records]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rebuilt:
            f.write(json.dumps(r) + "\n")

    n_rescue = sum(1 for r in rebuilt if r["sample_type"] == "RESCUE")
    print(f"Wrote {len(rebuilt)} records -> {out_path}", flush=True)
    print(f"All {len(rebuilt)} identity paths and all {n_rescue} RESCUE program paths "
          f"found in the log with consistent rewards (else this would have crashed above).",
          flush=True)


if __name__ == "__main__":
    main()
