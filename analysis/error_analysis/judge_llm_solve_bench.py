"""Benchmark the judge LLM's OWN DART-Math solve accuracy, CPU-only (just HTTP
calls, same shape as tag_with_llm.py -- no GPU/torch generation involved).

tag_with_llm.py uses the judge LLM as a fixed-prompt error-category tagger, on
pairs where correctness is already known from the strict grader
(re_polar/core/grader.py) -- the judge never decides correctness there. This script
asks a different, complementary question: if the judge LLM had to SOLVE the
same DART-Math questions itself, cold, how does its own accuracy compare to
the models this repo runs through the identity/MCTS pipeline locally? That's
a standard "judge competence" sanity check to report alongside the tagger --
NOT proof the tagger's error-category judgments are correct (critiquing a
given reasoning trace and solving cold are different skills), just
supporting evidence for whether the judge has any real math competence at
all.

Uses the EXACT SAME instruction text and structure as re_polar/mcts/rewards.py's
PAPER_INSTRUCTION / _paper_input_text -- reimplemented here
rather than imported, same reason tag_with_llm.py reimplements
_truncate_after_first_boxed: re_polar.mcts.rewards imports torch at module top
level, and this script (like tag_with_llm.py) has no other reason to pay
that cost.

--prompt-style selects which of rewards.py's prompt families to send
(default paper_minimal_fewshot = the one-demo variant used as the uniform
prompt style for every other model's headline MCTS number -- the project's
favored prompt everywhere, so it's the default here too, not just an
available choice; paper_default = the zero-shot D.4 wording, still available
via --prompt-style paper_default). The paper's own prompt-format sweep
found paper_minimal_fewshot ("oneshot" there) drops the echo rate to ~0%
on every model tested (0.02% residual on
Qwen3-8B, versus 43.8% under paper_default) even though it embeds
PAPER_INSTRUCTION's literal "\\boxed{ANSWER}" text TWICE (once in the worked
demo, once for the real question) -- seeing the model's own demo answer
already correctly filled in (`\\boxed{2}`, not the placeholder word)
apparently overrides the temptation to echo the instruction, rather than the
raw occurrence count mattering. Not confirmed for the judge LLM specifically
-- that sweep covered the non-reasoning local models, not this chat-only
reasoning endpoint.

Two necessary, deliberate deviations from strict apples-to-apples with the
local models' full-dataset numbers, both flagged so they aren't silently
glossed over when this gets cited:
  1. The judge LLM only exposes a chat-completions endpoint
     (analysis/error_analysis/llm/client.py), not a raw-completion
     endpoint, so the selected style's raw text is sent as a single
     user-turn chat message instead of a raw continuation -- closer to
     prompt_variant_pilot.py's `chat_prompt` delivery mechanism, but with
     the PAPER's own wording, not a different prompt.
  2. --max-tokens defaults to 16000, not the paper's PAPER_MAX_NEW_TOKENS=50
     -- a smoke test at the paper's own 50-token budget found the judge
     opens a <think> block on every response and the budget cuts it off
     mid-thought essentially always (has_boxed~4%, accuracy~0%), the same
     failure class tag_with_llm.py already hits with this exact model. This
     means the resulting number is NOT strictly budget-comparable to the
     local models' 50-token full-dataset row; report it as such, not as a
     literal apples-to-apples figure. See --max-tokens' own --help text.

prompt_template_echo (see analysis_utils.has_unfilled_boxed_placeholder):
reading actual flagged records under the paper_default protocol shows this
is NOT a genuine "model output the unfilled placeholder as its answer" --
PAPER_INSTRUCTION's own zero-shot text contains the literal substring
"\\boxed{ANSWER}" (it's the format instruction, not a fewshot demo), and the
judge, mid-<think>, routinely quotes that instruction back to itself while
deliberating about formatting, then runs out of budget before ever emitting
a real final answer -- a think-budget cutoff, not a prompt-confusion echo.
Flagged so this metric isn't misread the same way as the DIFFERENT
fewshot-demo-hallucination bug truncate_after_first_boxed guards against
below.

Full dataset = the DART-Math split's diff{1-5}/{train,val,test}.json (train
1250 + val 250 + test 500 per difficulty x 5), matching
prompt_variant_pilot.py's --full.

Concurrency/resume/timeout handling mirrors tag_with_llm.py: the judge's
silent-hang and slow-endpoint failure modes are endpoint-level, not specific
to the tagging prompt, so the same defenses apply here -- incremental JSONL
checkpointing per solved record, --output resumes (skips already-solved
query_ids, retries LLM_CALL_TIMEOUT by default), asyncio + Semaphore
(--max-workers), --call-timeout-s force-aborts a hung call via
LLMCallTimeout.

Grading is a SEPARATE pass after all generation finishes (not per-record,
unlike the checkpointing above): reuses prompt_variant_pilot.grade_all_pooled,
the same memory-capped subprocess pool GenerationReward's MCTS reward path
uses (re_polar.core.grader), so a pathological judge answer can't OOM this job the
way in-process grading would. Grading is idempotent and cheap compared to the
network calls, so a restart after a fully-completed run just re-grades
everything rather than needing its own resume tracking -- simpler, and the
expensive part (live network calls) is what's actually protected.

    python -m analysis.error_analysis.judge_llm_solve_bench --calibrate 5
    python -m analysis.error_analysis.judge_llm_solve_bench \\
        --output results/error_analysis/judge_llm_solve_bench.jsonl
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from analysis.error_analysis.llm import LLMCallTimeout, LLMClient, strip_think

# See tag_with_llm.py's own docstring for why this sys.path trick is needed
# under both `python .../judge_llm_solve_bench.py` and tests/'s importlib
# loading.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from prompt_variant_pilot import classify_formatting, grade_all_pooled  # noqa: E402

# MUST match re_polar/mcts/rewards.py's PAPER_INSTRUCTION / _paper_input_text
# EXACTLY (Appendix D.4).
PAPER_INSTRUCTION = ("Solve the following math problem and output ONLY the final "
                     "answer directly, formatted strictly as \\boxed{ANSWER}.")
PAPER_MAX_NEW_TOKENS = 50

# MUST match rewards.py's MINIMAL_FEWSHOT_QUESTION/_ANSWER exactly.
MINIMAL_FEWSHOT_QUESTION = "What is 1+1?"
MINIMAL_FEWSHOT_ANSWER = "2"


def paper_input_text(question: str) -> str:
    return (f"{PAPER_INSTRUCTION}\n"
            "### Problem Start\n"
            f"{question}\n"
            "### Problem End\n"
            "Answer:")


def paper_minimal_fewshot_input_text(question: str) -> str:
    """Mirrors rewards.py's _minimal_fewshot_content: one worked demo (the trivial
    1+1=2 example, PAPER_INSTRUCTION's own Problem-Start/End/Answer structure) +
    the real question, as a single flat string -- content-only, no tokenizer/chat-
    template involved (see module docstring)."""
    demo = paper_input_text(MINIMAL_FEWSHOT_QUESTION) + f" \\boxed{{{MINIMAL_FEWSHOT_ANSWER}}}"
    return demo + "\n\n" + paper_input_text(question)


PROMPT_STYLES = {
    "paper_default": paper_input_text,
    "paper_minimal_fewshot": paper_minimal_fewshot_input_text,
}


def load_dart_math(dataset_dir: str, difficulties: list, splits: list) -> list:
    rows = []
    for d in difficulties:
        for split in splits:
            path = Path(dataset_dir) / f"diff{d}" / f"{split}.json"
            rows.extend(json.loads(path.read_text()))
    return rows


async def solve_one(client: LLMClient, r: dict, max_tokens: int,
                     call_timeout_s: float | None,
                     prompt_fn=paper_minimal_fewshot_input_text) -> dict:
    """Solve one DART-Math question with the judge LLM, prompt_fn's output as
    the only content of a single user turn (see module docstring's note on
    the chat-vs-raw deviation). Never raises on a hung call -- isolates it as
    status=LLM_CALL_TIMEOUT (mirrors tag_with_llm.py's tag_one: a single
    stuck request must not stall a worker slot forever)."""
    prompt = prompt_fn(r["question"])
    try:
        raw = await client.achat([{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=max_tokens,
                                  timeout_s=call_timeout_s)
    except LLMCallTimeout as e:
        print(f"WARNING: LLM call timed out for {r['query_id']}: {e}", flush=True)
        return {**r, "generated_text": None, "raw_llm_response": None,
                "status": "LLM_CALL_TIMEOUT"}
    return {**r, "generated_text": strip_think(raw), "raw_llm_response": raw, "status": "OK"}


def load_existing(out_path: Path) -> dict:
    """query_id -> solved record, read back from a prior (possibly partial)
    run's --output file. Mirrors tag_with_llm.py's load_existing_tags exactly,
    including tolerating a truncated last line (process killed mid-write)."""
    if not out_path.exists():
        return {}
    existing = {}
    with open(out_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: skipping unparseable line {lineno} in {out_path} "
                      f"(likely truncated by a mid-write kill) -- record will be "
                      f"reprocessed", flush=True)
                continue
            existing[rec["query_id"]] = rec
    return existing


DEFAULT_RETRY_STATUSES = frozenset({"LLM_CALL_TIMEOUT"})


def partition_existing_for_retry(existing: dict, retry_statuses) -> tuple:
    keep, retry = {}, {}
    for qid, rec in existing.items():
        bucket = retry if rec.get("status") in retry_statuses else keep
        bucket[qid] = rec
    return keep, retry


async def _run_solving_async(records: list, client: LLMClient, max_workers: int,
                              max_tokens: int, call_timeout_s: float | None,
                              out_path: Path | None,
                              prompt_fn=paper_minimal_fewshot_input_text) -> list:
    sem = asyncio.Semaphore(max_workers)
    solved = [None] * len(records)
    n_done = 0
    out_f = open(out_path, "a") if out_path is not None else None

    async def worker(i: int, r: dict) -> None:
        nonlocal n_done
        async with sem:
            result = await solve_one(client, r, max_tokens, call_timeout_s, prompt_fn)
        solved[i] = result
        n_done += 1  # single-threaded event loop -- no lock needed
        if out_f is not None:
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
        snippet = " ".join((result["generated_text"] or "").split())[:120]
        print(f"solved {n_done}/{len(records)}: {result['query_id']} "
              f"gt={result['gt_ans']!r} -> [{snippet}] status={result['status']}", flush=True)

    try:
        await asyncio.gather(*(worker(i, r) for i, r in enumerate(records)))
    finally:
        if out_f is not None:
            out_f.close()
    return solved


def run_solving(records: list, client: LLMClient, max_workers: int, max_tokens: int,
                 call_timeout_s: float | None = None, out_path=None,
                 prompt_fn=paper_minimal_fewshot_input_text) -> list:
    out = Path(out_path) if out_path is not None else None
    return asyncio.run(_run_solving_async(records, client, max_workers, max_tokens,
                                           call_timeout_s, out, prompt_fn))


def grade_and_summarize(solved: list, grade_workers: int) -> list:
    """Second pass: grade every status=='OK' record through the memory-capped
    pool (see module docstring), leave LLM_CALL_TIMEOUT records as ungraded
    incorrect (never reached sympy, nothing to grade)."""
    from re_polar.vendor.dart_math.eval import extract_boxed

    gradeable_idx = [i for i, s in enumerate(solved) if s["status"] == "OK"]
    print(f"Grading {len(gradeable_idx)}/{len(solved)} solved records "
          f"({grade_workers} memory-capped workers)...", flush=True)
    refs = [solved[i]["gt_ans"] for i in gradeable_idx]
    texts = [solved[i]["generated_text"] for i in gradeable_idx]
    corrects_by_idx = dict(zip(gradeable_idx, grade_all_pooled(refs, texts, workers=grade_workers)))

    final = []
    for i, s in enumerate(solved):
        if i in corrects_by_idx:
            correct = corrects_by_idx[i]
            fmt = classify_formatting(extract_boxed, s["generated_text"])
        else:
            correct = False
            fmt = {"has_boxed": False, "boxed_content": "", "boxed_empty": False,
                   "is_placeholder_echo": False}
        final.append({**s, "correct": correct, **fmt})
    return final


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="./data/dart_math")
    p.add_argument("--difficulty", type=int, action="append", default=None,
                   help="1-5 (repeatable); default 1-5")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="default train+val+test = the full dataset on disk")
    p.add_argument("--output", required=True)
    p.add_argument("--calibrate", type=int, default=0,
                   help="if >0, run only on the first N questions and print raw "
                        "responses -- no output file, no grading")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="DEVIATES from PAPER_MAX_NEW_TOKENS=50 on purpose: a "
                        "--calibrate/--limit 50 smoke test showed the judge LLM "
                        "opens a <think> block on every response and the paper's "
                        "50-token budget cuts it off mid-thought essentially "
                        "always (has_boxed~4%%, accuracy~0%%). A smaller fix "
                        "(4096) looked fine on the easier difficulties but is "
                        "still an arbitrary number picked without seeing the "
                        "harder ones, and the whole POINT of this benchmark is "
                        "to see what the judge can actually do, not measure a "
                        "budget artifact -- raised to match tag_with_llm.py's "
                        "own budget for this exact model instead of guessing a "
                        "new number. This means the resulting number is NOT "
                        "strictly budget-comparable to the local models' "
                        "50-token full-dataset row; report it as such, not as a "
                        "literal apples-to-apples figure")
    p.add_argument("--max-workers", type=int, default=15,
                   help="concurrency against the judge LLM's own endpoint")
    p.add_argument("--call-timeout-s", type=float, default=None,
                   help="hard wall-clock cap per LLM call (see the llm client's "
                        "LLMCallTimeout). Default None = no forced abort. Set this "
                        "for any unattended run -- some endpoints have been observed "
                        "to go silent past their own read timeout without ever "
                        "raising.")
    p.add_argument("--grade-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-statuses", nargs="*", default=None,
                   help="status values treated as NOT done on a resume -- re-sent to "
                        f"the LLM. Default: {sorted(DEFAULT_RETRY_STATUSES)}. Pass with "
                        "no values (--retry-statuses) to disable retrying.")
    p.add_argument("--prompt-style", default="paper_minimal_fewshot", choices=sorted(PROMPT_STYLES),
                   help="which of rewards.py's prompt families to send (see module "
                        "docstring). Defaults to paper_minimal_fewshot, the "
                        "project's favored prompt everywhere, not just here. Pass "
                        "paper_default explicitly to reproduce the original zero-"
                        "shot D.4 run. A resume (--output pointing at an existing "
                        "file) does NOT check this matches the style the file was "
                        "originally written with -- use a distinct --output per "
                        "style, don't reuse one filename across styles.")
    args = p.parse_args(argv)
    prompt_fn = PROMPT_STYLES[args.prompt_style]

    difficulties = args.difficulty or [1, 2, 3, 4, 5]
    records = load_dart_math(args.dataset_dir, difficulties, args.splits)
    print(f"Loaded {len(records)} DART-Math records (difficulties={difficulties}, "
          f"splits={args.splits})", flush=True)
    if args.limit:
        records = records[:args.limit]

    with LLMClient() as client:
        if args.calibrate:
            for r in records[:args.calibrate]:
                prompt = prompt_fn(r["question"])
                resp = strip_think(client.chat([{"role": "user", "content": prompt}],
                                                temperature=0.0, max_tokens=args.max_tokens))
                print(f"=== {r['query_id']} (gt={r['gt_ans']!r}) ===")
                print(resp)
                print()
            return

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing_all = load_existing(out_path)
        retry_statuses = (set(args.retry_statuses) if args.retry_statuses is not None
                           else DEFAULT_RETRY_STATUSES)
        existing, to_retry = partition_existing_for_retry(existing_all, retry_statuses)
        if to_retry:
            print(f"Retrying {len(to_retry)} previously-timed-out records -- dropping "
                  f"their stale entries from {out_path} before this run starts", flush=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                for rec in existing.values():
                    f.write(json.dumps(rec) + "\n")
            tmp_path.replace(out_path)
        remaining = [r for r in records if r["query_id"] not in existing]
        if existing:
            print(f"Resuming {out_path}: {len(existing)}/{len(records)} already solved, "
                  f"{len(remaining)} remaining", flush=True)

        newly_solved = run_solving(remaining, client, args.max_workers, args.max_tokens,
                                    args.call_timeout_s, out_path=out_path, prompt_fn=prompt_fn)

    solved = list(existing.values()) + newly_solved
    n_timed_out = sum(1 for s in solved if s["status"] == "LLM_CALL_TIMEOUT")

    final = grade_and_summarize(solved, args.grade_workers)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for rec in final:
            f.write(json.dumps(rec) + "\n")
    tmp_path.replace(out_path)  # atomic -- a crash mid-write leaves the prior file intact

    n = len(final)
    n_correct = sum(r["correct"] for r in final)
    n_has_boxed = sum(r["has_boxed"] for r in final)
    n_boxed_empty = sum(r["boxed_empty"] for r in final)
    n_placeholder_echo = sum(r["is_placeholder_echo"] for r in final)

    print(f"Wrote {n} graded records -> {out_path}", flush=True)
    print(f"\n=== Summary (protocol={args.prompt_style} via chat, model=judge LLM, "
          f"DART-Math solve-accuracy benchmark) ===", flush=True)
    print(f"n={n}  accuracy={n_correct / n:.1%}  has_boxed={n_has_boxed / n:.1%}  "
          f"boxed_empty={n_boxed_empty / n:.1%}  placeholder_echo={n_placeholder_echo / n:.1%}  "
          f"llm_call_timeout={n_timed_out / n:.1%}", flush=True)


if __name__ == "__main__":
    main()
