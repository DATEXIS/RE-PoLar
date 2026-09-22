"""Generated-answer side-log for MCTS reward evaluation, reproducibility.

`re_polar/datasets/schemas.py::sample_record` stores only the pass/fail
outcome of a (question, program) evaluation -- which paths are valid/invalid
-- not the generated text behind that verdict. Regenerating the text later
doesn't reliably reproduce the original verdict: re-executing a
recorded-valid program in isolation mismatches its recorded label 13.3% of
the time (n=1000), dropping to 5.0% when the original batch=32 composition
is reproduced. Cause: bf16 batched-kernel non-associativity -- kernel/tiling
selection depends on batch shape, so which other questions share a batch
changes rounding, which occasionally flips an argmax over the 50
autoregressive tokens.

So this log records, per evaluation:
  1. the generated text itself, so a label can be checked by diffing text
     rather than re-deriving a batch-shape-sensitive verdict;
  2. the batch composition and order it was generated in, every row of one
     grading batch written as a single record in position order, so a
     reproduction attempt can rebuild the exact batch that produced it.

Design notes:
  * ONE JSONL LINE PER ROW, tagged with a shared batch id `b`, the row's
    position `i` within that batch and the batch size `n`. Regrouping on `b` and
    sorting by `i` recovers the exact batch. A per-batch record would have been
    more compact, but a batch of 32 answers is ~12 KB, and the replica pool
    (`MCTSRunner(reward_fns=[...])`) appends from several threads at once:
    writes above PIPE_BUF (4096 B) are not atomic under O_APPEND and would
    interleave into corrupt lines. A row line is ~400 B, comfortably atomic.
    (`max_new_tokens` is 50, so `text` is a few hundred chars; it is truncated
    defensively at _MAX_TEXT_CHARS anyway to keep that guarantee unconditional.)
  * Question text is written ONCE to a `.questions.jsonl` sidecar keyed by a
    stable content hash; main-log rows carry only the 16-char hash. Joining
    downstream = recompute `question_hash(question)`. Without this the question
    text (often longer than the answer) would dominate the file.
  * Append-only JSONL, flushed per batch: a killed/preempted job keeps every
    batch it finished, and a resumed run appends rather than truncating.
  * Thread-safe, and safe across replicas sharing one path (see the atomicity
    note above); each TextLog instance generates its own random batch ids, so
    two replicas never collide on `b`.

Purely additive: nothing here feeds back into rewards, the search, or the
supervision file. Off unless a `text_log_path` is passed.
"""

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

# keeps one row's JSON line well under PIPE_BUF (4096 B) so O_APPEND writes stay
# atomic across replica threads; 50 max_new_tokens never comes close anyway.
_MAX_TEXT_CHARS = 2000


def question_hash(question: str) -> str:
    """Stable 16-char content hash, the join key between this log's rows and the
    question text (in the `.questions.jsonl` sidecar, and in
    `merged_mcts_samples.json`'s `question` field). Deliberately content-based,
    not positional: query_id is not visible this far down the call stack, and a
    content hash stays valid across reruns, shards and split regenerations."""
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


class TextLog:
    """Append-only JSONL sink for generated answers + their batch composition."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.questions_path = self.path.with_suffix(".questions.jsonl")
        self._lock = threading.Lock()
        self._seen: set = set()
        self.batches = 0
        self.rows = 0

    def write_batch(self, questions: Sequence[str], gt_answers: Sequence[str],
                    texts: Sequence[str], rewards: Sequence[float],
                    paths: Sequence[List[int]],
                    difficulty: Optional[int] = None) -> None:
        """Record one grading batch. All five sequences are PARALLEL, one entry per
        row, already in the batch's own order, `paths[i]` is row i's executed layer
        path (a single program repeated for the serial `__call__` path, genuinely
        per-row under `masked_batch_call`)."""
        hashes = [question_hash(q) for q in questions]
        batch_id = uuid.uuid4().hex[:12]
        n = len(hashes)
        lines = []
        for i, (h, p, r, t) in enumerate(zip(hashes, paths, rewards, texts)):
            text = t if len(t) <= _MAX_TEXT_CHARS else t[:_MAX_TEXT_CHARS]
            row = {"b": batch_id, "i": i, "n": n, "q": h, "path": list(p),
                   "reward": float(r), "text": text}
            if len(text) != len(t):
                row["truncated"] = len(t)
            if difficulty is not None:
                row["difficulty"] = difficulty
            lines.append(json.dumps(row))
        with self._lock:
            # NB: mark seen as we go, not afterwards -- one batch can contain the
            # same question twice (masked_batch_call puts one row per (question,
            # program), so a question recurs once per distinct program that round).
            new = []
            for h, q, g in zip(hashes, questions, gt_answers):
                if h not in self._seen:
                    self._seen.add(h)
                    new.append((h, q, g))
            if new:
                with open(self.questions_path, "a") as f:
                    for h, q, g in new:
                        f.write(json.dumps({"q": h, "question": q, "gt_ans": g}) + "\n")
            with open(self.path, "a") as f:
                f.write("\n".join(lines) + "\n")
            self.batches += 1
            self.rows += n

    def summary(self) -> str:
        return (f"text log: {self.rows} graded rows in {self.batches} batches "
                f"-> {self.path} (+ {len(self._seen)} distinct questions "
                f"-> {self.questions_path})")
