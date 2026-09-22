"""Small, code-verifiable (non-LLM-judged) metrics for the RESCUE error
analysis, kept separate from tag_with_llm.py's LLM-assigned error_category
so they can't be confused with a qualitative judgment call.
"""
import re

# re_polar/mcts/rewards.py's PAPER_INSTRUCTION literally tells the model:
# "output ONLY the final answer directly, formatted strictly as \boxed{ANSWER}."
# -- the paper's own D.4 protocol. The base model frequently parrots that
# instruction's literal example back verbatim (never substituting an actual
# answer for the word "ANSWER") instead of following it. Tolerant of $...$
# math-mode wrapping and internal whitespace, case-insensitive on "ANSWER".
_BOXED_PLACEHOLDER_RE = re.compile(
    r"\$?\s*\\boxed\s*\{\s*\(?\s*(?:\\text\s*\{\s*)?ANSWER\s*\}?\s*\)?\s*\}\s*\$?",
    re.IGNORECASE)


def has_unfilled_boxed_placeholder(text: str) -> bool:
    return bool(_BOXED_PLACEHOLDER_RE.search(text))


def starts_with_unfilled_boxed_placeholder(text: str) -> bool:
    """Stricter form: the placeholder must be the very start of the
    (stripped) response, not just present anywhere -- used by the error-
    analysis artifact to distinguish "the model's first output IS the
    parroted template" from "the template is mentioned later after some
    real attempt" (a fuzzier, weaker signal). Same underlying pattern as
    `has_unfilled_boxed_placeholder`, so a variant found in one (e.g. the
    \\text{...}-wrapped and $-wrapped cases) can't silently drift
    out of sync between the two call sites."""
    return bool(_BOXED_PLACEHOLDER_RE.match(text.strip()))
