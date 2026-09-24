"""Bridge: mmlu_pro_domains MCTS supervision -> re_polar.router.train's expected shape.

`re_polar/datasets/schemas.py`'s `sample_record` is domain-agnostic -- it stores
whatever `inp["question"]` was handed to it. DART-Math hands it a plain
string; the MMLU-Pro-domains MCTS run hands it a NESTED DICT
`{"question", "options", "category"}` (the shape
`LogLikReward`/`MMLUProSample` need for grading). `re_polar.router.train` assumes
`sample["question"]` is a hashable STRING (used as an `Example.question` dict
key in `encode_examples`, and fed directly to the frozen text encoder), so
mmlu's merged_mcts_samples.json cannot be passed to `re_polar.router.train.main`
unmodified.

This module is the seam: it flattens each mmlu question dict into the same
lettered multiple-choice prompt text `LogLikReward`'s own benchmark uses
(`re_polar.core.mmlu_pro_scoring._build_prompt`, reused verbatim -- not reimplemented),
so `re_polar.router.train.build_examples` / `encode_examples` / `train` run on
mmlu supervision COMPLETELY UNCHANGED. The original structured dict is kept
alongside (under `question_struct`) for reward-function grading, which still
needs `{"question","options","category"}` + an integer `gt_ans`, not a
formatted string.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union

__all__ = ["format_mmlu_question", "load_mmlu_supervision"]


def format_mmlu_question(question: Dict, *, no_think: bool = True) -> str:
    """mmlu question dict `{"question","options","category"?}` -> router-encodable text.

    Reuses `re_polar.core.mmlu_pro_scoring._build_prompt` (the exact lettered-choice prompt
    `LogLikReward`/`run_mmlu_pro_domains` score against) so the router's frozen
    text encoder sees the same surface form the target LLM is graded on.
    """
    from re_polar.core.mmlu_pro_scoring import MMLUProSample, _build_prompt

    sample = MMLUProSample(
        id=0,
        question=str(question["question"]),
        options=tuple(str(o) for o in question["options"]),
        answer_index=0,
        category=str(question.get("category", "")),
    )
    return _build_prompt(sample, no_think)


def load_mmlu_supervision(
    samples_paths: Union[str, Path, Sequence[Union[str, Path]]], *, no_think: bool = True
) -> List[dict]:
    """Load one or more mmlu merged_mcts_samples.json -> re_polar.router.train-ready samples.

    Each returned sample is a shallow copy of the original with `question`
    replaced by its formatted STRING form; the original structured dict is kept
    at `question_struct` (`{"question","options","category"}`) for reward-fn
    grading (`re_polar.mcts.rewards.LogLikReward` needs the dict, not the string).
    `gt_ans` (an int answer_index, unlike DART-Math's string) and
    `final_valid_transitions`/`final_invalid_transitions` pass through untouched
    -- `re_polar.router.train.build_examples` never reads `gt_ans`.
    """
    from re_polar.router.train import load_supervision_many

    raw = load_supervision_many(samples_paths)
    out: List[dict] = []
    for s in raw:
        s = dict(s)
        q = s["question"]
        if not isinstance(q, dict):
            raise ValueError(
                f"expected an mmlu question dict (question/options/category), got "
                f"{type(q)} -- is this actually DART-Math supervision?"
            )
        s["question_struct"] = q
        s["question"] = format_mmlu_question(q, no_think=no_think)
        out.append(s)
    return out
