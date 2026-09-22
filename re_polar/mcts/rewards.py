"""Reward functions for MCTS program discovery.

GenerationReward, greedy generation with the program applied, answer
checked against gt with dart-math's EvaluatorMath (vendored in
re_polar/vendor/dart_math; see NOTICE.md for provenance status). LogLikReward
is the cheaper alternative for choice-style benchmarks (mmlu_pro_domains
argmax-option correctness, ~100x cheaper).

Protocol = the paper's (Appendix D.4, verified directly against the PDF): the model is
instructed to print ONLY the boxed final answer, no chain-of-thought, no
thinking mode, max 50 new tokens. (First pilot mistakenly used free CoT at
512 tokens, different task semantics AND ~10x the cost; results discarded.)

Grading runs in a memory- and time-capped process pool (re_polar/core/grader.py):
a pathological answer (tiny to write, astronomically expensive for sympy) used
to OOM-kill the whole job. Now such a grade just returns 0 and is
appended to the fail log for inspection.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from re_polar.core import ProgramExecutor

# grader lives in a TORCH-FREE module (re_polar.core.grader) so spawn workers stay light;
# _safe_grade re-exported for tests.
from re_polar.core.grader import _grade_worker, _grade_worker_init, _safe_grade  # noqa: F401
from re_polar.core import Program

# the paper's exact instruction (Appendix D.4)
PAPER_INSTRUCTION = (
    "Solve the following math problem and output ONLY the final "
    "answer directly, formatted strictly as \\boxed{ANSWER}."
)
PAPER_MAX_NEW_TOKENS = 50
GRADE_TIMEOUT_S = 5
GRADE_MEM_BYTES = 8 * 1024**3  # per grading worker; well above legit grading,
# far below the pathological-expression blow-up
GRADE_MAX_TASKS = 64  # recycle each worker after N grades so a killed/
# timed-out worker is replaced cheaply and OS
# semaphores/FDs never accumulate (mirrors
# dart-math's pool's own anti-storm design)


def _paper_input_text(question: str) -> str:
    """The paper's EXACT prompt string (Appendix D.4 'Direct Prompting'): instruction
    + ### Problem Start/End / Answer:. No chat template, no CoT, verbatim from the PDF."""
    return (
        f"{PAPER_INSTRUCTION}\n" "### Problem Start\n" f"{question}\n" "### Problem End\n" "Answer:"
    )


def default_prompt(tokenizer, question: str) -> str:
    """Paper's exact D.4 prompt as a RAW completion (no chat template). The paper
    describes this raw string and never mentions a chat template; empirically raw is
    also best for Qwen3-8B (the chat template makes it reason in plain text and
    truncate at 50 tokens)."""
    return _paper_input_text(question)


def _qwen3_chat_wrap(tokenizer, prompt_text: str) -> str:
    """PoLar's real Qwen3 mechanism, content-agnostic (mirrors their `_qwen3_apply_
    chat_template` in polar/eval.py): single user turn, thinking DISABLED, no system
    message. Shared by every `*_chat_prompt` variant below so the wrapping logic
    lives in exactly one place, rather than duplicated per-variant."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # non-Qwen3 tokenizers don't take enable_thinking
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _qwen25_chat_sys_wrap(tokenizer, prompt_text: str) -> str:
    """PoLar's real Qwen2.5-Instruct/Qwen1.5-MoE-Chat mechanism, content-agnostic
    (mirrors their `_qwen25_apply_chat_template`/`_qwen15_moe_apply_chat_template` in
    polar/eval.py -- identical code in both, verified directly): system message +
    single user turn. Shared by every `*_chat_sys_prompt` variant below."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chat_prompt(tokenizer, question: str) -> str:
    """Same paper prompt, but wrapped in the Qwen3 chat template with thinking DISABLED
    (what PoLar's own polar/eval.py does in code). Optional; raw is the default."""
    return _qwen3_chat_wrap(tokenizer, _paper_input_text(question))


def paper_chat_sys_prompt(tokenizer, question: str) -> str:
    """The paper's exact D.4 zero-shot content (same as default_prompt/chat_prompt,
    NOT paper_minimal_fewshot's added demo), wrapped in PoLar's real Qwen2.5-Instruct/
    Qwen1.5-MoE-Chat mechanism (verified directly in their polar/eval.py).
    Together with default_prompt (llama's real setting) and chat_prompt (qwen3's real
    setting), this is the third and last mechanism PoLar's code ever applies -- the
    trio covers all 4 of their published models' real settings on their real content."""
    return _qwen25_chat_sys_wrap(tokenizer, _paper_input_text(question))


# DR.LLM's instruction (their `prompts.py`'s `answer_math` -- confirmed to be
# the one actually used for ALL their instruct models, Qwen3 included;
# `answer_math_qwen` in the same file is dead code, never called).
# Empty `\boxed{}`, unlike the paper's own `\boxed{ANSWER}` -- a likely root
# cause of our ~55% prompt-template-echo rate with this prompt. No other
# prompt convention we found (upstream
# hkust-nlp/dart-math, lm-eval-harness minerva_math) ever puts a placeholder word
# inside \boxed{} either.
DRLLM_INSTRUCTION = "The final answer MUST BE put in \\boxed{} and no explanation."


def _drllm_input_text(question: str) -> str:
    return f"Question: {question}\n{DRLLM_INSTRUCTION}"


def drllm_chat_prompt(tokenizer, question: str) -> str:
    """DR.LLM's exact prompt, chat-template-wrapped -- the ONLY way they prompt an
    instruct model (data_generation.py's `prepare_prompt`, thinking disabled for
    Qwen3): raw completion in their code is base-model-only, via a completely
    different few-shot template (`answer_math_base`), not applicable here since
    Qwen3-8B is instruct."""
    messages = [{"role": "user", "content": _drllm_input_text(question)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# DR.LLM's chat-wrapped protocol collapses on Qwen3-8B (9.0% acc, 16.5%
# has_boxed) but works exactly as designed on Qwen2.5-7B-Instruct (36.5%,
# 100%) -- these four variants each isolate ONE candidate explanation, all
# still at DR.LLM's real 15-token DART budget, zero-shot, no fewshot (the
# goal: get Qwen3 to the Qwen2.5-level result WITHOUT fewshot scaffolding).
# Invented ablations, not reproductions -- unlike drllm_chat_prompt above.


def drllm_raw_prompt(tokenizer, question: str) -> str:
    """DR.LLM's exact wording, kept as a RAW completion instead of chat-wrapped.
    Isolates the chat-template-vs-raw variable: our OWN baseline prompt already has
    an ESTABLISHED finding (default_prompt's docstring, above) that raw beats chat-
    template for Qwen3-8B in general (chat-template triggers RLHF'd conversational
    narration even against instructions); never checked whether that also holds for
    DR.LLM's specific wording."""
    return _drllm_input_text(question)


def drllm_chat_prefill_prompt(tokenizer, question: str) -> str:
    """DR.LLM's wording, chat-wrapped, but the assistant turn is PRE-SEEDED to
    literally start with `\\boxed{` before generation begins -- a standard "response
    prefill" trick for getting an instruct model to skip straight to the answer:
    physically impossible to preface with reasoning once generation is already
    continuing from inside the box. Implemented by hand (no messages/generate()
    prefill API in play here -- we call tokenizer/generate() directly): the chat-
    templated string already ends right after the assistant-turn-start markup
    (add_generation_prompt=True), so appending literal text after that IS the
    prefill."""
    messages = [{"role": "user", "content": _drllm_input_text(question)}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return templated + "\\boxed{"


DRLLM_STRICT_INSTRUCTION = (
    "Do not show any reasoning or explanation. Respond with "
    "ONLY the final answer, wrapped in \\boxed{}, and nothing else."
)


def drllm_chat_strict_prompt(tokenizer, question: str) -> str:
    """Same structure as drllm_chat_prompt, but a more forceful negative-imperative
    instruction (DR.LLM's own wording is comparatively soft: "MUST BE put in \\boxed{}
    and no explanation") -- isolates whether wording STRENGTH alone (independent of
    chat-vs-raw or prefill) changes Qwen3's compliance."""
    messages = [{"role": "user", "content": f"Question: {question}\n{DRLLM_STRICT_INSTRUCTION}"}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# DR.LLM's OWN Qwen-specific prompt (their `prompts.py`'s `answer_math_qwen`)
# -- dead code in their repo (never imported/called anywhere), but written
# with Qwen's quirks in mind by someone on their team, so
# worth testing empirically even though it's not part of their actual shipped
# protocol. Verbatim text, chat-wrapped (same convention as their real instruct-model
# path).
DRLLM_QWEN_DEADCODE_INSTRUCTION = (
    "ONLY return the final result in LaTeX with no words.\n"
    "The result MUST be wrapped inside \\boxed{...}."
)


def drllm_chat_qwen_deadcode_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "user", "content": f"Question: {question}\n{DRLLM_QWEN_DEADCODE_INSTRUCTION}"}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# A minimal fewshot nudge: ONE trivial demonstration (What is 1+1? -> \boxed{2}),
# not the 4 real MATH problems paper_fewshot/drllm_fewshot use. Cheapest possible
# fewshot nudge -- if this alone fixes Qwen3's compliance, it's a near-zero-cost
# thing to actually add to production.
MINIMAL_FEWSHOT_QUESTION = "What is 1+1?"
MINIMAL_FEWSHOT_ANSWER = "2"


def _minimal_fewshot_content(question: str) -> str:
    """Shared content builder for the whole paper_minimal_fewshot family: one worked
    demo (PAPER_INSTRUCTION's own Problem-Start/End/Answer structure) + the real
    question, as a single flat string. Factored out here rather than duplicated
    in each of the three variants below."""
    demo = _paper_input_text(MINIMAL_FEWSHOT_QUESTION) + f" \\boxed{{{MINIMAL_FEWSHOT_ANSWER}}}"
    return demo + "\n\n" + _paper_input_text(question)


def paper_minimal_fewshot_prompt(tokenizer, question: str) -> str:
    """PAPER_INSTRUCTION's own structure (same pattern as paper_fewshot_prompt) but
    with only the one trivial example, raw completion."""
    return _minimal_fewshot_content(question)


def paper_minimal_fewshot_chat_prompt(tokenizer, question: str) -> str:
    """Chat-template probe: EXACT SAME CONTENT as `paper_minimal_fewshot_
    prompt` (same demo + same real question) -- the only variable toggled is whether
    that string is wrapped in the tokenizer's own chat template before generation,
    isolating the chat-template effect from any change in prompt wording. Same
    `_qwen3_chat_wrap` mechanism `chat_prompt` above uses for the zero-shot case."""
    return _qwen3_chat_wrap(tokenizer, _minimal_fewshot_content(question))


def paper_minimal_fewshot_chat_sys_prompt(tokenizer, question: str) -> str:
    """The PoLar-LITERAL chat-wrapped variant for the model families
    whose branch in their real `polar/eval.py` prepends a system message (unlike
    Qwen3's `_qwen3_chat_wrap` -- no system message -- or LLaMA -- no chat template
    at all). `paper_minimal_fewshot_chat_prompt` above omits the system message
    entirely, which is exactly right for Qwen3 but NOT faithful for Qwen2.5-Instruct/
    Qwen1.5-MoE-Chat -- this variant fixes that, same demo+question content, only the
    system-message presence differs. Same `_qwen25_chat_sys_wrap` mechanism
    `paper_chat_sys_prompt` above uses for the zero-shot case."""
    return _qwen25_chat_sys_wrap(tokenizer, _minimal_fewshot_content(question))


def drllm_chat_minimal_fewshot_prompt(tokenizer, question: str) -> str:
    """DR.LLM's wording, chat-wrapped, with the one trivial example demonstrated as an
    ACTUAL multi-turn chat exchange (real user/assistant turns) rather than text
    stuffed into a single message -- the natural way to show a pattern to a chat-
    tuned model, and how a genuine multi-shot conversation would look."""
    messages = [
        {"role": "user", "content": _drllm_input_text(MINIMAL_FEWSHOT_QUESTION)},
        {"role": "assistant", "content": f"\\boxed{{{MINIMAL_FEWSHOT_ANSWER}}}"},
        {"role": "user", "content": _drllm_input_text(question)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def drllm_chat_minimal_fewshot_prefill_prompt(tokenizer, question: str) -> str:
    """Combines BOTH winners: drllm_chat_minimal_fewshot's one real
    demo turn PLUS drllm_chat_prefill's response-prefill trick on the real question's
    answer (the demo turn already shows a complete answer, `\\boxed{2}}` -- only the
    REAL question's response gets prefilled, since that's the one actually being
    generated)."""
    return drllm_chat_minimal_fewshot_prompt(tokenizer, question) + "\\boxed{"


# lm-evaluation-harness's minerva_math task (EleutherAI/lm-evaluation-harness,
# lm_eval/tasks/minerva_math/{minerva_math_algebra.yaml,utils.py}) -- what most
# published MATH-benchmark numbers actually come from.
# 4-shot raw completion, doc_to_text = "Problem:\n{problem}\n\nSolution:", stop
# generation at "Problem:", max_new_tokens=256 (HFLM's default max_gen_toks --
# the yaml only overrides `until`/`do_sample`/`temperature`, not this). Fewshot
# examples are utils.py's list_fewshot_samples() verbatim -- worked solutions
# ending "...\\boxed{X}.\nFinal Answer: The final answer is $X$. I hope it is
# correct.", never a placeholder. Assembly (fewshot blocks joined by "\n\n",
# question+solution joined by lm-eval-harness's default target_delimiter=" ")
# follows their documented default TaskConfig; not re-derived from source line-
# by-line, flagged here so it's checkable rather than silently assumed exact.
MINERVA_MATH_MAX_NEW_TOKENS = 256
MINERVA_MATH_STOP = "Problem:"

_MINERVA_MATH_FEWSHOT = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. "
        "Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. "
        "Also, the denominator cannot be equal to zero, so $5-x>0$, which gives "
        "$x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
        "Final Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
        "$\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})"
        "(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\n"
        "Final Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two "
        "15-pound weights instead, how many times must Terrell lift them in "
        "order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
        "$2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound "
        "weights instead for $n$ times, he will lift a total of "
        "$2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, "
        "we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n"
        "\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\n"
        "Final Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n"
        "6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are "
        "both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n"
        "$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n"
        "$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\n"
        "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]


def _minerva_doc_to_text(problem: str) -> str:
    return f"Problem:\n{problem}\n\nSolution:"


def minerva_math_prompt(tokenizer, question: str) -> str:
    """lm-eval-harness's minerva_math prompt: 4 real worked fewshot examples (never a
    placeholder) + the target question, raw completion. See MINERVA_MATH_* above for
    the matching max_new_tokens/stop-string this prompt was designed with."""
    blocks = [
        _minerva_doc_to_text(ex["problem"]) + " " + ex["solution"] for ex in _MINERVA_MATH_FEWSHOT
    ]
    blocks.append(_minerva_doc_to_text(question))
    return "\n\n".join(blocks)


# Terse-fewshot ablation: does SHOWING solved examples fix format compliance
# on its own, independent of minerva_math's ~256-token CoT cost? Same 4 real
# MATH problems as _MINERVA_MATH_FEWSHOT (kept identical across experiments
# for comparability) but with ONLY the terse boxed answer, no reasoning
# shown -- isolates "fewshot presence" from "CoT-vs-terse", and stays at our
# own 50-token MCTS budget instead of minerva's 256 (the whole point: cheap
# enough to actually use in MCTS if it works). Not a reproduction of
# anything external -- an ablation built by repeating each prompt's OWN
# existing per-question structure with a real worked answer instead of
# inventing a new format.
_TERSE_FEWSHOT_ANSWERS = ["[2,5)", "24", "16", "-\\frac{2}{3}"]  # same 4 problems, terse


def paper_fewshot_prompt(tokenizer, question: str) -> str:
    """PAPER_INSTRUCTION's own Problem-Start/End/Answer structure, repeated once per
    fewshot example (terse \\boxed{X}, no reasoning) then once more for the real
    question. Raw completion, same as default_prompt -- fewshot is a completion
    technique, not naturally a single chat turn."""
    blocks = []
    for ex, ans in zip(_MINERVA_MATH_FEWSHOT, _TERSE_FEWSHOT_ANSWERS):
        blocks.append(_paper_input_text(ex["problem"]) + f" \\boxed{{{ans}}}")
    blocks.append(_paper_input_text(question))
    return "\n\n".join(blocks)


def drllm_fewshot_prompt(tokenizer, question: str) -> str:
    """DR.LLM's own Question/instruction structure, repeated once per fewshot example
    (terse \\boxed{X}) then once more for the real question. Raw completion -- DR.LLM
    itself has no fewshot mode for instruct models, this is our own ablation, not
    their protocol; drllm_chat_prompt remains the faithful reproduction."""
    blocks = []
    for ex, ans in zip(_MINERVA_MATH_FEWSHOT, _TERSE_FEWSHOT_ANSWERS):
        blocks.append(_drllm_input_text(ex["problem"]) + f"\n\\boxed{{{ans}}}")
    blocks.append(_drllm_input_text(question))
    return "\n\n".join(blocks)


PROMPT_STYLES = {
    "raw": default_prompt,
    "chat": chat_prompt,
    "drllm_chat": drllm_chat_prompt,
    "minerva_math": minerva_math_prompt,
    "paper_fewshot": paper_fewshot_prompt,
    "drllm_fewshot": drllm_fewshot_prompt,
    "drllm_raw": drllm_raw_prompt,
    "drllm_chat_prefill": drllm_chat_prefill_prompt,
    "drllm_chat_strict": drllm_chat_strict_prompt,
    "drllm_chat_qwen_deadcode": drllm_chat_qwen_deadcode_prompt,
    "paper_minimal_fewshot": paper_minimal_fewshot_prompt,
    "paper_minimal_fewshot_chat": paper_minimal_fewshot_chat_prompt,
    "paper_minimal_fewshot_chat_sys": paper_minimal_fewshot_chat_sys_prompt,
    "paper_chat_sys": paper_chat_sys_prompt,
    "drllm_chat_minimal_fewshot": drllm_chat_minimal_fewshot_prompt,
    "drllm_chat_minimal_fewshot_prefill": drllm_chat_minimal_fewshot_prefill_prompt,
}


def truncate_after_first_boxed(text: str) -> str:
    """Cut `text` right after the FIRST complete `\\boxed{...}` span closes (brace-
    depth matching, same approach as re_polar/vendor/dart_math/eval.py's extract_boxed).

    Guards against `extract_boxed`'s `resp.split("oxed")[-1]` LAST-match behavior
    grabbing the WRONG span: any multi-block prompt (the fewshot styles above) puts
    at least one earlier `\\boxed{...}` in front of the model's own completion, and
    within its token budget the model routinely keeps generating past a correct real
    answer and hallucinates the start of a NEXT block -- often re-emitting
    PAPER_INSTRUCTION's own literal, unfilled `\\boxed{ANSWER}` placeholder. Without
    this, that later placeholder -- not the real, already-correct answer -- is what
    gets graded. No-op if `text` has 0 or 1 boxed spans (nothing to disambiguate;
    also avoids stripping legitimate trailing text in the common single-answer case).

    Found first in a standalone prompt-format investigation, but not wired
    into THIS, the production `GenerationReward` reward path, until a real
    MCTS run using `--prompt-style paper_minimal_fewshot` hit the identical
    failure at scale: diff1's round-1 (identity-only) solved count was
    7/1500 instead of the ~30%+ the earlier investigation's own 36.5%
    accuracy number implied."""
    if text.count("oxed{") < 2:
        return text
    prefix, rest = text.split("oxed{", 1)
    depth = 1
    i = 0
    for i, c in enumerate(rest):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
    return prefix + "oxed{" + rest[: i + 1]


def _answer_checker():
    # vendored (see re_polar/vendor/dart_math + NOTICE.md); upstream pip install is
    # impossible without vllm. Runtime deps: requirements-mcts.txt
    from re_polar.vendor.dart_math.eval import EvaluatorMath

    return EvaluatorMath()


def _masked_batch_layer_pass(
    layers,
    state,
    paths: List[List[int]],
    position_ids,
    cos,
    sin,
    causal_mask,
    device,
    max_group_size: Optional[int] = None,
):
    """Shared masked/gathered-batching core, used by both GenerationReward and
    LogLikReward's `masked_batch_call`: runs `state` (the residual stream, one row
    per program) through each row's own layer-index path, gathering/scattering per
    underlying layer so a layer's weights are read once per GROUP, not once per row.

    Scheduling: GREEDY LARGEST-GROUP-FIRST (replaces an earlier "flush every
    ready group every iteration" version). At each step, form every
    currently-ready group (rows whose queue-front layer index agrees) but fire ONLY
    the single largest one, then recompute from scratch. This lets a small group that
    happens to be ready early (e.g. a few rows that skipped ahead) sit and wait for
    stragglers rather than firing immediately and fragmenting into two small forward
    calls where one bigger one would do -- no tunable wait-time/threshold knobs, no
    assumption about program/segment structure, and it cannot change WHAT any row
    computes or the ORDER of layers within a row's own path, only WHICH other rows
    happen to share a given forward call (same category of effect on bf16 rounding as
    the eager version already had -- not a new fidelity risk in kind, measured
    directly against the real-search comparison this replaces/extends).
    Measured on real MCTS-discovered programs (local sim, no GPU): ~1.4-1.8x fewer
    forward calls than the eager version at round-realistic N, ~50% of the way to the
    theoretical floor.

    `max_group_size` (default None -- uncapped, original behavior unchanged): the
    largest-group-first policy deliberately maximizes how many rows share one forward
    call, which is what makes it fast but also means peak per-call memory scales with
    however many rows happen to converge on the same layer at once -- up to all of
    them (confirmed OOM at N=300 on a 40GB GPU for LogLikReward). When set, an oversized group
    is split into sequential sub-chunks of at most `max_group_size` rows each --
    bounds peak memory regardless of N, WITHOUT reducing the dataset/round size
    itself. The GROUP SELECTION (still always the largest ready group) is unchanged;
    only how that one group's rows get split across forward calls changes. Chunk
    order is arbitrary (all chunks belong to the same layer, no dependency between
    them), so this cannot affect correctness, only which OTHER rows within the same
    logical group share a given call -- same category of bf16-rounding effect as the
    scheduling change itself, not a new kind of risk.
    """
    n_rows = state.shape[0]
    row_queue = [list(p) for p in paths]
    while any(row_queue):
        groups: Dict[int, List[int]] = {}
        for i in range(n_rows):
            if row_queue[i]:
                groups.setdefault(row_queue[i][0], []).append(i)
        layer_idx, rows = max(groups.items(), key=lambda kv: len(kv[1]))
        chunks = (
            [rows[i : i + max_group_size] for i in range(0, len(rows), max_group_size)]
            if max_group_size and len(rows) > max_group_size
            else [rows]
        )
        for chunk_rows in chunks:
            idx = torch.tensor(chunk_rows, device=device)
            sub_hidden = state.index_select(0, idx)
            sub_pos_ids = position_ids.index_select(0, idx)
            sub_cos = cos.index_select(0, idx) if cos.shape[0] == n_rows else cos
            sub_sin = sin.index_select(0, idx) if sin.shape[0] == n_rows else sin
            sub_mask = (
                causal_mask.index_select(0, idx)
                if torch.is_tensor(causal_mask) and causal_mask.shape[0] == n_rows
                else causal_mask
            )
            out = layers[layer_idx](
                sub_hidden,
                attention_mask=sub_mask,
                position_ids=sub_pos_ids,
                position_embeddings=(sub_cos, sub_sin),
                use_cache=False,
            )
            if isinstance(out, tuple):
                out = out[0]
            state = state.index_copy(0, idx, out)
            for i in chunk_rows:
                row_queue[i].pop(0)
    return state


def _bucket_row_indices(lengths: List[int], n_buckets: int) -> List[List[int]]:
    """Length-bucketing for masked_batch_call:
    partitions `range(len(lengths))` into `n_buckets` groups of rows with similar
    tokenized prompt length, so padding a bucket only wastes compute up to that
    bucket's OWN length spread, not the whole round's. Without this, one long
    outlier forces every row in the round to pad to its length -- measured on
    diff5: a single 2533-char question forces all ~1500 rows to L=1687 tokens
    when the median needs ~240, and because `use_cache=False` recomputes the
    full prefix every one of 50 decode steps, that waste is paid 50x over.

    Sorts row indices by `(length, original_index)` -- the index tiebreak makes
    this a pure function of the input (no reliance on Python's sort stability
    alone, though `sorted` is stable anyway): identical `lengths` always produces
    an identical partition, across runs and machines. Splits the sorted order
    into `n_buckets` contiguous slices, sizes differing by at most 1 (the first
    `len(lengths) % n_buckets` buckets get one extra row) -- every row appears in
    exactly one bucket, none lost or duplicated, `sum(len(b) for b in buckets) ==
    len(lengths)` always. `n_buckets` is clamped to `[1, len(lengths)]` so a
    bucket count larger than the row count degrades to one row per bucket
    instead of producing empty buckets."""
    n = len(lengths)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (lengths[i], i))
    n_buckets = max(1, min(n_buckets, n))
    base, extra = divmod(n, n_buckets)
    buckets: List[List[int]] = []
    pos = 0
    for b in range(n_buckets):
        size = base + (1 if b < extra else 0)
        buckets.append(order[pos : pos + size])
        pos += size
    return buckets


def _dispatch_buckets(n_rows: int, lengths: List[int], n_buckets: int, process_bucket):
    """Bucket `range(n_rows)` by `lengths` (via `_bucket_row_indices`), call
    `process_bucket(bucket_indices) -> List[float]` once per non-empty bucket
    (bucket_indices is the length-sorted row-index list for that bucket; the
    callback must return one result per row in THAT SAME order), and reassemble
    every bucket's results into a list aligned with ORIGINAL row order -- index
    `i` of the return value is row `i`'s result regardless of which bucket it
    landed in or the order buckets were processed in. Shared by both
    `GenerationReward.masked_batch_call` and `LogLikReward.masked_batch_call`;
    `process_bucket` is where the two diverge (decode loop + grading vs. one
    forward pass + choice scoring)."""
    results: List[Optional[float]] = [None] * n_rows
    for bucket_indices in _bucket_row_indices(lengths, n_buckets):
        if not bucket_indices:
            continue
        bucket_results = process_bucket(bucket_indices)
        for local_i, orig_i in enumerate(bucket_indices):
            results[orig_i] = bucket_results[local_i]
    return results


class GenerationReward:
    """reward(program, questions, gt_answers) -> [0/1, ...] via batched greedy generation."""

    def __init__(
        self,
        executor: ProgramExecutor,
        max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
        batch_size: int = 16,
        difficulty: Optional[int] = None,
        fail_log_path: Optional[str] = None,
        grade_timeout_s: int = GRADE_TIMEOUT_S,
        grade_mem_bytes: int = GRADE_MEM_BYTES,
        grade_workers: int = 2,
        prompt_style: str = "raw",
        masked_batch_max_group_size: Optional[int] = None,
        masked_batch_buckets: int = 8,
        text_log_path: Optional[str] = None,
    ):
        self.executor = executor
        self.tokenizer = executor.engine.tokenizer
        self.device = executor.engine.device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.difficulty = difficulty
        # caps peak per-call memory in masked_batch_call's greedy-largest-group
        # scheduling (see _masked_batch_layer_pass docstring) -- default None =
        # uncapped, unchanged behavior.
        self.masked_batch_max_group_size = masked_batch_max_group_size
        # length-bucketing: masked_batch_call
        # splits a round into this many length-sorted buckets (`_bucket_row_indices`)
        # instead of padding every row to the round's single longest prompt.
        # Default 8: diff5's measured length distribution is heavy-tailed (p95=604
        # chars, max=2533) -- with 8 roughly-equal-COUNT buckets the top bucket
        # holds ~12.5% of rows, comfortably isolating the p95+ outliers from the
        # bulk, without an excessive multiplication of the decode loop's per-step
        # fixed overhead (each bucket pays its own embed/rotary/mask setup for all
        # max_new_tokens steps, so more buckets = more redundant setup work, not
        # just finer length-matching). 1 = no bucketing (old behavior, one padded
        # batch for the whole round) -- the degenerate case `_bucket_row_indices`
        # already handles, not a separate code path.
        self.masked_batch_buckets = masked_batch_buckets
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {sorted(PROMPT_STYLES)}")
        self._prompt_fn = PROMPT_STYLES[prompt_style]
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"  # generation needs left padding
        self.grade_timeout_s = grade_timeout_s
        self._grade_mem_bytes = grade_mem_bytes
        self._grade_workers = grade_workers
        self._pool = None
        self.fail_log_path = Path(fail_log_path) if fail_log_path else None
        if self.fail_log_path:
            self.fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Opt-in generated-answer + batch-composition log (re_polar/mcts/textlog.py).
        # Written from _grade_batch, so it covers BOTH the serial __call__ path and
        # masked_batch_call with one hook. None (default) = not written at all.
        from re_polar.mcts.textlog import TextLog

        self.text_log = TextLog(text_log_path) if text_log_path else None

    def _new_pool(self, workers: int):
        import multiprocessing as mp
        from pebble import ProcessPool

        # spawn (not fork): grading workers must be free of this process's CUDA
        # context so their RLIMIT_AS cap actually bounds them; re_polar.core.grader is
        # torch-free so spawn stays cheap.
        return ProcessPool(
            max_workers=workers,
            max_tasks=GRADE_MAX_TASKS,
            context=mp.get_context("spawn"),
            initializer=_grade_worker_init,
            initargs=(self._grade_mem_bytes,),
        )

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = self._new_pool(self._grade_workers)
        return self._pool

    def _reset_pool(self):
        if self._pool is not None:
            try:
                self._pool.stop()
            except Exception:
                pass
        self._pool = None

    def _log_fail(
        self, question: str, ref: str, resp_text: str, program: Program, failure: str
    ) -> None:
        if not self.fail_log_path:
            return
        rec = {
            "failure": failure,
            "difficulty": self.difficulty,
            "question": question,
            "gt_ans": ref,
            "generated_answer": resp_text,
            "program": program.to_layer_path(),
        }
        with open(self.fail_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _grade_batch(
        self, refs: List[str], texts: List[str], questions: List[str], program
    ) -> List[float]:
        """Grade a batch in ONE persistent, worker-recycling pool.

        `program`: a single Program (fail-log label for the whole batch, the
        original/only shape) OR a List[Program] one-per-row (masked_batch_call,
        where a batch can span multiple distinct programs) -- resolved to a
        per-row label below; every existing single-Program caller is untouched.

        Each answer is an independent task with a time + memory cap. A
        pathological answer (a giant expression that slips past the boxed gate)
        times out or hits the memory cap IN ITS OWN WORKER; Pebble recycles that
        worker (max_tasks) and the task yields reward 0 + a logged sample, while
        sibling grades keep running. Verified on a real corpus of pathological
        explosion answers plus synthetic boxed-giants: 0 broken pools, 0 OSErrors.

        An earlier version recovered from a pool break by spawning a fresh
        single-worker pool PER leftover answer, which leaked OS semaphores/FDs
        until `spawn` itself failed with OSError -> a real incident where tens
        of thousands of grades were falsely scored 0 and cached. Here pool
        creation is bounded to at most TWO pools per batch: the persistent one,
        plus one rebuild if the whole pool genuinely breaks. A break only
        RETRIES the collateral (its exception name carries "Broken"/OSError); a
        contained per-answer timeout/expiry is scored 0 directly, never re-run.

        GRADING TEXT vs. LOGGED TEXT: `texts` (the raw model output) is
        what gets written to `text_log`/`_log_fail` for auditability, but grading
        itself runs on `truncate_after_first_boxed(texts[i])` -- see that function's
        docstring for why (a hallucinated continuation past a correct real answer
        must not let `extract_boxed`'s last-match behavior grab a later, wrong span
        instead). No-op for any text with <=1 boxed span, so this only ever changes
        the previously-broken multi-span cases."""
        n = len(refs)
        programs = program if isinstance(program, list) else [program] * n
        graded_texts = [truncate_after_first_boxed(t) for t in texts]
        results: List[Optional[float]] = [None] * n
        remaining = list(range(n))
        for attempt in range(2):  # persistent pool, then at most ONE fresh pool
            if not remaining:
                break
            if attempt == 1:
                self._reset_pool()  # discard a genuinely broken pool; rebuild once
            pool = self._ensure_pool()
            futures: Dict[int, object] = {}
            try:
                for i in remaining:
                    futures[i] = pool.schedule(
                        _grade_worker, args=(refs[i], graded_texts[i]), timeout=self.grade_timeout_s
                    )
            except Exception:
                pass  # pool broke while scheduling -> unscheduled stay in `remaining`
            still: List[int] = []
            for i in remaining:
                fut = futures.get(i)
                if fut is None:
                    still.append(i)
                    continue
                try:
                    results[i] = 1.0 if fut.result() else 0.0
                except Exception as e:
                    name = type(e).__name__
                    pool_broke = ("Broken" in name) or isinstance(e, OSError)
                    if pool_broke and attempt == 0:
                        still.append(i)  # collateral of a pool break -> retry, don't mislabel
                    else:
                        results[i] = 0.0  # contained (timeout/expiry) or final -> genuine 0
                        self._log_fail(questions[i], refs[i], graded_texts[i], programs[i], name)
            remaining = still
        for i in remaining:  # defensive: unresolved after the rebuild -> 0 + logged
            results[i] = 0.0
            self._log_fail(questions[i], refs[i], graded_texts[i], programs[i], "unresolved")
        final = [r if r is not None else 0.0 for r in results]
        if self.text_log is not None:
            # every generation path (serial __call__, masked_batch_call,
            # sample_pass_at_k) funnels through here, and `questions`/`texts` are
            # this batch in ITS OWN ORDER -- so one record here captures both the
            # generated answers and the exact batch composition that produced them
            # (re_polar/mcts/textlog.py explains why the composition matters).
            self.text_log.write_batch(
                questions,
                refs,
                texts,
                final,
                [p.to_layer_path() for p in programs],
                difficulty=self.difficulty,
            )
        return final

    def __call__(
        self, program: Program, questions: List[str], gt_answers: List[str]
    ) -> List[float]:
        prompts = [self._prompt_fn(self.tokenizer, q) for q in questions]
        rewards: List[float] = []
        with self.executor.apply(program) as model:
            for i in range(0, len(prompts), self.batch_size):
                chunk = prompts[i : i + self.batch_size]
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )  # no chat template
                texts = self.tokenizer.batch_decode(
                    out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                rewards.extend(
                    self._grade_batch(
                        gt_answers[i : i + self.batch_size],
                        texts,
                        questions[i : i + self.batch_size],
                        program,
                    )
                )
        return rewards

    def masked_batch_call(
        self, programs: List[Program], questions: List[str], gt_answers: List[str]
    ) -> List[float]:
        """Cross-program masked/gathered batching: ONE shared forward pass per decode
        step across ALL rows regardless of which DISTINCT program each row uses,
        instead of one `__call__` (== one `executor.apply(program)` + cached
        `generate()`) per distinct program. Validated standalone against an
        uncached control (bit-identical) and benchmarked separately: speedup
        scales with row count (0.79x @ N=4 -> 1.95x @ N=128 synthetic; 2.7-3.0x
        @ N=256/512 on REAL MCTS-search-derived programs), at a measured
        **8.6-8.8% verdict-flip-rate cost vs. the cached serial `__call__` path** on
        real programs (root cause: `use_cache=False` full-prefix recompute every
        step diverges from `use_cache=True` at the bf16-non-associativity level for
        repeat-heavy/deep programs -- the same bf16 non-associativity that makes
        greedy decoding diverge across GPU architectures, not a bug in the
        gather/scatter mechanism itself, which was proven bit-identical to an
        uncached control).

        `programs`/`questions`/`gt_answers` are PARALLEL arrays, one entry per row
        -- unlike `__call__`, `programs[i]` may differ per row (that's the whole
        point: many distinct programs share one forward pass). Rows do NOT go
        through `executor.apply()` (that mutates the model's layer list in place,
        one program at a time -- incompatible with concurrent distinct programs);
        instead each row's layers are looked up directly via `re_polar.core.layer_engine.
        get_layers` and gathered/scattered per layer, leaving the model's actual
        layer list untouched throughout.

        Grouping/scheduling: `_masked_batch_layer_pass` (module-level, shared with
        `LogLikReward.masked_batch_call` below) -- GREEDY LARGEST-GROUP-FIRST
        (replaces an earlier "flush every ready group every iteration" version;
        see that function's docstring for the mechanism and measured ~1.4-1.8x
        fewer forward calls on real programs).

        **OPT-IN ONLY.** Never called by `__call__`/the default MCTSRunner path --
        only reachable via `MCTSRunner(masked_batch_reward_fn=...)`
        (`re_polar/mcts/scheduler.py::_generate_masked_batch`). Given the flip-rate
        cost above, callers should treat this as a measured speed/fidelity
        tradeoff, not a drop-in replacement for `__call__`.

        LENGTH BUCKETING: the round is split into
        `self.masked_batch_buckets` length-sorted buckets (`_bucket_row_indices`)
        and each bucket runs its OWN independent decode loop
        (`_masked_batch_decode_bucket`) -- a long outlier only forces padding on
        the rows sharing ITS bucket, not the whole round. `_grade_batch` (and
        therefore the text log, `re_polar/mcts/textlog.py`) is called ONCE PER BUCKET
        with that bucket's own (questions, texts, programs) in bucket-local order
        -- the textlog's batch-composition record stays honest by construction,
        since a logged batch IS a bucket that genuinely shared a forward call.
        `_dispatch_buckets` reassembles the final return value into ORIGINAL row
        order regardless of bucket processing order -- callers see no difference
        in shape or ordering from the pre-bucketing contract."""
        from re_polar.core.layer_engine import get_layers

        model = self.executor.engine.model
        device = self.device
        inner = model.model
        layers = get_layers(model)
        n_rows = len(programs)
        paths = [p.to_layer_path() for p in programs]
        prompts = [self._prompt_fn(self.tokenizer, q) for q in questions]
        lengths = [len(self.tokenizer.encode(p)) for p in prompts]

        def process_bucket(bucket_indices: List[int]) -> List[float]:
            b_paths = [paths[i] for i in bucket_indices]
            b_prompts = [prompts[i] for i in bucket_indices]
            texts = self._masked_batch_decode_bucket(
                b_paths, b_prompts, layers, inner, model, device
            )
            b_programs = [programs[i] for i in bucket_indices]
            b_questions = [questions[i] for i in bucket_indices]
            b_gt = [gt_answers[i] for i in bucket_indices]
            return self._grade_batch(b_gt, texts, b_questions, b_programs)

        return _dispatch_buckets(n_rows, lengths, self.masked_batch_buckets, process_bucket)

    def _masked_batch_decode_bucket(
        self, paths: List[List[int]], prompts: List[str], layers, inner, model, device
    ) -> List[str]:
        """One length-bucket's worth of the masked-batch decode loop -- extracted
        from `masked_batch_call` so bucketing can call it once per bucket instead
        of once for the whole round. Mechanics unchanged from the pre-bucketing
        version (greedy largest-group-first scheduling via `_masked_batch_layer_pass`,
        slice-before-norm): bucketing only changes WHICH rows get padded together
        before this runs, not how a given set of rows is processed once inside it."""
        import torch
        from transformers.masking_utils import create_causal_mask

        n_rows = len(paths)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        seq = enc["input_ids"].clone()
        pad_mask = enc["attention_mask"].clone()
        generated: List[List[int]] = [[] for _ in range(n_rows)]

        with torch.no_grad():
            for _step in range(self.max_new_tokens):
                L = seq.shape[1]
                n_pad = (pad_mask == 0).sum(dim=1)
                position_ids = (
                    torch.arange(L, device=device).unsqueeze(0) - n_pad.unsqueeze(1)
                ).clamp(min=0)
                hidden = inner.embed_tokens(seq)
                cos, sin = inner.rotary_emb(hidden, position_ids)
                causal_mask = create_causal_mask(
                    config=model.config,
                    inputs_embeds=hidden,
                    attention_mask=pad_mask,
                    past_key_values=None,
                    position_ids=position_ids,
                )

                state = _masked_batch_layer_pass(
                    layers,
                    hidden,
                    paths,
                    position_ids,
                    cos,
                    sin,
                    causal_mask,
                    device,
                    max_group_size=self.masked_batch_max_group_size,
                )
                del hidden  # ~n_rows*L*hidden bf16, not needed past the layer pass

                # SLICE BEFORE NORM, not after. Bit-identical, not a numerics
                # change; purely a
                # memory fix (bucketing does not touch or interact with it -- each
                # bucket still slices its own `state` before norm).
                final = inner.norm(state[:, -1, :])
                logits = model.lm_head(final)
                next_tok = logits.argmax(-1)
                for i in range(n_rows):
                    generated[i].append(next_tok[i].item())
                seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
                pad_mask = torch.cat(
                    [pad_mask, torch.ones(n_rows, 1, dtype=pad_mask.dtype, device=device)], dim=1
                )

        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

    def passk(
        self,
        program: Program,
        questions: List[str],
        gt_answers: List[str],
        *,
        k: int,
        temperature: float,
    ) -> List[float]:
        """pass@k for a program via k STOCHASTIC samples per question.

        Mirrors ``__call__`` exactly (same prompt, program-apply, grading) but with
        ``do_sample=True, temperature`` and ``num_return_sequences=k``, then reduces
        each question's k samples to 1.0 iff ANY is correct (pass@k). Pure
        temperature sampling (``top_p=1.0, top_k=0``), the literal reading of the
        paper's "stochastic decoding with temperature τ". Greedy pass@1 stays
        ``__call__``; this is only the Base(sampling) baseline.

        Thin wrapper over :meth:`passk_curve`, ``passk_curve(..., k_max=k)[k]``.
        Kept for callers that only want one k (the curve costs the same either way,
        since it's derived from the same k_max generations).
        """
        return self.passk_curve(program, questions, gt_answers, k_max=k, temperature=temperature)[k]

    def passk_curve(
        self,
        program: Program,
        questions: List[str],
        gt_answers: List[str],
        *,
        k_max: int,
        temperature: float,
    ) -> Dict[int, List[float]]:
        """pass@k for EVERY k in 1..k_max, from ONE batch of k_max samples/question.

        An earlier version of this method (``passk``) only ever returned the
        single reduced-at-k boolean, so a caller sweeping k=1..5 (to reproduce
        the paper's per-k table, e.g. Table 7's "Base (sampling)" rows
        1/2/3/4/5) would need k_max GENERATION CALLS -- wasteful, and a real
        bug in an early baseline run that never called with k<5 at all, so it
        never produced the sampling pass@1 number -- a real, distinct number
        from Base (τ=0)'s greedy pass@1, confirmed against the paper's own
        Table 7: 37.0 vs 41.6 on DM-1, NOT the same baseline).

        Fix: generate k_max samples per question ONCE, then take prefix-of-length-k
        ("any of the FIRST k of the k_max samples correct") for every k in 1..k_max.
        Since all k_max samples are i.i.d. at this temperature, the prefix is a
        valid (if slightly correlated vs resampling fresh per k) pass@k estimate --
        standard practice for reporting a pass@k curve from one generation batch,
        and zero extra generation cost vs the single-k version.
        """
        prompts = [self._prompt_fn(self.tokenizer, q) for q in questions]
        n = len(questions)
        per_question_correct: List[List[float]] = [[] for _ in range(n)]  # [i] = k_max flags
        eff_batch = max(1, self.batch_size // max(1, k_max))
        with self.executor.apply(program) as model:
            for i in range(0, len(prompts), eff_batch):
                chunk = prompts[i : i + eff_batch]
                q_chunk = questions[i : i + eff_batch]
                g_chunk = gt_answers[i : i + eff_batch]
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=1.0,
                        top_k=0,
                        num_return_sequences=int(k_max),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                texts = self.tokenizer.batch_decode(
                    out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                flat_refs = [g_chunk[j] for j in range(len(chunk)) for _ in range(k_max)]
                flat_qs = [q_chunk[j] for j in range(len(chunk)) for _ in range(k_max)]
                flat_rewards = self._grade_batch(flat_refs, texts, flat_qs, program)
                for j in range(len(chunk)):
                    per_question_correct[i + j] = flat_rewards[j * k_max : (j + 1) * k_max]
        curve: Dict[int, List[float]] = {}
        for k in range(1, k_max + 1):
            curve[k] = [
                1.0 if any(r >= 1.0 for r in flags[:k]) else 0.0 for flags in per_question_correct
            ]
        return curve


class LogLikReward:
    """reward(program, questions, gt_answers) -> [0/1, ...] via ONE log-likelihood
    forward pass per batch, no generation loop, no subprocess grading.

    Reuses re_polar.core.mmlu_pro_domain_eval.run_mmlu_pro_domains's argmax-over-
    option-logit scoring directly (not reimplemented). ~100x cheaper than
    GenerationReward (one forward pass vs PAPER_MAX_NEW_TOKENS autoregressive
    steps + sympy grading), which is what makes 27-32B MCTS feasible.

    Mirrors GenerationReward's call signature, adapted to multiple-choice:
    `questions` is a list of dicts carrying the mmlu_pro_domains record fields
    minus the answer -- {"question": str, "options": [str, ...], "category":
    str (optional)}; `gt_answers` is the parallel list of answer_index ints.
    This is exactly the shape MCTSRunner passes through (inputs[i]["question"]
    / inputs[i]["gt_ans"]), so an mmlu_pro_domains input dict plugs into the
    same scheduler unchanged.

    `probs_log_path` (default None -- OFF): when set, additionally writes one
    JSONL line per (query, program) to that path with the FULL softmax
    probability distribution over answer choices (not just the argmax), via
    run_mmlu_pro_domains's `return_probs` (re_polar/core/mmlu_pro_domain_eval.py).
    Mirrors GenerationReward._log_fail's side-log pattern above
    -- a pure side effect. The __call__ RETURN VALUE is unaffected either way
    (still exactly [p["score"], ...], same order): MCTSRunner/ProgramMCTS/
    EvalCache/the scheduler's whole reward-fn contract need NO changes, so
    this cannot alter search behavior even if probs logging is on. `questions`
    entries may optionally carry a "query_id" key (purely for this log --
    scoring itself only reads question/options/category, unaffected by extra
    keys); falls back to the question text if absent, e.g. in tests that don't
    set one.
    """

    def __init__(
        self,
        executor: ProgramExecutor,
        batch_size: int = 8,
        no_think: bool = True,
        difficulty: Optional[str] = None,
        probs_log_path: Optional[str] = None,
        masked_batch_max_group_size: Optional[int] = None,
        masked_batch_buckets: int = 8,
    ):
        self.executor = executor
        self.tokenizer = executor.engine.tokenizer
        self.device = executor.engine.device
        self.batch_size = batch_size
        self.no_think = no_think
        self.difficulty = difficulty  # provenance only; scoring doesn't branch on it
        self.probs_log_path = Path(probs_log_path) if probs_log_path else None
        if self.probs_log_path:
            self.probs_log_path.parent.mkdir(parents=True, exist_ok=True)
        # caps peak per-call memory in masked_batch_call's greedy-largest-group
        # scheduling -- default None = uncapped. Added after an OOM at
        # N=300 on a 40GB GPU (a single group grew large enough to exceed budget;
        # see _masked_batch_layer_pass docstring). Especially relevant here since
        # LogLikReward's serial __call__ already internally chunks at batch_size
        # (default 8) -- masked_batch_call had no equivalent cap until this.
        # `masked_batch_buckets`: see GenerationReward.__init__ for the default-8
        # reasoning -- identical here, just a single forward pass per bucket
        # instead of a decode loop per bucket.
        self.masked_batch_max_group_size = masked_batch_max_group_size

    def __call__(
        self, program: Program, questions: List[Dict], gt_answers: List[int]
    ) -> List[float]:
        if not questions:
            return []
        # lazy: keeps this module importable without pulling in benchmarks.*
        # until a LogLikReward is actually called, mirrors _answer_checker()'s
        # lazy EvaluatorMath import above.
        from re_polar.core.mmlu_pro_scoring import MMLUProSample
        from re_polar.core.mmlu_pro_domain_eval import run_mmlu_pro_domains

        samples = [
            MMLUProSample(
                id=i,
                question=str(q["question"]),
                options=tuple(q["options"]),
                answer_index=int(a),
                category=str(q.get("category", "")),
            )
            for i, (q, a) in enumerate(zip(questions, gt_answers))
        ]
        with self.executor.apply(program) as model:
            result = run_mmlu_pro_domains(
                model,
                self.tokenizer,
                samples=samples,
                batch_size=self.batch_size,
                no_think=self.no_think,
                return_details=True,
                return_probs=self.probs_log_path is not None,
            )
        # per_prompt preserves input order (run_mmlu_pro_domains iterates items
        # in the given order and appends sequentially) -> 1:1 with `questions`.
        if self.probs_log_path:
            self._log_probs(program, questions, result["per_prompt"])
        return [p["score"] for p in result["per_prompt"]]

    def _log_probs(self, program: Program, questions: List[Dict], per_prompt: List[Dict]) -> None:
        path = program.to_layer_path()
        with open(self.probs_log_path, "a") as f:
            for q, entry in zip(questions, per_prompt):
                f.write(
                    json.dumps(
                        {
                            "query_id": q.get("query_id", q["question"]),
                            "path": path,
                            "predicted": entry["predicted"],
                            "score": entry["score"],
                            "probs": entry["probs"],
                        }
                    )
                    + "\n"
                )

    def masked_batch_call(
        self, programs: List[Program], questions: List[Dict], gt_answers: List[int]
    ) -> List[float]:
        """Cross-program masked/gathered batching for LogLikReward.

        Same gather/scatter mechanism as `GenerationReward.masked_batch_call`
        (shared core: `_masked_batch_layer_pass`, greedy largest-group-first
        scheduling), but ONE forward pass, no decode loop -- LogLikReward's serial
        baseline (`run_mmlu_pro_domains`, re_polar/core/mmlu_pro_domain_eval.py) is
        ALREADY `use_cache=False` even without masked-batch,
        since it never generates, it just reads `logits[:, -1, :]` once. That means
        masked-batching this reward carries NONE of GenerationReward's cache-vs-
        no-cache fidelity tradeoff (there is no cache being dropped -- nothing to
        diverge from) -- expected to be a much closer-to-risk-free speedup than the
        generation case, still worth validating empirically before trusting it the
        same way. Only the greedy largest-group-first
        scheduler is implemented here (no separate naive/eager version), since the
        eager version was already shown suboptimal on real programs before this
        method was written.

        `programs`/`questions`/`gt_answers` are PARALLEL arrays, one entry per row --
        same contract as `GenerationReward.masked_batch_call`. `questions[i]` is the
        mmlu_pro_domains dict shape ({"question", "options", "category", optionally
        "query_id"}), `gt_answers[i]` the answer_index int -- identical to `__call__`.

        **OPT-IN ONLY.** Never called by `__call__`/the default MCTSRunner path --
        only reachable via `MCTSRunner(masked_batch_reward_fn=...)`.

        LENGTH BUCKETING: same mechanism as
        `GenerationReward.masked_batch_call`, simpler here since there is no
        decode loop -- each length-bucket gets its own single forward pass
        (`_left_pad` + layer-pass + choice scoring) instead of one padded pass
        for the whole round. `_dispatch_buckets` reassembles into original row
        order. No text log here (LogLikReward has no generated text to log --
        `--log-probs` is per-row already, no batch-composition claim to keep
        honest the way `re_polar/mcts/textlog.py` does for GenerationReward)."""
        from re_polar.core.mmlu_pro_scoring import MMLUProSample, _build_prompt, _choice_token_ids, _left_pad
        from re_polar.core.layer_engine import get_layers
        from transformers.masking_utils import create_causal_mask

        model = self.executor.engine.model
        device = self.device
        inner = model.model
        layers = get_layers(model)
        paths = [p.to_layer_path() for p in programs]

        samples = [
            MMLUProSample(
                id=i,
                question=str(q["question"]),
                options=tuple(q["options"]),
                answer_index=int(a),
                category=str(q.get("category", "")),
            )
            for i, (q, a) in enumerate(zip(questions, gt_answers))
        ]
        prepared = []
        for s in samples:
            enc = self.tokenizer(_build_prompt(s, self.no_think), return_tensors="pt")
            prepared.append(
                {
                    "id": s.id,
                    "category": s.category,
                    "answer_index": s.answer_index,
                    "num_choices": len(s.options),
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                }
            )
        lengths = [p["input_ids"].shape[0] for p in prepared]
        max_choices = 10
        all_choice_ids = _choice_token_ids(self.tokenizer, max_choices)

        def process_bucket(bucket_indices: List[int]) -> List[float]:
            b_prepared = [prepared[i] for i in bucket_indices]
            b_paths = [paths[i] for i in bucket_indices]
            tensors = _left_pad(b_prepared, int(self.tokenizer.pad_token_id), device)
            seq = tensors["input_ids"]
            pad_mask = tensors["attention_mask"]

            with torch.no_grad():
                L = seq.shape[1]
                n_pad = (pad_mask == 0).sum(dim=1)
                position_ids = (
                    torch.arange(L, device=device).unsqueeze(0) - n_pad.unsqueeze(1)
                ).clamp(min=0)
                hidden = inner.embed_tokens(seq)
                cos, sin = inner.rotary_emb(hidden, position_ids)
                causal_mask = create_causal_mask(
                    config=model.config,
                    inputs_embeds=hidden,
                    attention_mask=pad_mask,
                    past_key_values=None,
                    position_ids=position_ids,
                )
                state = _masked_batch_layer_pass(
                    layers,
                    hidden,
                    b_paths,
                    position_ids,
                    cos,
                    sin,
                    causal_mask,
                    device,
                    max_group_size=self.masked_batch_max_group_size,
                )
                del hidden  # ~n_rows*L*hidden bf16, not needed past the layer pass
                # slice-before-norm: see the identical fix in GenerationReward.
                # masked_batch_call for why this is bit-identical and why it
                # matters. LogLikReward is single-step so the saving is one-shot
                # rather than per-decode-step, but it is the same allocation and
                # the same argument. Bucketing doesn't interact with it -- each
                # bucket still slices its own `state` before norm.
                final = inner.norm(state[:, -1, :])
                logits = model.lm_head(final)

            bucket_rewards: List[float] = []
            bucket_probs: List[Dict] = []
            for local_i, item in enumerate(b_prepared):
                n = item["num_choices"]
                choice_tensor = torch.tensor(all_choice_ids[:n], device=device)
                choice_logits = logits[local_i][choice_tensor]
                pred = choice_logits.argmax().item()
                score = 1.0 if pred == item["answer_index"] else 0.0
                bucket_rewards.append(score)
                if self.probs_log_path:
                    bucket_probs.append(
                        {
                            "predicted": pred,
                            "score": score,
                            "probs": choice_logits.softmax(dim=-1).tolist(),
                        }
                    )

            if self.probs_log_path:
                b_questions = [questions[i] for i in bucket_indices]
                with open(self.probs_log_path, "a") as f:
                    for q, path, entry in zip(b_questions, b_paths, bucket_probs):
                        f.write(
                            json.dumps(
                                {
                                    "query_id": q.get("query_id", q["question"]),
                                    "path": path,
                                    **entry,
                                }
                            )
                            + "\n"
                        )
            return bucket_rewards

        return _dispatch_buckets(len(programs), lengths, self.masked_batch_buckets, process_bucket)
