"""PROMPT_STYLES content checks (re_polar/mcts/rewards.py) -- pure string logic,
no torch/model needed. Guards each prompt variant's exact wording against
silent drift."""
from re_polar.mcts.rewards import (
    DRLLM_QWEN_DEADCODE_INSTRUCTION,
    DRLLM_STRICT_INSTRUCTION,
    MINERVA_MATH_STOP,
    MINIMAL_FEWSHOT_ANSWER,
    MINIMAL_FEWSHOT_QUESTION,
    PAPER_INSTRUCTION,
    PROMPT_STYLES,
    default_prompt,
    drllm_chat_minimal_fewshot_prefill_prompt,
    drllm_chat_minimal_fewshot_prompt,
    drllm_chat_prefill_prompt,
    drllm_chat_prompt,
    drllm_chat_qwen_deadcode_prompt,
    drllm_chat_strict_prompt,
    drllm_fewshot_prompt,
    drllm_raw_prompt,
    minerva_math_prompt,
    paper_chat_sys_prompt,
    paper_fewshot_prompt,
    paper_minimal_fewshot_chat_prompt,
    paper_minimal_fewshot_chat_sys_prompt,
    paper_minimal_fewshot_prompt,
)


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's apply_chat_template, records the messages
    it was called with so tests can assert on content without loading a real model."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        # joins every message's content (not just the first) so multi-turn prompts
        # (e.g. drllm_chat_minimal_fewshot_prompt) can be inspected too; single-
        # message callers get exactly messages[0]["content"] as before
        return "\n---\n".join(m["content"] for m in messages)


def test_prompt_styles_registry_has_expected_variants():
    assert set(PROMPT_STYLES) == {
        "raw", "chat", "drllm_chat", "minerva_math", "paper_fewshot", "drllm_fewshot",
        "drllm_raw", "drllm_chat_prefill", "drllm_chat_strict", "drllm_chat_qwen_deadcode",
        "paper_minimal_fewshot", "paper_minimal_fewshot_chat", "paper_minimal_fewshot_chat_sys",
        "paper_chat_sys",
        "drllm_chat_minimal_fewshot", "drllm_chat_minimal_fewshot_prefill",
    }


def test_paper_chat_sys_prompt_uses_zeroshot_content_not_the_fewshot_demo():
    """paper_chat_sys: PoLar's real Qwen2.5-Instruct/Qwen1.5-MoE-Chat system
    message, but on the paper's actual zero-shot content (default_prompt's
    _paper_input_text), NOT paper_minimal_fewshot's added demo -- the two
    "chat_sys" variants must differ only in content, same system message
    wrapping."""
    tok = _FakeTokenizer()
    chat_sys = paper_chat_sys_prompt(tok, "What is 2+2?")
    raw = default_prompt(tok, "What is 2+2?")
    assert chat_sys == "You are a helpful assistant.\n---\n" + raw
    # confirms it's NOT the fewshot content
    assert MINIMAL_FEWSHOT_QUESTION not in chat_sys


def test_paper_minimal_fewshot_chat_same_content_as_raw_variant():
    """The whole point of this variant: identical CONTENT to
    paper_minimal_fewshot_prompt, only the chat-template wrapping differs."""
    tok = _FakeTokenizer()
    raw = paper_minimal_fewshot_prompt(tok, "What is 2+2?")
    chat = paper_minimal_fewshot_chat_prompt(tok, "What is 2+2?")
    # _FakeTokenizer.apply_chat_template just joins message content back out --
    # so the chat-wrapped single-user-turn content should equal the raw string.
    assert chat == raw


def test_paper_minimal_fewshot_chat_sys_adds_polars_exact_system_message():
    """PoLar-literal variant for the Qwen2.5-Instruct/Qwen1.5-MoE-Chat branches
    of their real polar/eval.py (_qwen25_apply_chat_template/_qwen15_moe_apply_
    chat_template both prepend this exact system message, verified against
    their code). Same demo+question content as the no-sys variant, system
    message is the only difference."""
    tok = _FakeTokenizer()
    chat_sys = paper_minimal_fewshot_chat_sys_prompt(tok, "What is 2+2?")
    raw = paper_minimal_fewshot_prompt(tok, "What is 2+2?")
    assert chat_sys == "You are a helpful assistant.\n---\n" + raw


def test_paper_prompt_uses_boxed_answer_placeholder():
    # PAPER_INSTRUCTION's exact wording uses a literal "ANSWER" placeholder
    # inside \boxed{}, pin it so a future edit can't silently change it.
    assert "\\boxed{ANSWER}" in PAPER_INSTRUCTION
    assert "\\boxed{ANSWER}" in default_prompt(None, "What is 2+2?")


def test_drllm_chat_prompt_uses_empty_boxed_not_a_placeholder_word():
    # DR.LLM has no raw-completion mode for instruct models -- chat-wrapped is the
    # only faithful reproduction (see drllm_chat_prompt's docstring).
    text = drllm_chat_prompt(_FakeTokenizer(), "What is 2+2?")
    assert "\\boxed{}" in text
    assert "ANSWER" not in text
    assert "What is 2+2?" in text


def test_minerva_math_prompt_has_four_real_worked_fewshot_examples():
    text = minerva_math_prompt(None, "What is 2+2?")
    assert text.count("\\boxed{") == 4  # 4 fewshot solutions, one \boxed{} each; target has none yet
    assert "ANSWER" not in text
    assert text.endswith("Problem:\nWhat is 2+2?\n\nSolution:")
    # every fewshot example ends with the "I hope it is correct." tell -- never a
    # bare placeholder.
    assert text.count("I hope it is correct.") == 4


def test_minerva_math_stop_string_matches_its_doc_to_text_prefix():
    # the stop string must actually appear at the start of each fewshot block's own
    # doc_to_text, or truncate_at_stop (prompt_variant_pilot.py) would cut nothing.
    assert MINERVA_MATH_STOP == "Problem:"


def test_paper_fewshot_prompt_never_demonstrates_the_placeholder_as_a_real_answer():
    text = paper_fewshot_prompt(None, "What is 2+2?")
    # PAPER_INSTRUCTION itself says "...formatted strictly as \boxed{ANSWER}." and is
    # repeated once per block (4 fewshot + 1 target) -- that's expected/harmless, it's
    # describing the format, not demonstrating a filled-in answer. What must NEVER
    # happen is a fewshot example's ACTUAL answer (right after "Answer:") being the
    # placeholder itself -- each of the 4 real answers is a genuine value.
    assert text.count("ANSWER") == 5  # one per repeated instruction, not a bug
    assert "Answer: \\boxed{ANSWER}" not in text  # never demonstrated as a real answer
    assert text.count("\\boxed{[2,5)}") == 1
    assert text.count("\\boxed{24}") == 1
    assert text.count("\\boxed{16}") == 1
    assert text.count("\\boxed{-\\frac{2}{3}}") == 1
    # each fewshot example repeats the paper's OWN Problem-Start/End/Answer structure
    assert text.count("### Problem Start") == 5  # 4 fewshot + 1 target
    assert text.count(PAPER_INSTRUCTION) == 5
    assert text.endswith(
        f"{PAPER_INSTRUCTION}\n### Problem Start\nWhat is 2+2?\n### Problem End\nAnswer:")


def test_drllm_fewshot_prompt_shows_terse_real_answers():
    text = drllm_fewshot_prompt(None, "What is 2+2?")
    # DRLLM_INSTRUCTION itself mentions empty "\boxed{}" once per block (5 blocks) --
    # the 4 REAL terse answers are the additional occurrences on their own line.
    assert text.count("\\boxed{") == 9
    assert text.count("\\boxed{[2,5)}") == 1
    assert text.count("\\boxed{24}") == 1
    assert text.count("\\boxed{16}") == 1
    assert text.count("\\boxed{-\\frac{2}{3}}") == 1
    assert text.count("Question:") == 5  # 4 fewshot + 1 target
    assert text.endswith("Question: What is 2+2?\nThe final answer MUST BE put in "
                          "\\boxed{} and no explanation.")


def test_fewshot_prompts_use_the_same_underlying_problems_as_minerva():
    # kept identical across variants deliberately, for comparability -- see
    # rewards.py's _TERSE_FEWSHOT_ANSWERS comment.
    from re_polar.mcts.rewards import _MINERVA_MATH_FEWSHOT, _TERSE_FEWSHOT_ANSWERS
    assert len(_TERSE_FEWSHOT_ANSWERS) == len(_MINERVA_MATH_FEWSHOT) == 4


def test_drllm_raw_prompt_is_drllm_wording_with_no_chat_markup():
    text = drllm_raw_prompt(None, "What is 2+2?")
    assert text == "Question: What is 2+2?\nThe final answer MUST BE put in \\boxed{} and no explanation."
    assert "<|im_start|>" not in text  # no chat template markup at all


def test_drllm_chat_prefill_prompt_ends_with_an_open_boxed_brace():
    text = drllm_chat_prefill_prompt(_FakeTokenizer(), "What is 2+2?")
    # _FakeTokenizer echoes the message content verbatim then we append the prefill
    assert text.endswith("\\boxed{")
    assert "What is 2+2?" in text


def test_drllm_chat_strict_prompt_uses_a_harder_negative_instruction():
    text = drllm_chat_strict_prompt(_FakeTokenizer(), "What is 2+2?")
    assert DRLLM_STRICT_INSTRUCTION in text
    assert "Do not show any reasoning" in text
    assert "ANSWER" not in text


def test_drllm_chat_qwen_deadcode_prompt_matches_their_source_verbatim():
    # DR.LLM's own answer_math_qwen prompt variant, dead code in their upstream
    # repo (never actually called there), reproduced verbatim anyway since it's
    # a real prompt-engineering data point.
    text = drllm_chat_qwen_deadcode_prompt(_FakeTokenizer(), "What is 2+2?")
    assert DRLLM_QWEN_DEADCODE_INSTRUCTION in text
    assert "ONLY return the final result in LaTeX with no words." in text
    assert "\\boxed{...}" in text


def test_paper_minimal_fewshot_prompt_has_exactly_one_trivial_demo():
    text = paper_minimal_fewshot_prompt(None, "What is 2+2?")
    assert MINIMAL_FEWSHOT_QUESTION in text
    assert f"\\boxed{{{MINIMAL_FEWSHOT_ANSWER}}}" in text
    assert text.count("### Problem Start") == 2  # 1 demo + 1 target
    # 3 total: PAPER_INSTRUCTION's own "\boxed{ANSWER}" mention appears once per
    # block (demo + target = 2), plus the demo's one real terse answer
    assert text.count("\\boxed{") == 3
    assert text.endswith(
        f"{PAPER_INSTRUCTION}\n### Problem Start\nWhat is 2+2?\n### Problem End\nAnswer:")


def test_drllm_chat_minimal_fewshot_prompt_is_a_real_multiturn_exchange():
    text = drllm_chat_minimal_fewshot_prompt(_FakeTokenizer(), "What is 2+2?")
    parts = text.split("\n---\n")
    assert len(parts) == 3  # user (demo question), assistant (demo answer), user (real question)
    assert MINIMAL_FEWSHOT_QUESTION in parts[0]
    assert parts[1] == f"\\boxed{{{MINIMAL_FEWSHOT_ANSWER}}}"
    assert "What is 2+2?" in parts[2]


def test_drllm_chat_minimal_fewshot_prefill_combines_both_winners():
    text = drllm_chat_minimal_fewshot_prefill_prompt(_FakeTokenizer(), "What is 2+2?")
    base = drllm_chat_minimal_fewshot_prompt(_FakeTokenizer(), "What is 2+2?")
    # exactly the minimal-fewshot prompt with the prefill appended -- not a
    # reimplementation, so the two can't silently drift apart
    assert text == base + "\\boxed{"
    assert text.endswith("\\boxed{")
    # 4 total: DRLLM_INSTRUCTION's own "\boxed{}" mention (once per message: demo
    # question + real question = 2), the demo's real answer, and the prefill
    assert text.count("\\boxed{") == 4
