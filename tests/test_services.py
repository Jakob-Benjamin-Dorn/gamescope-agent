from gamescope_agent.memory import ConversationTurn
from gamescope_agent.services import MAX_HISTORY_ANSWER_CHARS, build_rewrite_prompt


def test_rewrite_prompt_contains_history_and_latest_question() -> None:
    history = [
        ConversationTurn(
            question="When was Pokémon Gold and Silver released?",
            answer="Answer: 1999.",
        )
    ]
    prompt = build_rewrite_prompt("And on which platform?", history)

    assert "User: When was Pokémon Gold and Silver released?" in prompt
    assert "Assistant: Answer: 1999." in prompt
    assert "Latest user message:\nAnd on which platform?" in prompt


def test_rewrite_prompt_shortens_long_answers() -> None:
    long_answer = "word " * 500
    prompt = build_rewrite_prompt(
        "And on which platform?",
        [ConversationTurn(question="Q", answer=long_answer)],
    )

    assistant_line = next(
        line for line in prompt.splitlines() if line.startswith("Assistant:")
    )
    assert len(assistant_line) < MAX_HISTORY_ANSWER_CHARS + 20
    assert assistant_line.endswith("...")
