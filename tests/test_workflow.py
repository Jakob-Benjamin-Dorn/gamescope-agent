import pytest

from gamescope_agent.models import (
    AgentRun,
    EvaluationReport,
    RetrievalResult,
    WebSource,
)
from gamescope_agent.workflow import GameScopeAgent, validate_workflow

RESULT = RetrievalResult(
    rank=1,
    record_id="006",
    distance=0.1,
    name="Pokémon Gold and Silver",
    platform="Game Boy Color",
    genre="Role-playing",
    publisher="Nintendo",
    year_of_release=1999,
    description="A second-generation role-playing pair.",
    citation="Local dataset: data/games/006.json",
)


class FakeRetriever:
    def search(self, query: str, n_results: int = 3):
        return [RESULT]


class FakeEvaluator:
    def __init__(self, useful: bool):
        self.useful = useful

    def evaluate(self, question: str, results: list[RetrievalResult]):
        return EvaluationReport(
            useful=self.useful,
            confidence="high",
            explanation="Deterministic test evaluation.",
            missing_information="None" if self.useful else "Current source required.",
        )


class FakeWebSearcher:
    def search(self, question: str, max_results: int = 5):
        return [
            WebSource(
                rank=1,
                title="Official source",
                url="https://example.com",
                content="Test evidence",
            )
        ]


class FakeAnswerGenerator:
    def generate(self, question, local_results, evaluation, web_sources):
        return "Test answer"


def build(useful: bool) -> GameScopeAgent:
    return GameScopeAgent(
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(useful),
        web_searcher=FakeWebSearcher(),
        answer_generator=FakeAnswerGenerator(),
    )


def test_sufficient_local_evidence_skips_web_search() -> None:
    run = build(useful=True).run("When was it released?")
    assert run.tool_sequence == ["retrieve_game", "evaluate_retrieval"]
    assert run.web_sources == []


def test_insufficient_local_evidence_uses_web_search() -> None:
    run = build(useful=False).run("What is happening now?")
    assert run.tool_sequence == [
        "retrieve_game",
        "evaluate_retrieval",
        "game_web_search",
    ]
    assert len(run.web_sources) == 1


def test_session_history_is_added_to_follow_up_query() -> None:
    agent = build(useful=True)
    agent.run("When was Pokémon Gold and Silver released?", session_id="s1")
    follow_up = agent.run("Which platform was it on?", session_id="s1")
    assert "Previous question:" in follow_up.contextual_query
    assert "Pokémon Gold and Silver" in follow_up.contextual_query


def test_unrelated_question_does_not_inherit_session_history() -> None:
    agent = build(useful=True)
    agent.run("When was Pokémon Gold and Silver released?", session_id="s2")
    independent = agent.run(
        "Which game was an early 3D Mario platformer?", session_id="s2"
    )
    assert independent.contextual_query == (
        "Which game was an early 3D Mario platformer?"
    )


class CapturingEvaluator(FakeEvaluator):
    def __init__(self):
        super().__init__(useful=True)
        self.question = ""

    def evaluate(self, question: str, results: list[RetrievalResult]):
        self.question = question
        return super().evaluate(question, results)


class CapturingAnswerGenerator(FakeAnswerGenerator):
    def __init__(self):
        self.question = ""

    def generate(self, question, local_results, evaluation, web_sources):
        self.question = question
        return super().generate(question, local_results, evaluation, web_sources)


def test_follow_up_context_reaches_evaluation_and_answer_generation() -> None:
    evaluator = CapturingEvaluator()
    generator = CapturingAnswerGenerator()
    agent = GameScopeAgent(
        retriever=FakeRetriever(),
        evaluator=evaluator,
        web_searcher=FakeWebSearcher(),
        answer_generator=generator,
    )
    agent.run("When was Pokémon Gold and Silver released?", session_id="s3")
    follow_up = agent.run(
        "What platform was the game from my first question released on?",
        session_id="s3",
    )

    assert "Pokémon Gold and Silver" in follow_up.contextual_query
    assert evaluator.question == follow_up.contextual_query
    assert generator.question == follow_up.contextual_query


def make_run(
    *, useful: bool, tool_calls: list[str], with_web: bool = False
) -> AgentRun:
    return AgentRun(
        session_id="check",
        question="Q",
        contextual_query="Q",
        retrieval_results=[RESULT],
        evaluation=EvaluationReport(
            useful=useful,
            confidence="high",
            explanation="Test.",
            missing_information="None",
        ),
        web_sources=(
            [WebSource(rank=1, title="T", url="https://example.com", content="C")]
            if with_web
            else []
        ),
        answer="A",
        trace=[],
        tool_calls=tool_calls,
    )


def test_validation_accepts_the_policy_conform_sequences() -> None:
    validate_workflow(
        make_run(useful=True, tool_calls=["retrieve_game", "evaluate_retrieval"])
    )
    validate_workflow(
        make_run(
            useful=False,
            tool_calls=["retrieve_game", "evaluate_retrieval", "game_web_search"],
            with_web=True,
        )
    )


def test_validation_rejects_web_search_despite_sufficient_evidence() -> None:
    run = make_run(
        useful=True,
        tool_calls=["retrieve_game", "evaluate_retrieval", "game_web_search"],
    )
    with pytest.raises(RuntimeError, match="Invalid tool sequence"):
        validate_workflow(run)


def test_validation_rejects_missing_web_search_for_insufficient_evidence() -> None:
    run = make_run(useful=False, tool_calls=["retrieve_game", "evaluate_retrieval"])
    with pytest.raises(RuntimeError, match="Invalid tool sequence"):
        validate_workflow(run)


def test_validation_rejects_skipped_evaluation() -> None:
    run = make_run(useful=True, tool_calls=["retrieve_game"])
    with pytest.raises(RuntimeError, match="Invalid tool sequence"):
        validate_workflow(run)


def test_validation_rejects_web_sources_without_web_search() -> None:
    run = make_run(
        useful=True,
        tool_calls=["retrieve_game", "evaluate_retrieval"],
        with_web=True,
    )
    with pytest.raises(RuntimeError, match="no web search"):
        validate_workflow(run)


class SkippingAgent(GameScopeAgent):
    """Simulates a regression: the agent never records the evaluation call."""

    def run(self, question: str, session_id: str = "default") -> AgentRun:
        run = super().run(question, session_id)
        broken = run.model_copy(update={"tool_calls": ["retrieve_game"]})
        validate_workflow(broken)
        return broken


def test_regression_in_recorded_calls_is_detected() -> None:
    agent = SkippingAgent(
        retriever=FakeRetriever(),
        evaluator=FakeEvaluator(useful=True),
        web_searcher=FakeWebSearcher(),
        answer_generator=FakeAnswerGenerator(),
    )
    with pytest.raises(RuntimeError):
        agent.run("When was it released?")


class FakeRewriter:
    def __init__(self, result: str):
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def rewrite(self, question, history):
        self.calls.append((question, len(history)))
        return self.result


class CapturingRetriever(FakeRetriever):
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, n_results: int = 3):
        self.queries.append(query)
        return super().search(query, n_results)


def build_with_rewriter(rewriter, retriever=None) -> GameScopeAgent:
    return GameScopeAgent(
        retriever=retriever or FakeRetriever(),
        evaluator=FakeEvaluator(useful=True),
        web_searcher=FakeWebSearcher(),
        answer_generator=FakeAnswerGenerator(),
        question_rewriter=rewriter,
    )


def test_rewriter_is_not_called_for_the_first_question() -> None:
    rewriter = FakeRewriter("should not be used")
    run = build_with_rewriter(rewriter).run("When was Pokémon Gold released?")

    assert rewriter.calls == []
    assert run.contextual_query == "When was Pokémon Gold released?"


def test_rewritten_follow_up_replaces_the_raw_history() -> None:
    retriever = CapturingRetriever()
    rewriter = FakeRewriter("Which platform was Pokémon Gold and Silver released on?")
    agent = build_with_rewriter(rewriter, retriever)
    agent.run("When was Pokémon Gold and Silver released?", session_id="r1")
    follow_up = agent.run("And on which platform?", session_id="r1")

    assert rewriter.calls == [("And on which platform?", 1)]
    assert follow_up.contextual_query == rewriter.result
    assert retriever.queries[-1] == rewriter.result
    assert "Previous question:" not in follow_up.contextual_query
    assert follow_up.question == "And on which platform?"


def test_empty_rewrite_falls_back_to_the_original_question() -> None:
    agent = build_with_rewriter(FakeRewriter("   "))
    agent.run("When was Pokémon Gold and Silver released?", session_id="r2")
    follow_up = agent.run("And on which platform?", session_id="r2")

    assert follow_up.contextual_query == "And on which platform?"


def test_unchanged_rewrite_is_reported_as_self_contained() -> None:
    question = "Which game was an early 3D Mario platformer?"
    agent = build_with_rewriter(FakeRewriter(question))
    agent.run("When was Pokémon Gold and Silver released?", session_id="r3")
    run = agent.run(question, session_id="r3")

    assert run.contextual_query == question
    assert "self-contained" in run.trace[0].summary
