"""Explicit, testable orchestration for the GameScope workflow."""

from __future__ import annotations

from typing import Protocol

from gamescope_agent.memory import SessionMemory
from gamescope_agent.models import (
    AgentRun,
    EvaluationReport,
    RetrievalResult,
    TraceEvent,
    WebSource,
)


class Retriever(Protocol):
    def search(self, query: str, n_results: int = 3) -> list[RetrievalResult]: ...


class Evaluator(Protocol):
    def evaluate(
        self, question: str, results: list[RetrievalResult]
    ) -> EvaluationReport: ...


class WebSearcher(Protocol):
    def search(self, question: str, max_results: int = 5) -> list[WebSource]: ...


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        local_results: list[RetrievalResult],
        evaluation: EvaluationReport,
        web_sources: list[WebSource],
    ) -> str: ...


class GameScopeAgent:
    """Stateful agent with a deterministic evidence-control workflow."""

    def __init__(
        self,
        retriever: Retriever,
        evaluator: Evaluator,
        web_searcher: WebSearcher,
        answer_generator: AnswerGenerator,
        memory: SessionMemory | None = None,
    ):
        self._retriever = retriever
        self._evaluator = evaluator
        self._web_searcher = web_searcher
        self._answer_generator = answer_generator
        self.memory = memory or SessionMemory()

    def run(self, question: str, session_id: str = "default") -> AgentRun:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("The question cannot be empty.")

        trace: list[TraceEvent] = []
        contextual_query = self.memory.contextualize(session_id, clean_question)
        used_history = contextual_query != clean_question
        trace.append(
            TraceEvent(
                step="resolve_context",
                summary=(
                    "Resolved a follow-up reference with recent session history."
                    if used_history
                    else "No follow-up reference detected; used the question unchanged."
                ),
            )
        )

        local_results = self._retriever.search(contextual_query, n_results=3)
        trace.append(
            TraceEvent(
                step="retrieve_local",
                summary=f"Retrieved {len(local_results)} local records.",
            )
        )

        evaluation = self._evaluator.evaluate(contextual_query, local_results)
        trace.append(
            TraceEvent(
                step="evaluate_local",
                summary=(
                    f"Local evidence useful={evaluation.useful}, "
                    f"confidence={evaluation.confidence}."
                ),
            )
        )

        local_is_sufficient = evaluation.useful and evaluation.confidence in {
            "high",
            "medium",
        }
        web_sources: list[WebSource] = []
        if not local_is_sufficient:
            web_sources = self._web_searcher.search(contextual_query, max_results=5)
            trace.append(
                TraceEvent(
                    step="search_web",
                    summary=f"Retrieved {len(web_sources)} web sources.",
                )
            )

        answer = self._answer_generator.generate(
            contextual_query,
            local_results,
            evaluation,
            web_sources,
        )
        trace.append(
            TraceEvent(
                step="generate_answer",
                summary="Generated a cited answer from the approved evidence path.",
            )
        )
        self.memory.add(session_id, clean_question, answer)

        run = AgentRun(
            session_id=session_id,
            question=clean_question,
            contextual_query=contextual_query,
            retrieval_results=local_results,
            evaluation=evaluation,
            web_sources=web_sources,
            answer=answer,
            trace=trace,
        )
        validate_workflow(run)
        return run


def validate_workflow(run: AgentRun) -> None:
    sequence = run.tool_sequence
    local_is_sufficient = run.evaluation.useful and run.evaluation.confidence in {
        "high",
        "medium",
    }
    expected = ["retrieve_game", "evaluate_retrieval"]
    if not local_is_sufficient:
        expected.append("game_web_search")

    if sequence != expected:
        raise RuntimeError(
            f"Invalid tool sequence: expected {expected}, received {sequence}."
        )
