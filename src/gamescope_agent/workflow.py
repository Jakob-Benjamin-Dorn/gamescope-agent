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


RETRIEVE_TOOL = "retrieve_game"
EVALUATE_TOOL = "evaluate_retrieval"
WEB_SEARCH_TOOL = "game_web_search"


def local_evidence_is_sufficient(evaluation: EvaluationReport) -> bool:
    """Single policy deciding whether the web fallback may be skipped."""
    return evaluation.useful and evaluation.confidence in {"high", "medium"}


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

        tool_calls: list[str] = []

        local_results = self._retriever.search(contextual_query, n_results=3)
        tool_calls.append(RETRIEVE_TOOL)
        trace.append(
            TraceEvent(
                step="retrieve_local",
                summary=f"Retrieved {len(local_results)} local records.",
            )
        )

        evaluation = self._evaluator.evaluate(contextual_query, local_results)
        tool_calls.append(EVALUATE_TOOL)
        trace.append(
            TraceEvent(
                step="evaluate_local",
                summary=(
                    f"Local evidence useful={evaluation.useful}, "
                    f"confidence={evaluation.confidence}."
                ),
            )
        )

        web_sources: list[WebSource] = []
        if not local_evidence_is_sufficient(evaluation):
            web_sources = self._web_searcher.search(contextual_query, max_results=5)
            tool_calls.append(WEB_SEARCH_TOOL)
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
            tool_calls=tool_calls,
        )
        validate_workflow(run)
        return run


def expected_tool_calls(evaluation: EvaluationReport) -> list[str]:
    """Tool sequence the evidence policy requires for this evaluation."""
    expected = [RETRIEVE_TOOL, EVALUATE_TOOL]
    if not local_evidence_is_sufficient(evaluation):
        expected.append(WEB_SEARCH_TOOL)
    return expected


def validate_workflow(run: AgentRun) -> None:
    """Check the recorded tool calls against the evidence policy.

    ``run.tool_calls`` is written at the moment each service is called, so a
    regression in ``GameScopeAgent.run`` (skipped evaluation, web search despite
    sufficient local evidence, or no web search despite insufficient evidence)
    is detected instead of silently passing.
    """
    expected = expected_tool_calls(run.evaluation)
    if run.tool_calls != expected:
        raise RuntimeError(
            f"Invalid tool sequence: expected {expected}, received {run.tool_calls}."
        )

    if run.web_sources and WEB_SEARCH_TOOL not in run.tool_calls:
        raise RuntimeError("Web sources are present although no web search ran.")
