"""Application assembly."""

from gamescope_agent.config import Settings
from gamescope_agent.services import (
    OpenAIAnswerGenerator,
    OpenAIEvidenceEvaluator,
    TavilyWebSearcher,
)
from gamescope_agent.vector_store import ChromaGameStore
from gamescope_agent.workflow import GameScopeAgent


def build_agent(settings: Settings) -> GameScopeAgent:
    store = ChromaGameStore.from_settings(settings)
    return GameScopeAgent(
        retriever=store,
        evaluator=OpenAIEvidenceEvaluator(settings),
        web_searcher=TavilyWebSearcher(settings),
        answer_generator=OpenAIAnswerGenerator(settings),
    )
