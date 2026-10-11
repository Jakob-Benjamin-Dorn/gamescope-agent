"""LLM evaluation, web search, and answer generation services."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from gamescope_agent.config import Settings
from gamescope_agent.memory import ConversationTurn
from gamescope_agent.models import EvaluationReport, RetrievalResult, WebSource

MAX_HISTORY_ANSWER_CHARS = 600


def _openai_client(settings: Settings) -> Any:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": settings.require_openai()}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def build_rewrite_prompt(question: str, history: list[ConversationTurn]) -> str:
    """Prompt asking the model to turn a follow-up into a standalone question."""
    transcript = "\n\n".join(
        f"User: {turn.question}\nAssistant: {_shorten(turn.answer)}" for turn in history
    )
    return f"""
Rewrite the latest user message as a fully self-contained question about video
games.

Conversation so far:
{transcript}

Latest user message:
{question}

Rules:
- Resolve references such as "it", "that game" or "my first question" using the
  conversation.
- Keep the user's intent and language. Do not answer the question.
- If the message is already self-contained, return it unchanged.
- Return only the rewritten question, without quotes or commentary.
""".strip()


def _shorten(text: str, limit: int = MAX_HISTORY_ANSWER_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


class OpenAIQuestionRewriter:
    def __init__(self, settings: Settings):
        self._client = _openai_client(settings)
        self._model = settings.openai_model

    def rewrite(self, question: str, history: list[ConversationTurn]) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions="You rewrite follow-up questions into standalone questions.",
            input=build_rewrite_prompt(question, history),
        )
        return response.output_text.strip().strip('"')


class OpenAIEvidenceEvaluator:
    def __init__(self, settings: Settings):
        self._client = _openai_client(settings)
        self._model = settings.openai_model

    def evaluate(
        self, question: str, results: list[RetrievalResult]
    ) -> EvaluationReport:
        evidence = json.dumps(
            [result.model_dump() for result in results], ensure_ascii=False
        )
        prompt = f"""
Evaluate whether the retrieved local records are sufficient to answer the exact
user question accurately.

Question:
{question}

Retrieved evidence:
{evidence}

Rules:
- Use only the retrieved evidence.
- Similarity alone is not sufficient; the requested fact must be explicit.
- Treat publisher and developer as different roles.
- Current or recent questions require current evidence.
- If evidence is missing, ambiguous, stale, or unrelated, set useful=false.
- Return only the structured evaluation, not private chain-of-thought.
""".strip()
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": "You are a strict evidence sufficiency evaluator.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=EvaluationReport,
        )
        if response.output_parsed is None:
            return EvaluationReport(
                useful=False,
                confidence="low",
                explanation="The structured evaluation could not be parsed.",
                missing_information="A reliable evaluation result is missing.",
            )
        return response.output_parsed


class TavilyWebSearcher:
    def __init__(self, settings: Settings):
        self._api_key = settings.tavily_api_key

    def search(self, question: str, max_results: int = 5) -> list[WebSource]:
        if not self._api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is required because local evidence was "
                "insufficient and web fallback was triggered."
            )
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._api_key)
        query = (
            f"{question}\nCurrent date: {date.today().isoformat()}. "
            "Prefer recent official or primary sources."
        )
        response = client.search(
            query=query,
            search_depth="basic",
            include_answer=False,
            max_results=min(max(max_results, 1), 5),
        )
        return [
            WebSource(
                rank=rank,
                title=str(result.get("title") or "Untitled source"),
                url=str(result.get("url") or ""),
                content=str(result.get("content") or ""),
                relevance_score=result.get("score"),
            )
            for rank, result in enumerate(response.get("results", []), start=1)
        ]


class OpenAIAnswerGenerator:
    def __init__(self, settings: Settings):
        self._client = _openai_client(settings)
        self._model = settings.openai_model

    def generate(
        self,
        question: str,
        local_results: list[RetrievalResult],
        evaluation: EvaluationReport,
        web_sources: list[WebSource],
    ) -> str:
        local_payload = json.dumps(
            [result.model_dump() for result in local_results], ensure_ascii=False
        )
        web_payload = json.dumps(
            [source.model_dump() for source in web_sources], ensure_ascii=False
        )
        path = (
            "Local retrieval -> Evaluation -> Web search -> Final answer"
            if web_sources
            else "Local retrieval -> Evaluation -> Final answer"
        )
        prompt = f"""
Answer the user's video-game question using only the supplied evidence.

Question:
{question}

Local evidence:
{local_payload}

Local evidence evaluation:
{evaluation.model_dump_json()}

Web evidence:
{web_payload}

Requirements:
- Give a direct answer.
- Do not infer unsupported dates, platforms, developers, publishers, or status.
- Prefer local evidence when it is sufficient.
- When web evidence is used, cite source title and URL.
- For local evidence, cite the exact citation field in square brackets.
- State uncertainty or conflicts explicitly.
- Distinguish native platform releases from backward compatibility.
- Use this exact final section structure:

Answer:
...

Confidence:
High, Medium, or Low.

Evidence summary:
...

Sources:
...

Retrieval path:
{path}
""".strip()
        response = self._client.responses.create(
            model=self._model,
            instructions="You are GameScope, an evidence-first research assistant.",
            input=prompt,
        )
        return response.output_text.strip()
