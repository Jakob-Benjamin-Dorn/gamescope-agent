"""Domain and workflow models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GameRecord(BaseModel):
    """Validated source record for one game."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="Name", min_length=1)
    platform: str = Field(alias="Platform", min_length=1)
    genre: str = Field(alias="Genre", min_length=1)
    publisher: str = Field(alias="Publisher", min_length=1)
    description: str = Field(alias="Description", min_length=1)
    year_of_release: int = Field(alias="YearOfRelease", ge=1970, le=2100)
    source_file: str | None = Field(default=None, alias="SourceFile")

    def as_document(self) -> str:
        return (
            f"Game title: {self.name}\n"
            f"Platform: {self.platform}\n"
            f"Genre: {self.genre}\n"
            f"Publisher: {self.publisher}\n"
            f"Year of release: {self.year_of_release}\n"
            f"Description: {self.description}"
        )

    def metadata(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "platform": self.platform,
            "genre": self.genre,
            "publisher": self.publisher,
            "description": self.description,
            "year_of_release": self.year_of_release,
            "source_file": self.source_file or "unknown.json",
        }


class RetrievalResult(BaseModel):
    rank: int
    record_id: str
    distance: float
    name: str
    platform: str
    genre: str
    publisher: str
    year_of_release: int
    description: str
    citation: str


class EvaluationReport(BaseModel):
    useful: bool
    confidence: Literal["high", "medium", "low"]
    explanation: str
    missing_information: str


class WebSource(BaseModel):
    rank: int
    title: str
    url: str
    content: str
    relevance_score: float | None = None


class TraceEvent(BaseModel):
    step: Literal[
        "resolve_context",
        "retrieve_local",
        "evaluate_local",
        "search_web",
        "generate_answer",
    ]
    summary: str


class AgentRun(BaseModel):
    session_id: str
    question: str
    contextual_query: str
    retrieval_results: list[RetrievalResult]
    evaluation: EvaluationReport
    web_sources: list[WebSource] = Field(default_factory=list)
    answer: str
    trace: list[TraceEvent]
    tool_calls: list[str] = Field(default_factory=list)

    @property
    def tool_sequence(self) -> list[str]:
        """Tools in the order they were actually called."""
        return list(self.tool_calls)
