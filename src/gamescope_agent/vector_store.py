"""Persistent Chroma-backed retrieval."""

from __future__ import annotations

from typing import Any

from gamescope_agent.config import Settings
from gamescope_agent.models import GameRecord, RetrievalResult


class ChromaGameStore:
    """Small adapter around a Chroma collection."""

    def __init__(self, collection: Any):
        self._collection = collection

    @classmethod
    def from_settings(cls, settings: Settings) -> ChromaGameStore:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        except ImportError as error:
            raise RuntimeError(
                "Chroma dependencies are missing. Install the project with "
                "`python -m pip install -e .`."
            ) from error

        settings.require_openai()
        embedding_kwargs: dict[str, Any] = {
            "api_key_env_var": "OPENAI_API_KEY",
            "model_name": settings.embedding_model,
        }
        if settings.openai_base_url:
            embedding_kwargs["api_base"] = settings.openai_base_url

        embedding_function = OpenAIEmbeddingFunction(**embedding_kwargs)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_function,
            metadata={"description": "Local video-game knowledge base for GameScope"},
        )
        return cls(collection)

    def ingest(self, records: list[GameRecord]) -> int:
        if not records:
            return 0

        self._collection.upsert(
            ids=[
                (record.source_file or f"record-{index:03d}.json").removesuffix(".json")
                for index, record in enumerate(records, start=1)
            ],
            documents=[record.as_document() for record in records],
            metadatas=[record.metadata() for record in records],
        )
        return self._collection.count()

    def count(self) -> int:
        return int(self._collection.count())

    def search(self, query: str, n_results: int = 3) -> list[RetrievalResult]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The retrieval query cannot be empty.")

        available = self.count()
        if available == 0:
            return []

        raw = self._collection.query(
            query_texts=[clean_query],
            n_results=min(max(n_results, 1), available),
            include=["metadatas", "distances"],
        )

        results: list[RetrievalResult] = []
        for rank, (record_id, metadata, distance) in enumerate(
            zip(
                raw["ids"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                strict=True,
            ),
            start=1,
        ):
            source_file = str(metadata.get("source_file", f"{record_id}.json"))
            results.append(
                RetrievalResult(
                    rank=rank,
                    record_id=record_id,
                    distance=round(float(distance), 4),
                    name=str(metadata["name"]),
                    platform=str(metadata["platform"]),
                    genre=str(metadata["genre"]),
                    publisher=str(metadata["publisher"]),
                    year_of_release=int(metadata["year_of_release"]),
                    description=str(metadata["description"]),
                    citation=f"Local dataset: data/games/{source_file}",
                )
            )
        return results
