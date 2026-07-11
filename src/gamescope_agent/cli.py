"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gamescope_agent.config import Settings
from gamescope_agent.data import load_game_records
from gamescope_agent.factory import build_agent
from gamescope_agent.vector_store import ChromaGameStore


def _settings() -> Settings:
    return Settings.load(Path.cwd())


def ingest() -> None:
    settings = _settings()
    records = load_game_records(settings.data_dir)
    store = ChromaGameStore.from_settings(settings)
    count = store.ingest(records)
    print(f"Stored {count} records in collection '{settings.collection_name}'.")


def ask(question: str, session_id: str) -> None:
    run = build_agent(_settings()).run(question, session_id=session_id)
    print(run.answer)
    print("\nTrace:")
    print(json.dumps([event.model_dump() for event in run.trace], indent=2))


def demo() -> None:
    agent = build_agent(_settings())
    session_id = "demo"
    questions = [
        "When was Pokémon Gold and Silver released?",
        "Which game in the local collection was an early 3D Mario platformer?",
        "Was Mortal Kombat X released natively for PlayStation 5?",
        "What platform was the game from my first question released on?",
    ]
    for question in questions:
        print("\n" + "=" * 80)
        print(question)
        print("=" * 80)
        run = agent.run(question, session_id=session_id)
        print(run.answer)
        print(f"\nTool sequence: {run.tool_sequence}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="gamescope")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Validate and index the local JSON corpus.")

    ask_parser = subparsers.add_parser("ask", help="Ask one question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--session", default="default")

    subparsers.add_parser("demo", help="Run the portfolio demonstration.")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        ask(args.question, args.session)
    else:
        demo()


if __name__ == "__main__":
    main()
