"""Run after `gamescope ingest` and configuring `.env`."""

from pathlib import Path

from gamescope_agent.config import Settings
from gamescope_agent.factory import build_agent

settings = Settings.load(Path.cwd())
agent = build_agent(settings)

for question in (
    "When was Pokémon Gold and Silver released?",
    "Was Mortal Kombat X released natively for PlayStation 5?",
):
    run = agent.run(question, session_id="example")
    print(run.answer)
    print(run.tool_sequence)
