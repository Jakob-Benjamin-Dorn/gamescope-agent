"""Minimal session memory for contextual follow-up questions."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str


class SessionMemory:
    _FOLLOW_UP_PATTERNS = (
        re.compile(
            r"\b(?:my|the)\s+(?:first|previous|last|earlier)\s+question\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:that|this|same|previous|earlier)\s+"
            r"(?:game|title|one|release|platform)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:it|its|that one|the former|the latter)\b",
            re.IGNORECASE,
        ),
    )

    def __init__(self, max_turns: int = 6):
        self._sessions: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._max_turns = max_turns

    def history(self, session_id: str) -> list[ConversationTurn]:
        return list(self._sessions.get(session_id, []))

    def add(self, session_id: str, question: str, answer: str) -> None:
        turns = self._sessions[session_id]
        turns.append(ConversationTurn(question=question, answer=answer))
        if len(turns) > self._max_turns:
            del turns[: -self._max_turns]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def contextualize(self, session_id: str, question: str) -> str:
        turns = self.history(session_id)
        if not turns or not self._is_follow_up(question):
            return question

        selected_turns = turns[-3:]
        if re.search(r"\b(?:my|the)\s+first\s+question\b", question, re.I):
            selected_turns = [turns[0]]

        history = "\n".join(
            f"Previous question: {turn.question}\nPrevious answer: {turn.answer}"
            for turn in selected_turns
        )
        return f"{history}\nCurrent question: {question}"

    @classmethod
    def _is_follow_up(cls, question: str) -> bool:
        return any(pattern.search(question) for pattern in cls._FOLLOW_UP_PATTERNS)
