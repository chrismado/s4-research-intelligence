"""
Conversation memory for multi-turn research queries.

Stores conversation history so follow-up questions have context.
"What about his educational claims?" should know we're still talking about Lazar.
"""

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    question: str
    answer: str
    sources_summary: str = ""


@dataclass
class ConversationMemory:
    """
    Simple conversation buffer for multi-turn research queries.

    Keeps the last N turns and provides a context summary
    that can be prepended to new questions for the LLM.
    """

    max_turns: int = 5
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str, sources_summary: str = ""):
        """Record a conversation turn."""
        self.turns.append(ConversationTurn(
            question=question,
            answer=answer,
            sources_summary=sources_summary,
        ))
        # Keep only the last N turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_prompt(self) -> str | None:
        """
        Build a context string from conversation history.

        Returns None if no history exists.
        """
        if not self.turns:
            return None

        parts = []
        for turn in self.turns:
            parts.append(f"User asked: {turn.question}")
            # Truncate answer to keep context manageable
            answer_preview = turn.answer[:300]
            if len(turn.answer) > 300:
                answer_preview += "..."
            parts.append(f"Assistant answered: {answer_preview}")
            if turn.sources_summary:
                parts.append(f"Sources referenced: {turn.sources_summary}")
            parts.append("")

        return "\n".join(parts)

    def enrich_question(self, question: str) -> str:
        """
        Prepend conversation context to a follow-up question.

        If no history exists, returns the question unchanged.
        """
        context = self.get_context_prompt()
        if not context:
            return question

        return (
            f"CONVERSATION HISTORY (for context on follow-up questions):\n"
            f"{context}\n"
            f"CURRENT QUESTION: {question}"
        )

    def clear(self):
        """Reset conversation history."""
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self.turns)
