"""Rolling summary generation for sessions."""

from .config import TURNS_BETWEEN_SUMMARIES
from .models.base import LLMBackend, Message
from .storage.schemas import Session, Turn

SUMMARY_SYSTEM_PROMPT = """You are a session summarizer for a Pathfinder 2E tabletop RPG game.

Your task is to create a concise summary of what has happened in the current session.
Focus on:
- Key events and decisions
- Important NPC interactions
- Discoveries and plot developments
- Current party status and location

Keep the summary under 200 words. Write in past tense. Be factual but engaging.
"""

SUMMARY_USER_TEMPLATE = """Please summarize the following session so far:

{previous_summary}

Recent events (turns {start_turn} to {end_turn}):
{turns_text}

Current scene: {location} ({time})

Write a cohesive summary incorporating the previous summary (if any) with the recent events.
"""


def should_update_summary(session: Session, force: bool = False) -> bool:
    """Check if it's time to update the rolling summary.

    Updates every TURNS_BETWEEN_SUMMARIES turns, or when forced.
    """
    if force:
        return True

    turn_count = len(session.turns)
    if turn_count == 0:
        return False

    # Update at intervals
    return turn_count % TURNS_BETWEEN_SUMMARIES == 0


def generate_summary_prompt(session: Session, since_turn: int = 0) -> list[Message]:
    """Generate messages for summary request."""
    turns_text = _format_turns_for_summary(session.turns[since_turn:])

    previous_summary = ""
    if session.summary:
        previous_summary = f"Previous summary:\n{session.summary}\n"

    user_content = SUMMARY_USER_TEMPLATE.format(
        previous_summary=previous_summary,
        start_turn=since_turn + 1,
        end_turn=len(session.turns),
        turns_text=turns_text,
        location=session.scene_state.location,
        time=session.scene_state.time_of_day,
    )

    return [
        Message(role="system", content=SUMMARY_SYSTEM_PROMPT),
        Message(role="user", content=user_content),
    ]


def _format_turns_for_summary(turns: list[Turn]) -> str:
    """Format turns into readable text for summarization."""
    if not turns:
        return "No turns yet."

    lines = []
    for i, turn in enumerate(turns):
        lines.append(f"Turn {i + 1}:")
        lines.append(f"  Player: {turn.player_input}")
        lines.append(f"  GM: {turn.gm_response[:500]}...")  # Truncate long responses
        if turn.tool_calls:
            tool_names = [tc.name for tc in turn.tool_calls]
            lines.append(f"  [Tools used: {', '.join(tool_names)}]")
        lines.append("")

    return "\n".join(lines)


async def generate_summary_async(
    llm: LLMBackend,
    session: Session,
    since_turn: int = 0,
) -> str:
    """Generate a rolling summary using the LLM (async version for future use)."""
    # For now, just call the sync version
    return generate_summary(llm, session, since_turn)


def generate_summary(
    llm: LLMBackend,
    session: Session,
    since_turn: int = 0,
) -> str:
    """Generate a rolling summary using the LLM.

    Args:
        llm: The LLM backend to use for generation
        session: The current session
        since_turn: Only include turns after this index (for incremental updates)

    Returns:
        The generated summary text
    """
    messages = generate_summary_prompt(session, since_turn)
    response = llm.chat(messages, tools=None)  # No tools for summarization
    return response.text.strip()


class RollingSummarizer:
    """Manages rolling summary generation for a session.

    Tracks when summaries were last generated and handles incremental updates.
    """

    def __init__(self, llm: LLMBackend, interval: int | None = None):
        self.llm = llm
        self.interval = interval or TURNS_BETWEEN_SUMMARIES
        self._last_summarized_turn = 0

    def maybe_update(self, session: Session, force: bool = False) -> str | None:
        """Update summary if needed.

        Returns the new summary if updated, None otherwise.
        """
        turn_count = len(session.turns)

        # Check if update is needed
        if not force:
            if turn_count == 0:
                return None
            if turn_count < self._last_summarized_turn + self.interval:
                return None

        # Generate new summary
        summary = generate_summary(
            self.llm,
            session,
            since_turn=max(0, self._last_summarized_turn - 2),  # Overlap for context
        )

        self._last_summarized_turn = turn_count
        return summary

    def reset(self) -> None:
        """Reset the summarizer state (e.g., for new session)."""
        self._last_summarized_turn = 0
