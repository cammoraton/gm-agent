"""Virtual player decision engine.

Uses an LLM backend + personality profile to generate decisions
for AI-driven participants in generation games.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from ...config import TEMPERATURE_CREATIVE

if TYPE_CHECKING:
    from ...models.base import LLMBackend

logger = logging.getLogger(__name__)

VIRTUAL_PLAYER_SYSTEM_PROMPT = """\
You are {name}, a virtual player in a collaborative worldbuilding game.

{personality_section}

You are creative, collaborative, and respectful of the established fiction.
Make choices that are interesting, surprising, and build on what came before.

{preferences_section}

IMPORTANT: Respond ONLY with a valid JSON object matching the requested format.
Do not include any other text outside the JSON."""

DECISION_USER_PROMPT = """\
It is your turn to make a decision.

**Decision type:** {decision_type}
**Available options:** {available_options}

**Current game state:**
{game_state}

Respond with a JSON object:
{{
  "decision_type": "{decision_type}",
  "content": {{ ... relevant fields for this decision ... }},
  "reasoning": "Brief explanation of your creative reasoning"
}}"""

MAX_DECISION_HISTORY = 5

HISTORY_SECTION = """
Your previous decisions this session:
{history_text}
Build on your earlier choices. Be consistent but don't repeat yourself."""


def _parse_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON object from LLM response")
    return {}


class VirtualPlayerEngine:
    """LLM-driven decision engine for virtual players."""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def generate_decision(
        self,
        player_name: str,
        personality_profile: dict[str, Any] | None = None,
        creative_preferences: dict[str, str] | None = None,
        game_state: dict[str, Any] | None = None,
        decision_type: str = "create",
        available_options: list[str] | None = None,
        decision_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a decision for a virtual player.

        Args:
            player_name: The virtual player's name.
            personality_profile: Serialized PersonalityProfile dict.
            creative_preferences: Free-text preferences (e.g. {"genre": "dark fantasy"}).
            game_state: Current game state dict.
            decision_type: What kind of decision to make.
            available_options: Optional list of allowed actions.
            decision_history: Previous decisions this session for continuity.

        Returns:
            Dict with decision_type, content, and reasoning keys.
        """
        from ...models.base import Message

        # Build personality section
        personality_section = ""
        if personality_profile:
            try:
                from .personality import PersonalityProfile
                pp = PersonalityProfile.from_dict(personality_profile)
                personality_section = f"Your personality:\n{pp.describe_for_prompt()}"
            except Exception:
                personality_section = ""

        # Build preferences section
        preferences_section = ""
        if creative_preferences:
            prefs = "\n".join(f"- {k}: {v}" for k, v in creative_preferences.items())
            preferences_section = f"Your creative preferences:\n{prefs}"

        system_msg = VIRTUAL_PLAYER_SYSTEM_PROMPT.format(
            name=player_name,
            personality_section=personality_section or "You have a balanced, neutral personality.",
            preferences_section=preferences_section or "",
        )

        options_str = ", ".join(available_options) if available_options else "any appropriate action"
        state_str = json.dumps(game_state or {}, indent=2, default=str)

        user_msg = DECISION_USER_PROMPT.format(
            decision_type=decision_type,
            available_options=options_str,
            game_state=state_str,
        )

        # Append decision history if available
        if decision_history:
            recent = decision_history[-MAX_DECISION_HISTORY:]
            history_lines = []
            for i, h in enumerate(recent, 1):
                content_str = json.dumps(h.get("content", {}), default=str)
                if len(content_str) > 200:
                    content_str = content_str[:200] + "..."
                history_lines.append(f"{i}. [{h.get('decision_type', '?')}] {content_str}")
            user_msg += HISTORY_SECTION.format(history_text="\n".join(history_lines))

        messages = [
            Message(role="system", content=system_msg),
            Message(role="user", content=user_msg),
        ]

        response = self.llm.chat(messages, temperature=TEMPERATURE_CREATIVE)
        parsed = _parse_json_object(response.text)

        return {
            "decision_type": parsed.get("decision_type", decision_type),
            "content": parsed.get("content", {}),
            "reasoning": parsed.get("reasoning", ""),
        }
