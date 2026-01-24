"""Context assembly for GM agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    GM_SYSTEM_PROMPT,
    MAX_RECENT_TURNS,
    RAG_PROMPTS,
    UNCERTAINTY_PROMPTS,
    DEFAULT_PREFERENCES,
)
from .models.base import Message, ToolCall
from .storage.schemas import Campaign, Session

if TYPE_CHECKING:
    from .mcp.foundry_vtt import FoundryVTTServer


def build_context(
    campaign: Campaign,
    session: Session,
    max_turns: int | None = None,
    foundry_server: FoundryVTTServer | None = None,
) -> list[Message]:
    """
    Build the 6-layer context for the GM agent.

    Layers:
    1. Campaign background
    2. Current arc
    3. Session summary (if available)
    4. Recent turns
    5. Current scene state (from Foundry VTT if connected, else JSON fallback)
    6. Tool definitions (handled separately by LLM backend)

    Also includes:
    - RAG aggressiveness prompt based on campaign preferences
    - Uncertainty mode prompt based on campaign preferences

    Args:
        campaign: Campaign data
        session: Current session data
        max_turns: Maximum recent turns to include
        foundry_server: Optional FoundryVTTServer for live scene data
    """
    max_turns = max_turns or MAX_RECENT_TURNS
    messages: list[Message] = []

    # Get preferences with defaults
    prefs = {**DEFAULT_PREFERENCES, **campaign.preferences}
    rag_level = prefs.get("rag_aggressiveness", "moderate")
    uncertainty_mode = prefs.get("uncertainty_mode", "gm")

    # Build system prompt with campaign context
    system_parts = [GM_SYSTEM_PROMPT]

    # Add RAG aggressiveness prompt
    if rag_level in RAG_PROMPTS:
        system_parts.append(RAG_PROMPTS[rag_level])

    # Add uncertainty mode prompt
    if uncertainty_mode in UNCERTAINTY_PROMPTS:
        system_parts.append(UNCERTAINTY_PROMPTS[uncertainty_mode])

    # Layer 1: Campaign background
    if campaign.background:
        system_parts.append(f"\n## Campaign Background\n{campaign.background}")

    # Layer 2: Current arc
    if campaign.current_arc:
        system_parts.append(f"\n## Current Story Arc\n{campaign.current_arc}")

    # Party information
    if campaign.party:
        party_info = "\n## Party Members"
        for member in campaign.party:
            party_info += f"\n- **{member.name}**: {member.ancestry} {member.class_name} (Level {member.level})"
            if member.notes:
                party_info += f" - {member.notes}"
        system_parts.append(party_info)

    # Layer 3: Session summary
    if session.summary:
        system_parts.append(f"\n## Previous Session Summary\n{session.summary}")

    # Layer 5: Current scene state
    # Try Foundry VTT for live scene data, fall back to JSON scene state
    scene_info = _build_scene_info(session, foundry_server)
    if scene_info:
        system_parts.append(scene_info)

    # Combine system message
    messages.append(Message(role="system", content="\n".join(system_parts)))

    # Layer 4: Recent turns
    recent_turns = session.turns[-max_turns:] if session.turns else []
    for turn in recent_turns:
        # Player input
        messages.append(Message(role="user", content=turn.player_input))

        # GM response (with tool calls if any)
        if turn.tool_calls:
            # Add assistant message that made tool calls
            messages.append(
                Message(
                    role="assistant",
                    content="",  # Empty content when making tool calls
                    tool_calls=[
                        ToolCall(id=f"call_{i}", name=tc.name, args=tc.args)
                        for i, tc in enumerate(turn.tool_calls)
                    ],
                )
            )

            # Add tool results
            for i, tc in enumerate(turn.tool_calls):
                messages.append(
                    Message(
                        role="tool",
                        content=tc.result,
                        tool_call_id=f"call_{i}",
                    )
                )

        # Final assistant response
        messages.append(Message(role="assistant", content=turn.gm_response))

    return messages


def _build_scene_info(
    session: Session,
    foundry_server: FoundryVTTServer | None = None,
) -> str | None:
    """Build scene information for Layer 5 context.

    Tries Foundry VTT for live data first, falls back to JSON scene state.
    """
    # Try Foundry VTT if available and connected
    if foundry_server and foundry_server.is_connected():
        try:
            result = foundry_server.call_tool("get_scene", {})
            if result.success:
                return f"\n## Current Scene (Live from Foundry)\n{result.data}"
        except Exception:
            pass  # Fall back to JSON scene state

    # Fall back to JSON scene state
    scene = session.scene_state
    if scene.location == "Unknown" and not scene.npcs_present and not scene.conditions:
        return None

    scene_info = "\n## Current Scene"
    scene_info += f"\n- **Location**: {scene.location}"
    if scene.time_of_day:
        scene_info += f"\n- **Time**: {scene.time_of_day}"
    if scene.npcs_present:
        scene_info += f"\n- **NPCs Present**: {', '.join(scene.npcs_present)}"
    if scene.conditions:
        scene_info += f"\n- **Conditions**: {', '.join(scene.conditions)}"
    if scene.notes:
        scene_info += f"\n- **Notes**: {scene.notes}"

    return scene_info


def estimate_tokens(messages: list[Message]) -> int:
    """Rough estimate of token count for messages."""
    total_chars = sum(len(m.content) for m in messages)
    # Rough estimate: 4 characters per token
    return total_chars // 4


def render_prompt_template(template: str, variables: dict[str, str]) -> str:
    """Render a prompt template with variables.

    Supports simple variable substitution with {variable_name} syntax.
    Missing variables are replaced with empty strings.

    Args:
        template: The template string with {variable} placeholders
        variables: Dictionary of variable names to values

    Returns:
        Rendered prompt with variables substituted

    Example:
        >>> render_prompt_template(
        ...     "It's {actor_name}'s turn. {content}",
        ...     {"actor_name": "Voz", "content": "What do you do?"}
        ... )
        "It's Voz's turn. What do you do?"
    """
    result = template
    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, str(value))
    return result
