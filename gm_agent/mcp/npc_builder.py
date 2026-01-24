"""NPC Builder MCP server for automatic character profile generation from RAG."""

from typing import Any

from ..config import CAMPAIGNS_DIR
from ..models.base import LLMBackend, Message
from ..storage.characters import CharacterStore
from ..storage.schemas import CharacterProfile
from .base import MCPServer, ToolDef, ToolParameter, ToolResult
from .pf2e_rag import PF2eRAGServer


# System prompt for extracting character information
EXTRACTION_PROMPT = """You are analyzing Pathfinder 2e content to build a character profile.

Based on the provided information about a character, extract the following details:

**For NPCs:**
- Personality: Core personality traits and demeanor (2-3 sentences)
- Speech patterns: How they talk, common phrases, accent (1-2 sentences)
- Knowledge: What they know (list of specific facts/topics)
- Goals: What they want or are trying to achieve (list)
- Secrets: Information they hide or don't share freely (list)

**For Monsters:**
- Personality: Behavioral traits and nature
- Intelligence level: animal/low/average/high/genius
- Instincts: Natural behaviors and drives (list)
- Morale: When/how they flee or fight to the death
- Goals: What drives them (food, territory, orders, etc.)

If information is missing, leave those fields empty. Be specific and reference the source material.

Format your response as JSON with these fields:
```json
{
  "character_type": "npc" or "monster",
  "personality": "...",
  "speech_patterns": "...",
  "knowledge": ["...", "..."],
  "goals": ["...", "..."],
  "secrets": ["...", "..."],
  "intelligence": "average",
  "instincts": ["...", "..."],
  "morale": "..."
}
```

Return ONLY the JSON, no other text."""


class NPCBuilderServer(MCPServer):
    """MCP server for building NPC profiles from RAG data.

    Automatically generates character profiles by querying the RAG system
    and using an LLM to extract personality, goals, and other traits.
    """

    def __init__(self, campaign_id: str, llm: LLMBackend | None = None):
        """Initialize the NPC builder server.

        Args:
            campaign_id: Campaign ID for storing character profiles
            llm: LLM backend for extraction (required)
        """
        self.campaign_id = campaign_id
        self.llm = llm
        self._character_store = CharacterStore(campaign_id, base_dir=CAMPAIGNS_DIR)
        self._rag_server = PF2eRAGServer()
        self._tools = self._build_tools()

    def _build_tools(self) -> list[ToolDef]:
        """Build the tool definitions."""
        return [
            ToolDef(
                name="build_npc_profile",
                description="Automatically build an NPC or monster character profile from Pathfinder content. Queries RAG for information and extracts personality, goals, and knowledge.",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="Name of the NPC or creature to build a profile for (e.g., 'Voz Lirayne', 'Goblin Warrior')",
                    ),
                    ToolParameter(
                        name="force_rebuild",
                        type="boolean",
                        description="If true, rebuild the profile even if one already exists",
                        required=False,
                        default=False,
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        """List all available tools."""
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool by name with arguments."""
        if name == "build_npc_profile":
            return self._build_npc_profile(
                args["name"],
                args.get("force_rebuild", False)
            )
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def _build_npc_profile(self, name: str, force_rebuild: bool = False) -> ToolResult:
        """Build an NPC profile from RAG data.

        Args:
            name: Character name to build profile for
            force_rebuild: If true, rebuild even if profile exists

        Returns:
            ToolResult with the created CharacterProfile
        """
        if not self.llm:
            return ToolResult(
                success=False,
                error="LLM backend required for NPC profile building"
            )

        # Check if profile already exists
        existing = self._character_store.get_by_name(name)
        if existing and not force_rebuild:
            return ToolResult(
                success=True,
                data=f"Character profile for '{name}' already exists (ID: {existing.id}). Use force_rebuild=true to recreate."
            )

        # Query RAG for information about this character
        rag_data = self._gather_rag_data(name)

        if not rag_data:
            return ToolResult(
                success=False,
                error=f"No information found for '{name}' in Pathfinder content. Cannot build profile."
            )

        # Use LLM to extract character details
        try:
            extracted_data = self._extract_character_data(name, rag_data)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to extract character data: {e}"
            )

        # Create or update the character profile
        try:
            if existing and force_rebuild:
                # Update existing profile
                profile = existing
                for key, value in extracted_data.items():
                    if value:  # Only update non-empty fields
                        setattr(profile, key, value)
                profile = self._character_store.update(profile)
                action = "updated"
            else:
                # Create new profile
                profile = self._character_store.create(
                    name=name,
                    **extracted_data
                )
                action = "created"

            return ToolResult(
                success=True,
                data=f"Character profile {action} for '{name}' (ID: {profile.id})\n"
                     f"Type: {profile.character_type}\n"
                     f"Personality: {profile.personality[:100]}...\n"
                     f"Goals: {', '.join(profile.goals[:3]) if profile.goals else 'None'}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to save character profile: {e}"
            )

    def _gather_rag_data(self, name: str) -> str:
        """Gather information about a character from RAG.

        Args:
            name: Character name to search for

        Returns:
            Combined text from RAG results, or empty string if nothing found
        """
        data_parts = []

        # Try creature lookup first (for monsters)
        creature_result = self._rag_server.call_tool("lookup_creature", {"name": name})
        if creature_result.success and creature_result.data:
            data_parts.append(f"=== Creature Data ===\n{creature_result.data}")

        # Search general content for NPCs and named characters
        content_result = self._rag_server.call_tool(
            "search_content",
            {
                "query": name,
                "types": "npc,creature,page_text",
                "limit": 5
            }
        )
        if content_result.success and content_result.data:
            data_parts.append(f"\n=== Content Search ===\n{content_result.data}")

        # Search lore for additional context
        lore_result = self._rag_server.call_tool(
            "search_lore",
            {"query": name, "limit": 3}
        )
        if lore_result.success and lore_result.data:
            data_parts.append(f"\n=== Lore ===\n{lore_result.data}")

        return "\n".join(data_parts)

    def _extract_character_data(self, name: str, rag_data: str) -> dict[str, Any]:
        """Use LLM to extract structured character data from RAG results.

        Args:
            name: Character name
            rag_data: Raw text from RAG queries

        Returns:
            Dictionary of character attributes
        """
        # Build extraction prompt
        messages = [
            Message(role="system", content=EXTRACTION_PROMPT),
            Message(
                role="user",
                content=f"Character: {name}\n\nInformation:\n{rag_data}"
            )
        ]

        # Get LLM extraction
        response = self.llm.chat(messages, tools=[])

        # Parse JSON response
        import json
        try:
            # Try to extract JSON from markdown code blocks
            text = response.text.strip()
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            data = json.loads(text)

            # Validate and clean up the data
            result = {
                "character_type": data.get("character_type", "npc"),
                "personality": data.get("personality", ""),
                "speech_patterns": data.get("speech_patterns", ""),
                "knowledge": data.get("knowledge", []),
                "goals": data.get("goals", []),
                "secrets": data.get("secrets", []),
            }

            # Add monster-specific fields if applicable
            if result["character_type"] == "monster":
                result["intelligence"] = data.get("intelligence", "average")
                result["instincts"] = data.get("instincts", [])
                result["morale"] = data.get("morale", "")

            return result

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: return minimal data
            return {
                "character_type": "npc",
                "personality": f"Character from Pathfinder content (auto-generated)",
                "speech_patterns": "",
                "knowledge": [f"Information about {name}"],
                "goals": [],
                "secrets": [],
                "notes": f"Failed to parse LLM extraction: {e}. Raw data available in RAG."
            }

    def close(self) -> None:
        """Clean up resources."""
        if self._rag_server:
            self._rag_server.close()
