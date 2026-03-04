"""Grounding server — links fiction tree nodes to PF2e mechanical content.

Enables the "generate -> link -> ground -> play" pipeline: fiction tree
nodes from generation games get grounded into PF2e mechanical entities
via RAG search and optional LLM-driven selection.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

from ..config import TEMPERATURE_MECHANICAL
from .base import MCPServer, ToolDef, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..rag.search import PathfinderSearch
    from ..storage.fiction_tree import FictionTreeStore
    from ..models.base import LLMBackend

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """\
You are a Pathfinder 2e grounding assistant. Given a fiction element and a list of \
candidate PF2e mechanical matches, select the most relevant matches and explain why \
each is a good fit. Return a JSON array of objects with keys: index, relevance (1-10), reasoning."""

RERANK_USER_TEMPLATE = """\
**Fiction element:** {title}
**Description:** {description}

**Candidates:**
{candidates}

Select the top {top_k} most relevant candidates. Return a JSON array:
[{{"index": 0, "relevance": 8, "reasoning": "..."}}]"""


class GroundingServer(MCPServer):
    """MCP server for grounding fiction tree nodes into PF2e mechanics.

    Requires a fiction tree store and a PF2e RAG search instance.
    Optionally uses an LLM for intelligent match selection.
    """

    def __init__(
        self,
        campaign_id: str,
        fiction_store: FictionTreeStore,
        search: PathfinderSearch,
        llm: LLMBackend | None = None,
    ):
        self.campaign_id = campaign_id
        self.fiction_store = fiction_store
        self.search = search
        self.llm = llm
        self._tools = self._build_tools()

    def _build_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="ground_node",
                description=(
                    "Ground a fiction tree node into PF2e mechanics by searching "
                    "for matching rules content, creatures, items, or lore."
                ),
                parameters=[
                    ToolParameter(name="node_id", type="string",
                                  description="Fiction tree node ID to ground."),
                    ToolParameter(name="content_types", type="string",
                                  description="Comma-separated PF2e content types to search (e.g. 'creature,npc,item'). "
                                  "Leave empty to search all.",
                                  required=False, default=""),
                ],
            ),
            ToolDef(
                name="suggest_groundings",
                description=(
                    "Analyze a fiction subtree and suggest which nodes should be "
                    "grounded into PF2e mechanics."
                ),
                parameters=[
                    ToolParameter(name="root_node_id", type="string",
                                  description="Root node ID of the subtree to analyze."),
                    ToolParameter(name="max_suggestions", type="integer",
                                  description="Maximum suggestions to return.",
                                  required=False, default=10),
                ],
            ),
            ToolDef(
                name="ground_settlement",
                description=(
                    "Ground an Ex Novo settlement into PF2e: suggest matching "
                    "locations, NPCs, factions, and local creatures."
                ),
                parameters=[
                    ToolParameter(name="node_id", type="string",
                                  description="Fiction tree node ID of the settlement."),
                ],
            ),
            ToolDef(
                name="ground_dungeon",
                description=(
                    "Ground a Delve/Ex Umbra dungeon into PF2e: suggest matching "
                    "encounters, creatures, hazards, and treasure."
                ),
                parameters=[
                    ToolParameter(name="node_id", type="string",
                                  description="Fiction tree node ID of the dungeon."),
                    ToolParameter(name="party_level", type="integer",
                                  description="Party level for encounter scaling.",
                                  required=False, default=1),
                ],
            ),
            ToolDef(
                name="ground_timeline",
                description=(
                    "Ground a Microscope timeline into PF2e: suggest matching "
                    "world lore, historical events, and knowledge entries."
                ),
                parameters=[
                    ToolParameter(name="node_id", type="string",
                                  description="Fiction tree node ID of the timeline root."),
                ],
            ),
            ToolDef(
                name="list_groundings",
                description="List all grounded PF2e entities linked to a fiction node.",
                parameters=[
                    ToolParameter(name="node_id", type="string",
                                  description="Fiction tree node ID to query."),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "ground_node":
                return self._ground_node(args)
            elif name == "suggest_groundings":
                return self._suggest_groundings(args)
            elif name == "ground_settlement":
                return self._ground_settlement(args)
            elif name == "ground_dungeon":
                return self._ground_dungeon(args)
            elif name == "ground_timeline":
                return self._ground_timeline(args)
            elif name == "list_groundings":
                return self._list_groundings(args)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    # -------------------------------------------------------------------
    # LLM reranking
    # -------------------------------------------------------------------

    def _llm_rerank(
        self,
        results: list[dict[str, Any]],
        title: str,
        description: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Use LLM to rerank search results by relevance to the fiction element.

        Falls back to raw results on any error.
        """
        if not results:
            return results

        from ..models.base import Message

        # Build candidate list
        candidate_lines = []
        for i, r in enumerate(results):
            name = r.get("name", "Unknown")
            rtype = r.get("type", "?")
            book = r.get("book", "?")
            preview = (r.get("content") or "")[:100].replace("\n", " ")
            candidate_lines.append(f"{i}. **{name}** ({rtype}) — {book}: {preview}")

        user_msg = RERANK_USER_TEMPLATE.format(
            title=title,
            description=description or "(no description)",
            candidates="\n".join(candidate_lines),
            top_k=min(top_k, len(results)),
        )

        try:
            response = self.llm.chat(
                [
                    Message(role="system", content=RERANK_SYSTEM_PROMPT),
                    Message(role="user", content=user_msg),
                ],
                temperature=TEMPERATURE_MECHANICAL,
            )

            # Parse JSON array from response
            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

            # Try to find JSON array
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                rankings = json.loads(match.group())
            else:
                rankings = json.loads(text)

            if not isinstance(rankings, list):
                logger.warning("LLM rerank returned non-list, falling back to raw results")
                return results[:top_k]

            # Rebuild results in ranked order with reasoning
            reranked = []
            for entry in rankings[:top_k]:
                idx = int(entry.get("index", -1))
                if 0 <= idx < len(results):
                    r = dict(results[idx])
                    r["grounding_relevance"] = entry.get("relevance", 0)
                    r["grounding_reasoning"] = entry.get("reasoning", "")
                    reranked.append(r)

            return reranked if reranked else results[:top_k]

        except Exception as e:
            logger.warning("LLM reranking failed, falling back to raw results: %s", e)
            return results[:top_k]

    # -------------------------------------------------------------------
    # ground_node
    # -------------------------------------------------------------------

    def _ground_node(self, args: dict[str, Any]) -> ToolResult:
        node = self.fiction_store.get_node(args["node_id"])
        if node is None:
            return ToolResult(success=False, error=f"Node '{args['node_id']}' not found.")

        # Build search query from node content
        query = f"{node.title} {node.summary}".strip()
        if not query:
            return ToolResult(success=False, error="Node has no title or summary to search with.")

        # Parse content type filter
        content_types_str = args.get("content_types", "")
        include_types = [t.strip() for t in content_types_str.split(",") if t.strip()] or None

        results = self.search.search(query, include_types=include_types, top_k=10)

        if not results:
            return ToolResult(
                success=True,
                data=f"No PF2e matches found for '{node.title}'. Try broader search terms.",
            )

        # LLM reranking if available
        if self.llm:
            results = self._llm_rerank(results, node.title, node.summary or "")

        lines = [f"**Grounding suggestions for:** {node.title}\n"]
        rag_refs = []
        for r in results:
            lines.append(f"- **{r.get('name', 'Unknown')}** ({r.get('type', '?')}) "
                         f"— {r.get('book', '?')} [score: {abs(r.get('score', 0)):.1f}]")
            if r.get("grounding_reasoning"):
                lines.append(f"  *{r['grounding_reasoning']}*")
            if r.get("content"):
                preview = r["content"][:120].replace("\n", " ")
                lines.append(f"  {preview}...")
            rag_refs.append(r.get("id", r.get("name", "")))

        # Store references on the node
        linked = dict(node.linked_entities)
        linked["rag_refs"] = rag_refs
        self.fiction_store.update_node(args["node_id"], linked_entities=linked)

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # suggest_groundings
    # -------------------------------------------------------------------

    def _suggest_groundings(self, args: dict[str, Any]) -> ToolResult:
        root_node = self.fiction_store.get_node(args["root_node_id"])
        if root_node is None:
            return ToolResult(success=False, error=f"Node '{args['root_node_id']}' not found.")

        max_suggestions = int(args.get("max_suggestions", 10))

        # Walk the subtree and find nodes that could be grounded
        suggestions = []
        self._walk_for_suggestions(args["root_node_id"], suggestions, max_suggestions)

        if not suggestions:
            return ToolResult(
                success=True,
                data="No groundable nodes found in this subtree.",
            )

        lines = [f"**Grounding suggestions for subtree:** {root_node.title}\n"]
        for node_id, title, zoom_level, reason in suggestions:
            lines.append(f"- **{title}** ({zoom_level}) — ID: {node_id}")
            lines.append(f"  Suggestion: {reason}")

        return ToolResult(success=True, data="\n".join(lines))

    def _walk_for_suggestions(
        self,
        node_id: str,
        suggestions: list[tuple[str, str, str, str]],
        max_count: int,
    ) -> None:
        """Recursively walk subtree looking for groundable nodes."""
        if len(suggestions) >= max_count:
            return

        node = self.fiction_store.get_node(node_id)
        if node is None:
            return

        # Check if this node is worth grounding
        already_grounded = bool(node.linked_entities.get("rag_refs"))
        has_content = bool(node.title.strip())

        if has_content and not already_grounded and node.zoom_level in ("event", "scene", "detail"):
            reason = self._suggest_reason(node)
            if reason:
                suggestions.append((node.id, node.title, node.zoom_level, reason))

        # Walk children
        children = self.fiction_store.get_children(node_id)
        for child in children:
            if len(suggestions) >= max_count:
                break
            self._walk_for_suggestions(child.id, suggestions, max_count)

    def _suggest_reason(self, node: Any) -> str:
        """Suggest why a node should be grounded."""
        title_lower = node.title.lower()
        content_lower = (node.summary or "").lower() + " " + (node.content or "").lower()
        combined = title_lower + " " + content_lower

        if any(w in combined for w in ("battle", "fight", "war", "attack", "siege", "conflict")):
            return "Contains conflict — ground with creatures and encounters"
        if any(w in combined for w in ("creature", "beast", "monster", "dragon", "demon")):
            return "References creatures — ground with PF2e bestiary entries"
        if any(w in combined for w in ("treasure", "artifact", "weapon", "armor", "magic item")):
            return "References items — ground with PF2e equipment/items"
        if any(w in combined for w in ("city", "town", "village", "settlement", "district")):
            return "References a location — ground with PF2e setting content"
        if any(w in combined for w in ("character", "hero", "villain", "king", "queen", "wizard")):
            return "References notable characters — ground with NPC templates"
        if any(w in combined for w in ("spell", "magic", "ritual", "enchantment")):
            return "References magic — ground with PF2e spells/rituals"
        if any(w in combined for w in ("trap", "hazard", "puzzle", "locked")):
            return "References hazards — ground with PF2e hazards"

        return ""

    # -------------------------------------------------------------------
    # ground_settlement (Ex Novo → PF2e)
    # -------------------------------------------------------------------

    def _ground_settlement(self, args: dict[str, Any]) -> ToolResult:
        node = self.fiction_store.get_node(args["node_id"])
        if node is None:
            return ToolResult(success=False, error=f"Node '{args['node_id']}' not found.")

        query = f"{node.title} settlement"
        children = self.fiction_store.get_children(args["node_id"])

        lines = [f"**Grounding settlement:** {node.title}\n"]
        all_refs: dict[str, list[str]] = {"locations": [], "npcs": [], "creatures": [], "factions": []}

        # Search for the settlement itself
        results = self.search.search(query, include_types=["settlement", "landmark", "location"], top_k=5)
        if self.llm and results:
            results = self._llm_rerank(results, node.title, node.summary or "", top_k=5)
        if results:
            lines.append("**Location matches:**")
            for r in results:
                lines.append(f"  - {r.get('name', '?')} ({r.get('type', '?')}) — {r.get('book', '?')}")
                if r.get("grounding_reasoning"):
                    lines.append(f"    *{r['grounding_reasoning']}*")
                all_refs["locations"].append(r.get("id", r.get("name", "")))

        # Search for each notable child node
        for child in children[:10]:
            child_query = f"{child.title} {child.summary or ''}"
            child_results = self.search.search(child_query, top_k=3)
            if child_results:
                lines.append(f"\n**{child.title}:**")
                for r in child_results:
                    rtype = r.get("type", "?")
                    lines.append(f"  - {r.get('name', '?')} ({rtype}) — {r.get('book', '?')}")
                    if rtype in ("npc", "character"):
                        all_refs["npcs"].append(r.get("id", r.get("name", "")))
                    elif rtype == "creature":
                        all_refs["creatures"].append(r.get("id", r.get("name", "")))

        # Store references
        linked = dict(node.linked_entities)
        linked.update(all_refs)
        self.fiction_store.update_node(args["node_id"], linked_entities=linked)

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # ground_dungeon (Delve/Ex Umbra → PF2e)
    # -------------------------------------------------------------------

    def _ground_dungeon(self, args: dict[str, Any]) -> ToolResult:
        node = self.fiction_store.get_node(args["node_id"])
        if node is None:
            return ToolResult(success=False, error=f"Node '{args['node_id']}' not found.")

        party_level = int(args.get("party_level", 1))
        children = self.fiction_store.get_children(args["node_id"])

        lines = [f"**Grounding dungeon:** {node.title} (party level {party_level})\n"]
        all_refs: dict[str, list[str]] = {"encounters": [], "creatures": [], "hazards": [], "items": []}

        for child in children[:15]:
            child_lower = (child.title + " " + (child.summary or "")).lower()

            # Determine what to search for based on node content
            if any(w in child_lower for w in ("trap", "hazard", "puzzle")):
                results = self.search.search(child.title, include_types=["hazard"], top_k=3)
                category = "hazards"
            elif any(w in child_lower for w in ("treasure", "loot", "reward", "chest")):
                results = self.search.search(child.title, include_types=["item", "equipment"], top_k=3)
                category = "items"
            else:
                results = self.search.search(child.title, include_types=["creature", "npc"], top_k=3)
                category = "creatures"

            if self.llm and results:
                results = self._llm_rerank(results, child.title, child.summary or "", top_k=3)
            if results:
                lines.append(f"**{child.title}:**")
                for r in results:
                    lines.append(f"  - {r.get('name', '?')} ({r.get('type', '?')}) — {r.get('book', '?')}")
                    if r.get("grounding_reasoning"):
                        lines.append(f"    *{r['grounding_reasoning']}*")
                    all_refs[category].append(r.get("id", r.get("name", "")))

        # Store references
        linked = dict(node.linked_entities)
        linked.update(all_refs)
        self.fiction_store.update_node(args["node_id"], linked_entities=linked)

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # ground_timeline (Microscope → PF2e)
    # -------------------------------------------------------------------

    def _ground_timeline(self, args: dict[str, Any]) -> ToolResult:
        node = self.fiction_store.get_node(args["node_id"])
        if node is None:
            return ToolResult(success=False, error=f"Node '{args['node_id']}' not found.")

        periods = self.fiction_store.get_children(args["node_id"])

        lines = [f"**Grounding timeline:** {node.title}\n"]
        all_refs: dict[str, list[str]] = {"lore": [], "creatures": [], "npcs": []}

        for period in periods:
            events = self.fiction_store.get_children(period.id)
            for event in events[:5]:
                query = f"{event.title} {event.summary or ''}"
                results = self.search.search(query, top_k=3)
                if self.llm and results:
                    results = self._llm_rerank(results, event.title, event.summary or "", top_k=3)
                if results:
                    lines.append(f"**{period.title} > {event.title}:**")
                    for r in results:
                        rtype = r.get("type", "?")
                        lines.append(f"  - {r.get('name', '?')} ({rtype}) — {r.get('book', '?')}")
                        if r.get("grounding_reasoning"):
                            lines.append(f"    *{r['grounding_reasoning']}*")
                        ref_id = r.get("id", r.get("name", ""))
                        if rtype in ("creature",):
                            all_refs["creatures"].append(ref_id)
                        elif rtype in ("npc", "character", "deity"):
                            all_refs["npcs"].append(ref_id)
                        else:
                            all_refs["lore"].append(ref_id)

        # Store references
        linked = dict(node.linked_entities)
        linked.update(all_refs)
        self.fiction_store.update_node(args["node_id"], linked_entities=linked)

        return ToolResult(success=True, data="\n".join(lines))

    # -------------------------------------------------------------------
    # list_groundings
    # -------------------------------------------------------------------

    def _list_groundings(self, args: dict[str, Any]) -> ToolResult:
        node = self.fiction_store.get_node(args["node_id"])
        if node is None:
            return ToolResult(success=False, error=f"Node '{args['node_id']}' not found.")

        linked = node.linked_entities
        if not linked:
            return ToolResult(
                success=True,
                data=f"No groundings linked to '{node.title}'.",
            )

        lines = [f"**Groundings for:** {node.title}\n"]
        for category, refs in linked.items():
            if refs:
                lines.append(f"**{category.replace('_', ' ').title()}:** ({len(refs)})")
                for ref in refs[:10]:
                    lines.append(f"  - {ref}")
                if len(refs) > 10:
                    lines.append(f"  ... and {len(refs) - 10} more")

        return ToolResult(success=True, data="\n".join(lines))

    def close(self) -> None:
        """Clean up (fiction store and search managed externally)."""
        pass
