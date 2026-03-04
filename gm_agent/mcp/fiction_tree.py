"""Fiction Tree MCP server.

Provides read/write/navigate tools for the shared Fiction Tree,
available to all game systems. Campaign-scoped.
"""

import json
from typing import Any

from ..config import CAMPAIGNS_DIR
from ..storage.fiction_tree import FictionTreeStore, FictionNode
from .base import MCPServer, ToolDef, ToolParameter, ToolResult


class FictionTreeServer(MCPServer):
    """MCP server for the Fiction Tree (shared narrative index).

    Campaign-scoped server providing CRUD, navigation, and search
    across the hierarchical fiction tree.
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._store: FictionTreeStore | None = None
        self._tools = self._build_tools()

    @property
    def store(self) -> FictionTreeStore:
        if self._store is None:
            self._store = FictionTreeStore(self.campaign_id, base_dir=CAMPAIGNS_DIR)
        return self._store

    def _build_tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="browse_fiction_tree",
                description=(
                    "Browse the fiction tree. With no arguments, lists root nodes. "
                    "With a node_id, lists its children. Returns titles, zoom levels, "
                    "tones, and summaries."
                ),
                parameters=[
                    ToolParameter(
                        name="node_id",
                        type="string",
                        description="Parent node ID to browse children of. Omit for roots.",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="get_fiction_node",
                description=(
                    "Get full detail for a single fiction tree node including content, "
                    "cross-references, palette, and ancestry path."
                ),
                parameters=[
                    ToolParameter(
                        name="node_id",
                        type="string",
                        description="The fiction node ID to retrieve.",
                    ),
                ],
            ),
            ToolDef(
                name="search_fiction",
                description="Full-text search across the fiction tree.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query.",
                    ),
                    ToolParameter(
                        name="zoom_level",
                        type="string",
                        description="Filter by zoom level: era, period, event, scene, detail.",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="add_fiction_node",
                description=(
                    "Create a new node in the fiction tree. Returns the created node. "
                    "Used by generation games and for manual edits."
                ),
                parameters=[
                    ToolParameter(
                        name="title",
                        type="string",
                        description="Node title.",
                    ),
                    ToolParameter(
                        name="zoom_level",
                        type="string",
                        description="Zoom level: era, period, event, scene, detail.",
                    ),
                    ToolParameter(
                        name="parent_id",
                        type="string",
                        description="Parent node ID. Omit for root nodes.",
                        required=False,
                    ),
                    ToolParameter(
                        name="tone",
                        type="string",
                        description="Tone: light, dark, or ambiguous.",
                        required=False,
                    ),
                    ToolParameter(
                        name="summary",
                        type="string",
                        description="Brief summary of the node.",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="Full description/content.",
                        required=False,
                        default="",
                    ),
                    ToolParameter(
                        name="source_system",
                        type="string",
                        description="Source system: microscope, ex_novo, delve, ex_umbra, manual.",
                        required=False,
                        default="manual",
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="JSON array of tags.",
                        required=False,
                        default="[]",
                    ),
                ],
            ),
            ToolDef(
                name="update_fiction_node",
                description="Edit a fiction tree node's title, summary, content, tone, or tags.",
                parameters=[
                    ToolParameter(
                        name="node_id",
                        type="string",
                        description="The node ID to update.",
                    ),
                    ToolParameter(
                        name="title",
                        type="string",
                        description="New title.",
                        required=False,
                    ),
                    ToolParameter(
                        name="summary",
                        type="string",
                        description="New summary.",
                        required=False,
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="New content.",
                        required=False,
                    ),
                    ToolParameter(
                        name="tone",
                        type="string",
                        description="New tone: light, dark, ambiguous.",
                        required=False,
                    ),
                    ToolParameter(
                        name="tags",
                        type="string",
                        description="JSON array of tags.",
                        required=False,
                    ),
                ],
            ),
            ToolDef(
                name="link_fiction_node",
                description=(
                    "Add cross-references from a fiction node to characters, "
                    "locations, knowledge entries, or other entities."
                ),
                parameters=[
                    ToolParameter(
                        name="node_id",
                        type="string",
                        description="The fiction node to add links to.",
                    ),
                    ToolParameter(
                        name="link_type",
                        type="string",
                        description="Link category: characters, locations, knowledge, items.",
                    ),
                    ToolParameter(
                        name="entity_ids",
                        type="string",
                        description="JSON array of entity IDs to link.",
                    ),
                ],
            ),
        ]

    def list_tools(self) -> list[ToolDef]:
        return self._tools

    def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "browse_fiction_tree":
                return self._browse(args.get("node_id"))
            elif name == "get_fiction_node":
                return self._get_node(args["node_id"])
            elif name == "search_fiction":
                return self._search(args["query"], args.get("zoom_level"))
            elif name == "add_fiction_node":
                return self._add_node(args)
            elif name == "update_fiction_node":
                return self._update_node(args)
            elif name == "link_fiction_node":
                return self._link_node(args)
            else:
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _browse(self, node_id: str | None) -> ToolResult:
        children = self.store.get_children(node_id)
        if not children:
            if node_id:
                return ToolResult(success=True, data="No children for this node.")
            return ToolResult(success=True, data="Fiction tree is empty.")

        lines = []
        if node_id:
            parent = self.store.get_node(node_id)
            if parent:
                lines.append(f"**Children of: {parent.title}** ({parent.zoom_level})\n")
        else:
            lines.append("**Fiction Tree Roots**\n")

        for child in children:
            tone = f" [{child.tone}]" if child.tone else ""
            lines.append(f"- **{child.title}**{tone} ({child.zoom_level}) — ID: {child.id}")
            if child.summary:
                lines.append(f"  {child.summary[:120]}")

        return ToolResult(success=True, data="\n".join(lines))

    def _get_node(self, node_id: str) -> ToolResult:
        node = self.store.get_node(node_id)
        if node is None:
            return ToolResult(success=False, error=f"Node '{node_id}' not found.")

        ancestors = self.store.get_ancestors(node_id)
        path = " > ".join(a.title for a in ancestors)
        if path:
            path += f" > {node.title}"
        else:
            path = node.title

        lines = [
            f"**{node.title}** ({node.zoom_level})",
            f"**Path:** {path}",
        ]
        if node.tone:
            lines.append(f"**Tone:** {node.tone}")
        if node.summary:
            lines.append(f"**Summary:** {node.summary}")
        if node.content:
            lines.append(f"\n{node.content}")
        if node.tags:
            lines.append(f"\n**Tags:** {', '.join(node.tags)}")
        if node.linked_entities:
            lines.append("\n**Linked Entities:**")
            for link_type, ids in node.linked_entities.items():
                lines.append(f"  {link_type}: {', '.join(ids)}")
        if node.palette_yes:
            lines.append(f"\n**Palette YES:** {', '.join(node.palette_yes)}")
        if node.palette_no:
            lines.append(f"**Palette NO:** {', '.join(node.palette_no)}")

        lines.append(f"\n**ID:** {node.id} | **Source:** {node.source_system}")

        return ToolResult(success=True, data="\n".join(lines))

    def _search(self, query: str, zoom_level: str | None) -> ToolResult:
        results = self.store.search(query, zoom_level=zoom_level)
        if not results:
            return ToolResult(success=True, data=f"No results for '{query}'.")

        lines = [f"**Fiction Tree Search:** '{query}' ({len(results)} results)\n"]
        for node in results:
            tone = f" [{node.tone}]" if node.tone else ""
            lines.append(f"- **{node.title}**{tone} ({node.zoom_level}) — ID: {node.id}")
            if node.summary:
                lines.append(f"  {node.summary[:120]}")

        return ToolResult(success=True, data="\n".join(lines))

    def _add_node(self, args: dict[str, Any]) -> ToolResult:
        tags = args.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)

        node = FictionNode(
            campaign_id=self.campaign_id,
            parent_id=args.get("parent_id"),
            zoom_level=args["zoom_level"],
            tone=args.get("tone"),
            title=args["title"],
            summary=args.get("summary", ""),
            content=args.get("content", ""),
            source_system=args.get("source_system", "manual"),
            tags=tags,
        )

        node = self.store.add_node(node)

        return ToolResult(
            success=True,
            data=(
                f"**Created:** {node.title} ({node.zoom_level})\n"
                f"**ID:** {node.id}\n"
                f"**Parent:** {node.parent_id or 'root'}"
            ),
        )

    def _update_node(self, args: dict[str, Any]) -> ToolResult:
        node_id = args.pop("node_id")
        updates = {}
        for key in ("title", "summary", "content", "tone"):
            if key in args and args[key] is not None:
                updates[key] = args[key]
        if "tags" in args and args["tags"] is not None:
            tags = args["tags"]
            if isinstance(tags, str):
                tags = json.loads(tags)
            updates["tags"] = tags

        if not updates:
            return ToolResult(success=False, error="No fields to update.")

        node = self.store.update_node(node_id, **updates)
        if node is None:
            return ToolResult(success=False, error=f"Node '{node_id}' not found.")

        return ToolResult(
            success=True,
            data=f"**Updated:** {node.title} ({node.zoom_level}) — ID: {node.id}",
        )

    def _link_node(self, args: dict[str, Any]) -> ToolResult:
        node_id = args["node_id"]
        link_type = args["link_type"]
        entity_ids = args["entity_ids"]
        if isinstance(entity_ids, str):
            entity_ids = json.loads(entity_ids)

        node = self.store.get_node(node_id)
        if node is None:
            return ToolResult(success=False, error=f"Node '{node_id}' not found.")

        linked = dict(node.linked_entities)
        existing = linked.get(link_type, [])
        for eid in entity_ids:
            if eid not in existing:
                existing.append(eid)
        linked[link_type] = existing

        self.store.update_node(node_id, linked_entities=linked)

        return ToolResult(
            success=True,
            data=f"Linked {len(entity_ids)} {link_type} to '{node.title}'.",
        )

    def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
