"""Celery tasks for MCP tool execution.

These tasks handle MCP tool execution in Celery workers, enabling
distributed tool processing for the remote mode architecture.
"""

import logging
from typing import Any

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def _create_server(server_name: str, context: dict[str, Any]):
    """Create an MCP server instance.

    Args:
        server_name: Name of the server to create
        context: Execution context (campaign_id, session_id, etc.)

    Returns:
        MCPServer instance

    Raises:
        ValueError: If server name is unknown
    """
    campaign_id = context.get("campaign_id")

    if server_name == "pf2e-rag":
        from .mcp.pf2e_rag import PF2eRAGServer

        return PF2eRAGServer()

    elif server_name == "dice":
        from .mcp.dice import DiceServer

        return DiceServer()

    elif server_name == "encounter":
        from .mcp.encounter import EncounterServer

        return EncounterServer()

    elif server_name == "notes":
        # Check if Redis-backed store should be used
        session_id = context.get("session_id")
        if session_id:
            from .mcp.redis_notes import RedisNotesServer

            return RedisNotesServer(session_id=session_id)
        else:
            from .mcp.notes import NotesServer

            return NotesServer()

    elif server_name == "campaign-state":
        if not campaign_id:
            raise ValueError("campaign_id required for campaign-state server")
        from .mcp.campaign_state import CampaignStateServer

        return CampaignStateServer(campaign_id)

    elif server_name == "character-runner":
        if not campaign_id:
            raise ValueError("campaign_id required for character-runner server")
        from .mcp.character_runner import CharacterRunnerServer
        from .models.factory import get_backend

        llm = get_backend()
        return CharacterRunnerServer(campaign_id, llm=llm)

    else:
        raise ValueError(f"Unknown server: {server_name}")


@celery_app.task(bind=True, soft_time_limit=60, max_retries=1)
def execute_mcp_tool(
    self,
    server_name: str,
    tool_name: str,
    args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Execute an MCP tool in a Celery worker.

    This task creates the appropriate MCP server, executes the tool,
    and returns the result.

    Args:
        server_name: Name of the MCP server (e.g., "pf2e-rag", "dice")
        tool_name: Name of the tool to execute
        args: Tool arguments
        context: Execution context (campaign_id, session_id, etc.)

    Returns:
        dict with success, data, and error fields
    """
    server = None
    try:
        logger.info(f"Executing MCP tool: {tool_name} on {server_name}")

        # Create server instance
        server = _create_server(server_name, context)

        # Execute tool
        result = server.call_tool(tool_name, args)

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }

    except ValueError as e:
        # Invalid server/context
        logger.error(f"MCP tool execution failed: {e}")
        return {
            "success": False,
            "data": None,
            "error": str(e),
        }

    except Exception as e:
        logger.exception(f"MCP tool execution error: {e}")
        # Retry on unexpected errors
        raise self.retry(exc=e, countdown=5)

    finally:
        # Clean up server resources
        if server:
            try:
                server.close()
            except Exception:
                pass


@celery_app.task(bind=True, soft_time_limit=120, max_retries=1)
def execute_mcp_tools_batch(
    self,
    tool_calls: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute multiple MCP tools in a single worker.

    This task is more efficient for batch execution as it can reuse
    server instances across multiple tool calls.

    Args:
        tool_calls: List of dicts with server_name, tool_name, args
        context: Shared execution context

    Returns:
        List of results in the same order as input
    """
    from .mcp.registry import get_server_for_tool

    servers: dict[str, Any] = {}
    results = []

    try:
        for call in tool_calls:
            tool_name = call.get("tool_name")
            args = call.get("args", {})

            # Get server name for this tool
            server_name = call.get("server_name") or get_server_for_tool(tool_name)
            if not server_name:
                results.append(
                    {
                        "success": False,
                        "data": None,
                        "error": f"Unknown tool: {tool_name}",
                    }
                )
                continue

            # Get or create server
            if server_name not in servers:
                try:
                    servers[server_name] = _create_server(server_name, context)
                except ValueError as e:
                    results.append(
                        {
                            "success": False,
                            "data": None,
                            "error": str(e),
                        }
                    )
                    continue

            # Execute tool
            try:
                result = servers[server_name].call_tool(tool_name, args)
                results.append(
                    {
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "data": None,
                        "error": str(e),
                    }
                )

        return results

    except Exception as e:
        logger.exception(f"Batch MCP tool execution error: {e}")
        raise self.retry(exc=e, countdown=5)

    finally:
        # Clean up all server resources
        for server in servers.values():
            try:
                server.close()
            except Exception:
                pass
