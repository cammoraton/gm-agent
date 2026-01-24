#!/usr/bin/env python3
"""CLI for Pathfinder 2e GM Agent."""

import sys
from pathlib import Path

import click

# Ensure gm_agent is importable
sys.path.insert(0, str(Path(__file__).parent))

from gm_agent.agent import GMAgent
from gm_agent.models.factory import get_backend, list_backends
from gm_agent.storage.campaign import campaign_store
from gm_agent.storage.session import session_store


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Pathfinder 2e GM Agent - AI-powered Game Master assistant."""
    pass


# ============================================================================
# Campaign commands
# ============================================================================


@cli.group()
def campaign():
    """Manage campaigns."""
    pass


@campaign.command("create")
@click.argument("name")
@click.option("--background", "-b", default="", help="Campaign background text")
def campaign_create(name: str, background: str):
    """Create a new campaign."""
    try:
        c = campaign_store.create(name, background=background)
        click.echo(f"Created campaign: {c.id}")
        click.echo(f"  Directory: {campaign_store._campaign_dir(c.id)}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@campaign.command("list")
def campaign_list():
    """List all campaigns."""
    campaigns = campaign_store.list()
    if not campaigns:
        click.echo("No campaigns found.")
        return

    click.echo("Campaigns:")
    for c in campaigns:
        sessions = session_store.list(c.id)
        current = session_store.get_current(c.id)
        status = " (active session)" if current else ""
        click.echo(f"  {c.id}: {c.name} [{len(sessions)} sessions]{status}")


@campaign.command("show")
@click.argument("campaign_id")
def campaign_show(campaign_id: str):
    """Show campaign details."""
    c = campaign_store.get(campaign_id)
    if not c:
        click.echo(f"Campaign '{campaign_id}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Campaign: {c.name}")
    click.echo(f"  ID: {c.id}")
    click.echo(f"  Created: {c.created_at}")

    if c.background:
        click.echo(f"\nBackground:\n  {c.background}")

    if c.current_arc:
        click.echo(f"\nCurrent Arc:\n  {c.current_arc}")

    if c.party:
        click.echo("\nParty:")
        for member in c.party:
            click.echo(f"  - {member.name}: {member.ancestry} {member.class_name} L{member.level}")

    sessions = session_store.list(c.id)
    if sessions:
        click.echo(f"\nSessions: {len(sessions)}")


@campaign.command("update")
@click.argument("campaign_id")
@click.option("--background", "-b", help="Update campaign background")
@click.option("--arc", "-a", help="Update current story arc")
def campaign_update(campaign_id: str, background: str | None, arc: str | None):
    """Update campaign details."""
    c = campaign_store.get(campaign_id)
    if not c:
        click.echo(f"Campaign '{campaign_id}' not found.", err=True)
        sys.exit(1)

    if background is not None:
        c.background = background
    if arc is not None:
        c.current_arc = arc

    campaign_store.update(c)
    click.echo(f"Updated campaign: {c.id}")


@campaign.command("delete")
@click.argument("campaign_id")
@click.confirmation_option(prompt="Are you sure you want to delete this campaign?")
def campaign_delete(campaign_id: str):
    """Delete a campaign and all its sessions."""
    if campaign_store.delete(campaign_id):
        click.echo(f"Deleted campaign: {campaign_id}")
    else:
        click.echo(f"Campaign '{campaign_id}' not found.", err=True)
        sys.exit(1)


# ============================================================================
# Session commands
# ============================================================================


@cli.group()
def session():
    """Manage sessions."""
    pass


@session.command("start")
@click.argument("campaign_id")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls")
def session_start(campaign_id: str, verbose: bool):
    """Start an interactive GM session (REPL)."""
    c = campaign_store.get(campaign_id)
    if not c:
        click.echo(f"Campaign '{campaign_id}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Starting session for: {c.name}")
    click.echo("Type your actions or questions. Commands:")
    click.echo("  /end     - End the session")
    click.echo("  /scene   - Update scene state")
    click.echo("  /status  - Show current scene")
    click.echo("  /help    - Show this help")
    click.echo("-" * 40)

    try:
        agent = GMAgent(campaign_id, verbose=verbose)
    except Exception as e:
        click.echo(f"Error initializing agent: {e}", err=True)
        sys.exit(1)

    try:
        _run_repl(agent)
    except KeyboardInterrupt:
        click.echo("\n\nSession interrupted.")
    finally:
        agent.close()


def _run_repl(agent: GMAgent):
    """Run the interactive REPL loop."""
    while True:
        try:
            player_input = input("\n> ").strip()
        except EOFError:
            break

        if not player_input:
            continue

        # Handle commands
        if player_input.startswith("/"):
            cmd = player_input.lower().split()[0]

            if cmd == "/end":
                summary = input("Session summary (optional): ").strip()
                agent.end_session(summary)
                click.echo("Session ended and saved.")
                break

            elif cmd == "/scene":
                _handle_scene_update(agent)
                continue

            elif cmd == "/status":
                scene = agent.session.scene_state
                click.echo(f"\nCurrent Scene:")
                click.echo(f"  Location: {scene.location}")
                click.echo(f"  Time: {scene.time_of_day}")
                click.echo(f"  NPCs: {', '.join(scene.npcs_present) or 'None'}")
                click.echo(f"  Conditions: {', '.join(scene.conditions) or 'None'}")
                continue

            elif cmd == "/help":
                click.echo("\nCommands:")
                click.echo("  /end     - End the session")
                click.echo("  /scene   - Update scene state")
                click.echo("  /status  - Show current scene")
                click.echo("  /help    - Show this help")
                continue

            else:
                click.echo(f"Unknown command: {cmd}")
                continue

        # Process player turn
        try:
            response = agent.process_turn(player_input)
            click.echo(f"\n[GM] {response}")
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)


def _handle_scene_update(agent: GMAgent):
    """Handle interactive scene update."""
    scene = agent.session.scene_state

    location = input(f"Location [{scene.location}]: ").strip()
    time_of_day = input(f"Time of day [{scene.time_of_day}]: ").strip()
    npcs = input(f"NPCs present (comma-separated) [{', '.join(scene.npcs_present)}]: ").strip()
    conditions = input(f"Conditions (comma-separated) [{', '.join(scene.conditions)}]: ").strip()

    agent.update_scene(
        location=location if location else scene.location,
        time_of_day=time_of_day if time_of_day else scene.time_of_day,
        npcs_present=([n.strip() for n in npcs.split(",")] if npcs else scene.npcs_present),
        conditions=([c.strip() for c in conditions.split(",")] if conditions else scene.conditions),
    )
    click.echo("Scene updated.")


@session.command("list")
@click.argument("campaign_id")
def session_list(campaign_id: str):
    """List sessions for a campaign."""
    sessions = session_store.list(campaign_id)
    current = session_store.get_current(campaign_id)

    if current:
        click.echo(f"Active: {current.id} ({len(current.turns)} turns)")

    if sessions:
        click.echo("\nArchived sessions:")
        for s in sessions:
            click.echo(
                f"  {s.id}: {s.started_at.strftime('%Y-%m-%d %H:%M')} ({len(s.turns)} turns)"
            )
    elif not current:
        click.echo("No sessions found.")


@session.command("show")
@click.argument("campaign_id")
@click.argument("session_id")
@click.option("--turns", "-t", default=10, help="Number of recent turns to show")
def session_show(campaign_id: str, session_id: str, turns: int):
    """Show session details."""
    s = session_store.get(campaign_id, session_id)
    if not s:
        # Check if it's the current session
        current = session_store.get_current(campaign_id)
        if current and current.id == session_id:
            s = current
        else:
            click.echo(f"Session '{session_id}' not found.", err=True)
            sys.exit(1)

    click.echo(f"Session: {s.id}")
    click.echo(f"  Started: {s.started_at}")
    if s.ended_at:
        click.echo(f"  Ended: {s.ended_at}")
    click.echo(f"  Turns: {len(s.turns)}")

    if s.summary:
        click.echo(f"\nSummary: {s.summary}")

    if s.turns:
        click.echo(f"\nRecent turns (last {min(turns, len(s.turns))}):")
        for turn in s.turns[-turns:]:
            click.echo(f"\n  > {turn.player_input}")
            click.echo(f"  [GM] {turn.gm_response[:200]}...")


# ============================================================================
# Utility commands
# ============================================================================


@cli.command("search")
@click.argument("query")
@click.option("--type", "-t", "doc_type", help="Filter by document type")
@click.option("--limit", "-l", default=5, help="Number of results")
def search(query: str, doc_type: str | None, limit: int):
    """Search the Pathfinder 2e database directly."""
    from gm_agent.mcp.pf2e_rag import PF2eRAGServer

    server = PF2eRAGServer()
    try:
        if doc_type:
            result = server.call_tool(
                "search_content", {"query": query, "types": doc_type, "limit": limit}
            )
        else:
            result = server.call_tool("search_content", {"query": query, "limit": limit})

        if result.success:
            click.echo(result.to_string())
        else:
            click.echo(f"Search error: {result.error}", err=True)
    finally:
        server.close()


@cli.command("chat")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "openai", "anthropic", "openrouter"]),
    default=None,
    help="LLM backend to use (default: from LLM_BACKEND env)",
)
def chat(verbose: bool, backend: str | None):
    """Start an interactive chat with the GM assistant.

    A lightweight chat mode for rules lookups and GM prep.
    Does not require or use campaign/session state.

    \b
    Commands:
      /clear  - Clear conversation history
      /tools  - List available tools
      /quit   - Exit chat
    """
    from gm_agent.chat import ChatAgent

    click.echo("GM Assistant Chat")
    click.echo("Ask questions about Pathfinder 2e rules, creatures, spells, and more.")
    click.echo("Type /quit to exit, /clear to reset, /tools to list available tools.")
    click.echo("-" * 40)

    try:
        llm = get_backend(backend) if backend else None
        agent = ChatAgent(llm=llm, verbose=verbose)
        click.echo(f"Using backend: {agent.llm.get_model_name()}")
    except Exception as e:
        click.echo(f"Error initializing chat agent: {e}", err=True)
        sys.exit(1)

    try:
        _run_chat_repl(agent)
    except KeyboardInterrupt:
        click.echo("\n\nChat ended.")
    finally:
        agent.close()


def _run_chat_repl(agent):
    """Run the chat REPL loop."""
    from gm_agent.chat import ChatAgent

    while True:
        try:
            user_input = input("\n> ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                click.echo("Goodbye!")
                break

            elif cmd == "/clear":
                agent.clear_history()
                click.echo("Conversation history cleared.")
                continue

            elif cmd == "/tools":
                tools = agent.get_tools()
                click.echo("\nAvailable tools:")
                for tool in tools:
                    click.echo(f"  {tool.name}: {tool.description[:60]}...")
                continue

            elif cmd == "/help":
                click.echo("\nCommands:")
                click.echo("  /clear  - Clear conversation history")
                click.echo("  /tools  - List available tools")
                click.echo("  /quit   - Exit chat")
                continue

            else:
                click.echo(f"Unknown command: {cmd}")
                continue

        # Process the message
        try:
            response = agent.chat(user_input)
            click.echo(f"\n{response}")
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)


@cli.command("test-connection")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "openai", "anthropic", "openrouter"]),
    default=None,
    help="Backend to test (default: from LLM_BACKEND env)",
)
def test_connection(backend: str | None):
    """Test connection to LLM backend."""
    from gm_agent.config import LLM_BACKEND

    backend_name = backend or LLM_BACKEND
    click.echo(f"Testing backend: {backend_name}")
    click.echo(f"Available backends: {', '.join(list_backends())}")

    try:
        llm = get_backend(backend_name)
        click.echo(f"Model: {llm.get_model_name()}")

        if llm.is_available():
            click.echo(f"\nBackend '{backend_name}' is available!")

            # For Ollama, also list available models
            if backend_name == "ollama":
                from gm_agent.models.ollama import OllamaBackend

                if isinstance(llm, OllamaBackend):
                    models = llm.list_models()
                    click.echo(f"Available Ollama models: {', '.join(models)}")
        else:
            click.echo(
                f"\nWarning: Backend '{backend_name}' is not available or not configured.",
                err=True,
            )
            if backend_name in ["openai", "openrouter"]:
                click.echo("Check that OPENAI_API_KEY or OPENROUTER_API_KEY is set.", err=True)
            elif backend_name == "anthropic":
                click.echo("Check that ANTHROPIC_API_KEY is set.", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\nConnection failed: {e}", err=True)
        sys.exit(1)


@cli.command("replay")
@click.argument("campaign_id")
@click.argument("session_id")
@click.option("--speed", "-s", default=0.0, type=float, help="Speed multiplier (0=instant, 1=real-time, 10=10x)")
@click.option("--backend", "-b", help="LLM backend to use (ollama, openai, anthropic)")
@click.option("--model", "-m", help="Model name to use")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def replay_session_cli(campaign_id: str, session_id: str, speed: float, backend: str | None, model: str | None, verbose: bool):
    """Replay a recorded session for debugging/testing.

    Examples:

    \b
    # Replay instantly (default)
    gm replay my-campaign session-123

    \b
    # Replay at real-time speed
    gm replay my-campaign session-123 --speed 1.0

    \b
    # Replay 10x faster
    gm replay my-campaign session-123 --speed 10.0

    \b
    # Replay with different model
    gm replay my-campaign session-123 --backend openai --model gpt-4
    """
    from gm_agent.replay import SessionReplayer

    try:
        # Get LLM backend if specified
        llm = None
        if backend:
            llm = get_backend(backend=backend, model=model)

        replayer = SessionReplayer(campaign_id)

        click.echo(f"Replaying session {session_id} at {speed}x speed...")
        if backend:
            click.echo(f"Using backend: {backend}" + (f" ({model})" if model else ""))
        click.echo()

        turn_count = 0
        for result in replayer.replay(session_id, speed_multiplier=speed, llm=llm, verbose=verbose):
            turn_count += 1

            if "error" in result:
                click.echo(f"\n[Turn {result['turn_number']} - ERROR]", err=True)
                click.echo(f"Error: {result['error']}", err=True)
            else:
                click.echo(f"\n[Turn {result['turn_number']}]")
                click.echo(f"Player: {result['player_input']}")
                click.echo(f"GM: {result['replayed_response']}")

                if verbose:
                    click.echo(f"Processing time: {result['processing_time_ms']:.1f}ms")
                    metadata = result.get("metadata", {})
                    if metadata.get("tool_count"):
                        click.echo(f"Tool calls: {metadata['tool_count']}")

        click.echo(f"\n✓ Replay complete: {turn_count} turns")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Replay failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("compare")
@click.argument("campaign_id")
@click.argument("session_id")
@click.option("--models", "-m", required=True, help="Comma-separated list of model configs (e.g., 'ollama:llama3,openai:gpt-4')")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def compare_models_cli(campaign_id: str, session_id: str, models: str, verbose: bool):
    """Compare different models by replaying the same session.

    Model format: backend:model (e.g., ollama:llama3, openai:gpt-4)

    Example:

    \b
    gm compare my-campaign session-123 --models "ollama:llama3,openai:gpt-4,anthropic:claude-3-sonnet"
    """
    from gm_agent.replay import SessionReplayer

    try:
        # Parse model configs
        model_backends = []
        for model_config in models.split(","):
            parts = model_config.strip().split(":")
            if len(parts) != 2:
                click.echo(f"Error: Invalid model config '{model_config}'. Use format 'backend:model'", err=True)
                sys.exit(1)

            backend_name, model_name = parts
            backend = get_backend(backend=backend_name, model=model_name)
            model_backends.append((model_config, backend))

        click.echo(f"Comparing {len(model_backends)} models on session {session_id}...")
        for name, _ in model_backends:
            click.echo(f"  - {name}")
        click.echo()

        replayer = SessionReplayer(campaign_id)
        results = replayer.compare_models(session_id, model_backends, verbose=verbose)

        # Display results
        click.echo("\n=== Comparison Results ===\n")

        summary = results.get("_summary", {})
        performance = summary.get("performance", {})
        errors = summary.get("errors", {})

        for model_name in [k for k in results.keys() if not k.startswith("_")]:
            model_perf = performance.get(model_name, {})
            model_errors = errors.get(model_name, 0)

            click.echo(f"{model_name}:")
            click.echo(f"  Total time: {model_perf.get('total_time_ms', 0):.1f}ms")
            click.echo(f"  Avg per turn: {model_perf.get('avg_time_per_turn_ms', 0):.1f}ms")
            click.echo(f"  Total tokens: {model_perf.get('total_tokens', 0)}")
            click.echo(f"  Turns completed: {model_perf.get('turns_completed', 0)}")
            if model_errors > 0:
                click.echo(f"  Errors: {model_errors} ⚠️")
            click.echo()

        # Find fastest
        if performance:
            fastest = min(performance.items(), key=lambda x: x[1].get("total_time_ms", float("inf")))
            click.echo(f"⚡ Fastest: {fastest[0]} ({fastest[1].get('total_time_ms', 0):.1f}ms)")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Comparison failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("server")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=5000, type=int, help="Port to bind to")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.option("--auth", is_flag=True, help="Enable JWT authentication")
def server(host: str, port: int, debug: bool, auth: bool):
    """Start the REST API server (development only).

    For production, use uWSGI or Gunicorn with wsgi.py:

    \b
    uWSGI:
        uwsgi --http :5000 --wsgi-file wsgi.py --callable app --processes 4

    \b
    Gunicorn:
        gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
    """
    import os
    from api import create_app

    if auth:
        os.environ["API_AUTH_ENABLED"] = "true"
        click.echo("JWT authentication enabled")

    click.echo(f"Starting development server at http://{host}:{port}")
    click.echo("Swagger UI available at: http://{host}:{port}/api/docs/")
    click.echo("\nPress Ctrl+C to stop\n")

    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    cli()
