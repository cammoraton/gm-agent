"""CLI command tests."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

import sys
from pathlib import Path

# Ensure cli is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import cli
from gm_agent.storage.campaign import CampaignStore
from gm_agent.storage.session import SessionStore
from gm_agent.storage.schemas import PartyMember

from tests.conftest import MockLLMBackend, MockMCPServer


@pytest.fixture
def cli_runner():
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def cli_with_stores(
    tmp_campaigns_dir: Path,
    campaign_store: CampaignStore,
    session_store: SessionStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Set up CLI with temporary storage."""
    monkeypatch.setattr("cli.campaign_store", campaign_store)
    monkeypatch.setattr("cli.session_store", session_store)
    return campaign_store, session_store


class TestCampaignCreate:
    """Tests for 'gm campaign create' command."""

    def test_create_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign create should create a new campaign."""
        campaign_store, _ = cli_with_stores

        result = cli_runner.invoke(cli, ["campaign", "create", "Test Campaign"])

        assert result.exit_code == 0
        assert "Created campaign: test-campaign" in result.output

        # Verify campaign exists
        campaign = campaign_store.get("test-campaign")
        assert campaign is not None
        assert campaign.name == "Test Campaign"

    def test_create_campaign_with_background(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign create should accept background option."""
        campaign_store, _ = cli_with_stores

        result = cli_runner.invoke(
            cli,
            [
                "campaign",
                "create",
                "Background Test",
                "--background",
                "A dark fantasy setting",
            ],
        )

        assert result.exit_code == 0

        campaign = campaign_store.get("background-test")
        assert campaign.background == "A dark fantasy setting"

    def test_create_duplicate_fails(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign create should fail for duplicate names."""
        campaign_store, _ = cli_with_stores

        cli_runner.invoke(cli, ["campaign", "create", "Duplicate Test"])
        result = cli_runner.invoke(cli, ["campaign", "create", "Duplicate Test"])

        assert result.exit_code == 1
        assert "already exists" in result.output


class TestCampaignList:
    """Tests for 'gm campaign list' command."""

    def test_list_empty(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign list should show message when empty."""
        result = cli_runner.invoke(cli, ["campaign", "list"])

        assert result.exit_code == 0
        assert "No campaigns found" in result.output

    def test_list_campaigns(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign list should show all campaigns."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(name="Alpha Campaign")
        campaign_store.create(name="Beta Campaign")

        result = cli_runner.invoke(cli, ["campaign", "list"])

        assert result.exit_code == 0
        assert "alpha-campaign" in result.output
        assert "beta-campaign" in result.output
        assert "Alpha Campaign" in result.output
        assert "Beta Campaign" in result.output

    def test_list_shows_session_count(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign list should show session counts."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Sessions Test")
        session_store.start(campaign.id)
        session_store.end(campaign.id)
        session_store.start(campaign.id)
        session_store.end(campaign.id)

        result = cli_runner.invoke(cli, ["campaign", "list"])

        assert result.exit_code == 0
        assert "2 sessions" in result.output

    def test_list_shows_active_session(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign list should indicate active sessions."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Active Session Test")
        session_store.start(campaign.id)

        result = cli_runner.invoke(cli, ["campaign", "list"])

        assert result.exit_code == 0
        assert "active session" in result.output


class TestCampaignShow:
    """Tests for 'gm campaign show' command."""

    def test_show_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign show should display campaign details."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(
            name="Show Test",
            background="A great adventure",
            current_arc="Finding the treasure",
        )

        result = cli_runner.invoke(cli, ["campaign", "show", "show-test"])

        assert result.exit_code == 0
        assert "Show Test" in result.output
        assert "show-test" in result.output
        assert "A great adventure" in result.output
        assert "Finding the treasure" in result.output

    def test_show_missing_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign show should error for missing campaign."""
        result = cli_runner.invoke(cli, ["campaign", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_show_campaign_with_party(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
        sample_party: list[PartyMember],
    ):
        """campaign show should display party members."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(name="Party Test", party=sample_party)

        result = cli_runner.invoke(cli, ["campaign", "show", "party-test"])

        assert result.exit_code == 0
        assert "Party:" in result.output
        assert "Valeros" in result.output
        assert "Fighter" in result.output


class TestCampaignUpdate:
    """Tests for 'gm campaign update' command."""

    def test_update_background(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign update should update background."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(name="Update Test", background="Original")

        result = cli_runner.invoke(
            cli,
            ["campaign", "update", "update-test", "--background", "Updated background"],
        )

        assert result.exit_code == 0
        assert "Updated campaign" in result.output

        campaign = campaign_store.get("update-test")
        assert campaign.background == "Updated background"

    def test_update_arc(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign update should update story arc."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(name="Arc Test")

        result = cli_runner.invoke(
            cli,
            ["campaign", "update", "arc-test", "--arc", "New adventure arc"],
        )

        assert result.exit_code == 0

        campaign = campaign_store.get("arc-test")
        assert campaign.current_arc == "New adventure arc"

    def test_update_missing_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign update should error for missing campaign."""
        result = cli_runner.invoke(
            cli,
            ["campaign", "update", "nonexistent", "--background", "Test"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


class TestCampaignDelete:
    """Tests for 'gm campaign delete' command."""

    def test_delete_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign delete should remove campaign."""
        campaign_store, _ = cli_with_stores

        campaign_store.create(name="Delete Test")

        result = cli_runner.invoke(
            cli,
            ["campaign", "delete", "delete-test"],
            input="y\n",  # Confirm deletion
        )

        assert result.exit_code == 0
        assert "Deleted campaign" in result.output
        assert campaign_store.get("delete-test") is None

    def test_delete_missing_campaign(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """campaign delete should error for missing campaign."""
        result = cli_runner.invoke(
            cli,
            ["campaign", "delete", "nonexistent"],
            input="y\n",
        )

        assert result.exit_code == 1
        assert "not found" in result.output


class TestSessionList:
    """Tests for 'gm session list' command."""

    def test_list_no_sessions(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session list should show message when empty."""
        campaign_store, _ = cli_with_stores
        campaign_store.create(name="No Sessions")

        result = cli_runner.invoke(cli, ["session", "list", "no-sessions"])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_list_active_session(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session list should show active session."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Active List Test")
        session = session_store.start(campaign.id)

        result = cli_runner.invoke(cli, ["session", "list", "active-list-test"])

        assert result.exit_code == 0
        assert "Active:" in result.output
        assert session.id in result.output

    def test_list_archived_sessions(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session list should show archived sessions."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Archive List Test")
        session_store.start(campaign.id)
        session_store.end(campaign.id, "First session")
        session_store.start(campaign.id)
        session_store.end(campaign.id, "Second session")

        result = cli_runner.invoke(cli, ["session", "list", "archive-list-test"])

        assert result.exit_code == 0
        assert "Archived sessions:" in result.output


class TestSessionShow:
    """Tests for 'gm session show' command."""

    def test_show_archived_session(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session show should display archived session."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Show Session Test")
        session = session_store.start(campaign.id)
        session_store.add_turn(
            campaign_id=campaign.id,
            player_input="Test input",
            gm_response="Test response",
        )
        session_store.end(campaign.id, "Great session!")

        result = cli_runner.invoke(cli, ["session", "show", "show-session-test", session.id])

        assert result.exit_code == 0
        assert session.id in result.output
        assert "Great session!" in result.output

    def test_show_current_session(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session show should work with current session ID."""
        campaign_store, session_store = cli_with_stores

        campaign = campaign_store.create(name="Current Show Test")
        session = session_store.start(campaign.id)
        session_store.add_turn(
            campaign_id=campaign.id,
            player_input="Test input",
            gm_response="Test response",
        )

        result = cli_runner.invoke(cli, ["session", "show", "current-show-test", session.id])

        assert result.exit_code == 0
        assert session.id in result.output

    def test_show_missing_session(
        self,
        cli_runner: CliRunner,
        cli_with_stores: tuple[CampaignStore, SessionStore],
    ):
        """session show should error for missing session."""
        campaign_store, _ = cli_with_stores
        campaign_store.create(name="Missing Session Test")

        result = cli_runner.invoke(cli, ["session", "show", "missing-session-test", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestSearchCommand:
    """Tests for 'gm search' command."""

    def test_search_basic(
        self,
        cli_runner: CliRunner,
        mock_pathfinder_search,
    ):
        """search command should call RAG server."""
        with patch("gm_agent.mcp.pf2e_rag.PF2eRAGServer") as mock_rag_class:
            from gm_agent.mcp.base import ToolResult

            mock_server = MagicMock()
            mock_server.call_tool.return_value = ToolResult(
                success=True,
                data="**Goblin** (creature)\nA small green humanoid",
            )
            mock_rag_class.return_value = mock_server

            result = cli_runner.invoke(cli, ["search", "goblin"])

            assert result.exit_code == 0
            mock_server.call_tool.assert_called_once()
            mock_server.close.assert_called_once()

    def test_search_with_type(
        self,
        cli_runner: CliRunner,
    ):
        """search command should pass type filter."""
        with patch("gm_agent.mcp.pf2e_rag.PF2eRAGServer") as mock_rag_class:
            from gm_agent.mcp.base import ToolResult

            mock_server = MagicMock()
            mock_server.call_tool.return_value = ToolResult(
                success=True,
                data="**Fireball** (spell)\nA burst of flame",
            )
            mock_rag_class.return_value = mock_server

            result = cli_runner.invoke(cli, ["search", "fireball", "--type", "spell"])

            assert result.exit_code == 0
            call_args = mock_server.call_tool.call_args
            assert call_args[0][1]["types"] == "spell"

    def test_search_with_limit(
        self,
        cli_runner: CliRunner,
    ):
        """search command should pass limit option."""
        with patch("gm_agent.mcp.pf2e_rag.PF2eRAGServer") as mock_rag_class:
            from gm_agent.mcp.base import ToolResult

            mock_server = MagicMock()
            mock_server.call_tool.return_value = ToolResult(success=True, data="Results")
            mock_rag_class.return_value = mock_server

            result = cli_runner.invoke(cli, ["search", "test", "--limit", "3"])

            assert result.exit_code == 0
            call_args = mock_server.call_tool.call_args
            assert call_args[0][1]["limit"] == 3

    def test_search_error(
        self,
        cli_runner: CliRunner,
    ):
        """search command should display errors."""
        with patch("gm_agent.mcp.pf2e_rag.PF2eRAGServer") as mock_rag_class:
            from gm_agent.mcp.base import ToolResult

            mock_server = MagicMock()
            mock_server.call_tool.return_value = ToolResult(
                success=False,
                error="Database connection failed",
            )
            mock_rag_class.return_value = mock_server

            result = cli_runner.invoke(cli, ["search", "test"])

            assert "Database connection failed" in result.output


class TestTestConnection:
    """Tests for 'gm test-connection' command."""

    def test_connection_success(
        self,
        cli_runner: CliRunner,
    ):
        """test-connection should show available models."""
        with patch("cli.get_backend") as mock_get_backend:
            from gm_agent.models.ollama import OllamaBackend

            mock_backend = MagicMock(spec=OllamaBackend)
            mock_backend.get_model_name.return_value = "gpt-oss:20b"
            mock_backend.is_available.return_value = True
            mock_backend.list_models.return_value = ["gpt-oss:20b", "llama3:8b"]
            mock_get_backend.return_value = mock_backend

            result = cli_runner.invoke(cli, ["test-connection"])

            assert result.exit_code == 0
            assert "available" in result.output.lower() or "Available" in result.output

    def test_connection_not_available(
        self,
        cli_runner: CliRunner,
    ):
        """test-connection should warn if backend not available."""
        with patch("cli.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.get_model_name.return_value = "gpt-oss:20b"
            mock_backend.is_available.return_value = False
            mock_get_backend.return_value = mock_backend

            result = cli_runner.invoke(cli, ["test-connection"])

            assert result.exit_code == 1
            assert "not available" in result.output or "Warning" in result.output

    def test_connection_failure(
        self,
        cli_runner: CliRunner,
    ):
        """test-connection should show error on connection failure."""
        with patch("cli.get_backend") as mock_get_backend:
            mock_get_backend.side_effect = Exception("Connection refused")

            result = cli_runner.invoke(cli, ["test-connection"])

            assert result.exit_code == 1
            assert "Connection" in result.output or "failed" in result.output


class TestVersionOption:
    """Tests for version option."""

    def test_version(self, cli_runner: CliRunner):
        """--version should show version."""
        result = cli_runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestHelpMessages:
    """Tests for help messages."""

    def test_main_help(self, cli_runner: CliRunner):
        """Main help should show available commands."""
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "campaign" in result.output
        assert "session" in result.output
        assert "search" in result.output

    def test_campaign_help(self, cli_runner: CliRunner):
        """Campaign help should show subcommands."""
        result = cli_runner.invoke(cli, ["campaign", "--help"])

        assert result.exit_code == 0
        assert "create" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "update" in result.output
        assert "delete" in result.output

    def test_session_help(self, cli_runner: CliRunner):
        """Session help should show subcommands."""
        result = cli_runner.invoke(cli, ["session", "--help"])

        assert result.exit_code == 0
        assert "start" in result.output
        assert "list" in result.output
        assert "show" in result.output

    def test_chat_help(self, cli_runner: CliRunner):
        """Chat help should show usage."""
        result = cli_runner.invoke(cli, ["chat", "--help"])

        assert result.exit_code == 0
        assert "interactive chat" in result.output.lower()
        assert "/quit" in result.output
        assert "/clear" in result.output

    def test_server_help(self, cli_runner: CliRunner):
        """Server help should show usage."""
        result = cli_runner.invoke(cli, ["server", "--help"])

        assert result.exit_code == 0
        assert "REST API" in result.output
        assert "uWSGI" in result.output
        assert "Gunicorn" in result.output


class TestChatCommand:
    """Tests for the 'gm chat' command."""

    def test_chat_quit_command(self, cli_runner: CliRunner):
        """Chat should exit on /quit."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/quit\n")

        assert result.exit_code == 0
        assert "Goodbye" in result.output
        mock_agent.close.assert_called_once()

    def test_chat_exit_command(self, cli_runner: CliRunner):
        """Chat should exit on /exit."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/exit\n")

        assert result.exit_code == 0
        assert "Goodbye" in result.output

    def test_chat_clear_command(self, cli_runner: CliRunner):
        """Chat should clear history on /clear."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/clear\n/quit\n")

        assert "history cleared" in result.output.lower()
        mock_agent.clear_history.assert_called_once()

    def test_chat_tools_command(self, cli_runner: CliRunner):
        """Chat should list tools on /tools."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_tool = MagicMock()
            mock_tool.name = "evaluate_encounter"
            mock_tool.description = "Evaluate an encounter's difficulty"
            mock_agent.get_tools.return_value = [mock_tool]
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/tools\n/quit\n")

        assert "Available tools" in result.output
        assert "evaluate_encounter" in result.output

    def test_chat_help_command(self, cli_runner: CliRunner):
        """Chat should show help on /help."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/help\n/quit\n")

        assert "/clear" in result.output
        assert "/tools" in result.output
        assert "/quit" in result.output

    def test_chat_unknown_command(self, cli_runner: CliRunner):
        """Chat should report unknown commands."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="/unknown\n/quit\n")

        assert "Unknown command" in result.output

    def test_chat_process_message(self, cli_runner: CliRunner):
        """Chat should process regular messages."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "Goblins are small creatures level -1."
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="What is a goblin?\n/quit\n")

        assert "Goblins are small creatures" in result.output
        mock_agent.chat.assert_called_once_with("What is a goblin?")

    def test_chat_verbose_flag(self, cli_runner: CliRunner):
        """Chat should pass verbose flag to agent."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.llm = MagicMock()
            mock_agent.llm.get_model_name.return_value = "mock-model"
            mock_agent_class.return_value = mock_agent

            cli_runner.invoke(cli, ["chat", "-v"], input="/quit\n")

        # With the new --backend option, llm may be None if no backend specified
        mock_agent_class.assert_called_once_with(llm=None, verbose=True)

    def test_chat_error_handling(self, cli_runner: CliRunner):
        """Chat should handle errors gracefully."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.chat.side_effect = Exception("LLM connection failed")
            mock_agent_class.return_value = mock_agent

            result = cli_runner.invoke(cli, ["chat"], input="Hello\n/quit\n")

        assert "Error:" in result.output
        assert "LLM connection failed" in result.output

    def test_chat_empty_input_ignored(self, cli_runner: CliRunner):
        """Chat should ignore empty input."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            cli_runner.invoke(cli, ["chat"], input="\n\n/quit\n")

        mock_agent.chat.assert_not_called()

    def test_chat_initialization_error(self, cli_runner: CliRunner):
        """Chat should handle initialization errors."""
        with patch("gm_agent.chat.ChatAgent") as mock_agent_class:
            mock_agent_class.side_effect = Exception("Failed to initialize")

            result = cli_runner.invoke(cli, ["chat"])

        assert result.exit_code == 1
        assert "Error initializing" in result.output


class TestServerCommand:
    """Tests for the 'gm server' command."""

    def test_server_default_options(self, cli_runner: CliRunner):
        """Server should use default host and port."""
        with patch("api.create_app") as mock_create_app:
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app

            cli_runner.invoke(cli, ["server"])

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False)

    def test_server_custom_host_port(self, cli_runner: CliRunner):
        """Server should accept custom host and port."""
        with patch("api.create_app") as mock_create_app:
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app

            cli_runner.invoke(cli, ["server", "-h", "0.0.0.0", "-p", "8080"])

        mock_app.run.assert_called_once_with(host="0.0.0.0", port=8080, debug=False)

    def test_server_debug_mode(self, cli_runner: CliRunner):
        """Server should enable debug mode."""
        with patch("api.create_app") as mock_create_app:
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app

            cli_runner.invoke(cli, ["server", "--debug"])

        mock_app.run.assert_called_once_with(host="127.0.0.1", port=5000, debug=True)

    def test_server_auth_flag(self, cli_runner: CliRunner):
        """Server should set auth environment variable."""
        import os

        with patch("api.create_app") as mock_create_app:
            mock_app = MagicMock()
            mock_create_app.return_value = mock_app

            cli_runner.invoke(cli, ["server", "--auth"])

        assert os.environ.get("API_AUTH_ENABLED") == "true"
        # Clean up
        os.environ.pop("API_AUTH_ENABLED", None)
