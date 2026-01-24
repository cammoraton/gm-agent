"""Tests for Phase 0 features: Configurable Prompts, Dry Run Mode, Tool Usage Analytics."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from gm_agent.context import render_prompt_template
from gm_agent.game_loop import GameLoopController
from gm_agent.storage.schemas import TurnMetadata, Campaign, Session


class TestConfigurablePrompts:
    """Tests for configurable prompt templates."""

    def test_render_prompt_template_basic(self):
        """Test basic variable substitution."""
        template = "It's {actor_name}'s turn. {content}"
        variables = {"actor_name": "Voz", "content": "What do you do?"}
        result = render_prompt_template(template, variables)
        assert result == "It's Voz's turn. What do you do?"

    def test_render_prompt_template_missing_variable(self):
        """Test that missing variables don't cause errors."""
        template = "It's {actor_name}'s turn. {missing}"
        variables = {"actor_name": "Voz"}
        result = render_prompt_template(template, variables)
        assert result == "It's Voz's turn. {missing}"

    def test_render_prompt_template_multiple_occurrences(self):
        """Test that same variable can be used multiple times."""
        template = "{name} says hello. {name} waves."
        variables = {"name": "Voz"}
        result = render_prompt_template(template, variables)
        assert result == "Voz says hello. Voz waves."

    def test_render_prompt_template_empty(self):
        """Test rendering with empty template."""
        template = ""
        variables = {"actor_name": "Voz"}
        result = render_prompt_template(template, variables)
        assert result == ""

    def test_render_prompt_template_no_variables(self):
        """Test rendering template with no variables."""
        template = "Static text"
        variables = {}
        result = render_prompt_template(template, variables)
        assert result == "Static text"

    @patch("gm_agent.game_loop.logger")
    def test_game_loop_uses_custom_player_chat_prompt(self, mock_logger):
        """Test that GameLoopController uses custom player_chat_prompt from preferences."""
        # Create mock campaign with custom prompt
        mock_campaign = Mock(spec=Campaign)
        mock_campaign.preferences = {
            "player_chat_prompt": "[{actor_name}] {content}"
        }

        mock_agent = Mock()
        mock_agent.campaign = mock_campaign
        mock_agent.process_turn = Mock(return_value="Response")

        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0  # Immediate flush for testing
        )

        # Simulate player chat event
        controller.start()
        controller._handle_player_chat({
            "actorName": "Ezren",
            "content": "Hello",
            "playerId": "player123"
        })

        # Verify custom template was used
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args
        prompt = call_args[0][0]
        assert prompt == "[Ezren] Hello"

    @patch("gm_agent.game_loop.logger")
    def test_game_loop_uses_default_when_no_custom_prompt(self, mock_logger):
        """Test that GameLoopController uses default format when no custom prompt."""
        mock_campaign = Mock(spec=Campaign)
        mock_campaign.preferences = {}

        mock_agent = Mock()
        mock_agent.campaign = mock_campaign
        mock_agent.process_turn = Mock(return_value="Response")

        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0
        )

        controller.start()
        controller._handle_player_chat({
            "actorName": "Ezren",
            "content": "Hello",
            "playerId": "player123"
        })

        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args
        prompt = call_args[0][0]
        assert prompt == "Ezren: Hello"


class TestDryRunMode:
    """Tests for dry run mode."""

    def test_dry_run_constructor(self):
        """Test that dry_run flag is set in constructor."""
        mock_agent = Mock()
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            dry_run=True
        )

        assert controller.dry_run is True

    def test_dry_run_default_false(self):
        """Test that dry_run defaults to False."""
        mock_agent = Mock()
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge
        )

        assert controller.dry_run is False

    @patch("gm_agent.game_loop.logger")
    def test_dry_run_logs_instead_of_posting(self, mock_logger):
        """Test that dry run mode logs instead of posting to Foundry."""
        mock_agent = Mock()
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            dry_run=True
        )

        controller._post_response("Test response")

        # Verify bridge was NOT called
        mock_bridge.send_command.assert_not_called()

        # Verify logger was called
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "DRY RUN" in log_message
        assert "Test response" in log_message

    @patch("gm_agent.game_loop.logger")
    def test_normal_mode_posts_to_foundry(self, mock_logger):
        """Test that normal mode posts to Foundry."""
        mock_agent = Mock()
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            dry_run=False
        )

        controller._post_response("Test response")

        # Verify bridge was called
        mock_bridge.send_command.assert_called_once_with(
            "createChat",
            {
                "content": "Test response",
                "speaker": "GM Agent",
            }
        )

    def test_dry_run_in_stats(self):
        """Test that dry_run flag appears in stats."""
        mock_agent = Mock()
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            dry_run=True
        )

        stats = controller.get_stats()
        assert "dry_run" in stats
        assert stats["dry_run"] is True

    @patch("gm_agent.game_loop.logger")
    def test_dry_run_still_tracks_stats(self, mock_logger):
        """Test that dry run mode still tracks statistics."""
        mock_agent = Mock()
        mock_agent.process_turn = Mock(return_value="Response")
        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge,
            batch_window_seconds=0,
            dry_run=True
        )

        controller.start()
        controller._handle_player_chat({
            "actorName": "Ezren",
            "content": "Hello",
            "playerId": "player123"
        })

        stats = controller.get_stats()
        assert stats["player_chat_count"] == 1
        assert stats["response_count"] == 1


class TestToolUsageAnalytics:
    """Tests for tool usage analytics."""

    def test_turn_metadata_tool_usage_field(self):
        """Test that TurnMetadata has tool_usage field."""
        metadata = TurnMetadata()
        assert hasattr(metadata, "tool_usage")
        assert metadata.tool_usage == {}

    def test_turn_metadata_tool_failures_field(self):
        """Test that TurnMetadata has tool_failures field."""
        metadata = TurnMetadata()
        assert hasattr(metadata, "tool_failures")
        assert metadata.tool_failures == []

    def test_turn_metadata_with_tool_usage(self):
        """Test creating TurnMetadata with tool usage."""
        metadata = TurnMetadata(
            tool_usage={"search_rules": 2, "roll_dice": 1},
            tool_failures=["get_combat_state"]
        )
        assert metadata.tool_usage == {"search_rules": 2, "roll_dice": 1}
        assert metadata.tool_failures == ["get_combat_state"]

    def test_turn_metadata_serialization(self):
        """Test that TurnMetadata with tool analytics serializes correctly."""
        metadata = TurnMetadata(
            tool_usage={"search_rules": 2},
            tool_failures=["roll_dice"]
        )
        data = metadata.model_dump()
        assert "tool_usage" in data
        assert "tool_failures" in data
        assert data["tool_usage"] == {"search_rules": 2}
        assert data["tool_failures"] == ["roll_dice"]


class TestIntegrationPhase0:
    """Integration tests for Phase 0 features."""

    @patch("gm_agent.game_loop.logger")
    def test_custom_npc_combat_turn_prompt(self, mock_logger):
        """Test custom NPC combat turn prompt integration."""
        mock_campaign = Mock(spec=Campaign)
        mock_campaign.preferences = {
            "npc_combat_turn_prompt": "Now {actor_name} attacks!"
        }

        mock_agent = Mock()
        mock_agent.campaign = mock_campaign
        mock_agent.process_turn = Mock(return_value="The orc swings!")

        mock_bridge = Mock()

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge
        )

        controller._run_npc_turn("Orc Warrior")

        # Verify custom template was used
        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args
        prompt = call_args[0][0]
        assert prompt == "Now Orc Warrior attacks!"

    @patch("gm_agent.game_loop.logger")
    def test_custom_npc_narration_prompt(self, mock_logger):
        """Test custom NPC narration prompt integration."""
        mock_campaign = Mock(spec=Campaign)
        mock_campaign.preferences = {
            "npc_narration_prompt": "Narrate {actor_name}'s actions. {context}"
        }

        mock_agent = Mock()
        mock_agent.campaign = mock_campaign
        mock_agent.process_turn = Mock(return_value="The orc roars!")

        mock_bridge = Mock()
        mock_bridge.send_command = Mock(return_value={"active": False})

        controller = GameLoopController(
            campaign_id="test",
            agent=mock_agent,
            bridge=mock_bridge
        )

        controller._narrate_npc_turn("Orc Warrior")

        mock_agent.process_turn.assert_called_once()
        call_args = mock_agent.process_turn.call_args
        prompt = call_args[0][0]
        assert "Narrate Orc Warrior's actions." in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
