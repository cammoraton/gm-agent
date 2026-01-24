"""Tests for secret tracking and revelation system."""

import pytest
from unittest.mock import patch

from gm_agent.storage.secrets import SecretStore
from gm_agent.storage.schemas import Secret
from gm_agent.mcp.campaign_state import CampaignStateServer
from gm_agent.storage.session import session_store


class TestSecretSchema:
    """Tests for Secret schema."""

    def test_secret_creation(self):
        """Test creating a Secret."""
        secret = Secret(
            id="secret-001",
            campaign_id="test",
            content="The mayor is actually a dragon in disguise",
            importance="critical",
            known_by_character_ids=["npc1", "npc2"],
            consequences=["Mayor transforms", "City panics"],
            tags=["plot", "mayor"],
        )

        assert secret.id == "secret-001"
        assert secret.content == "The mayor is actually a dragon in disguise"
        assert secret.importance == "critical"
        assert len(secret.known_by_character_ids) == 2
        assert len(secret.consequences) == 2
        assert secret.revealed_to_party is False

    def test_secret_defaults(self):
        """Test Secret default values."""
        secret = Secret(
            id="secret-001",
            campaign_id="test",
            content="Test secret"
        )

        assert secret.importance == "major"
        assert secret.known_by_character_ids == []
        assert secret.known_by_faction_ids == []
        assert secret.revealed_to_party is False
        assert secret.revelation_event is None
        assert secret.consequences == []
        assert secret.triggered_consequences == []
        assert secret.tags == []


class TestSecretStore:
    """Tests for SecretStore."""

    def test_create_secret(self, tmp_path):
        """Test creating a secret."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create(
            content="The artifact is cursed",
            importance="major",
            consequences=["Party member gets cursed", "Seek cleansing"],
        )

        assert secret.content == "The artifact is cursed"
        assert secret.importance == "major"
        assert len(secret.consequences) == 2
        assert "cursed" in secret.consequences[0].lower()

    def test_get_secret(self, tmp_path):
        """Test getting a secret by ID."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        created = store.create("Test secret")
        loaded = store.get(created.id)

        assert loaded is not None
        assert loaded.content == "Test secret"

    def test_update_secret(self, tmp_path):
        """Test updating a secret."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create("Test secret")
        secret.importance = "critical"
        store.update(secret)

        loaded = store.get(secret.id)
        assert loaded.importance == "critical"

    def test_delete_secret(self, tmp_path):
        """Test deleting a secret."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create("Test secret")
        success = store.delete(secret.id)
        assert success is True

        loaded = store.get(secret.id)
        assert loaded is None

    def test_list_all_secrets(self, tmp_path):
        """Test listing all secrets."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        store.create("Secret One")
        store.create("Secret Two")
        store.create("Secret Three")

        secrets = store.list_all()
        assert len(secrets) == 3

    def test_list_by_revelation_status(self, tmp_path):
        """Test filtering secrets by revelation status."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret1 = store.create("Hidden secret")
        secret2 = store.create("Revealed secret")

        store.reveal_to_party(secret2.id)

        hidden = store.list_all(revealed=False)
        revealed = store.list_all(revealed=True)

        assert len(hidden) == 1
        assert len(revealed) == 1
        assert hidden[0].content == "Hidden secret"
        assert revealed[0].content == "Revealed secret"

    def test_add_knower(self, tmp_path):
        """Test adding a character who knows a secret."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create("Test secret")
        success = store.add_knower(secret.id, "char1")
        assert success is True

        loaded = store.get(secret.id)
        assert "char1" in loaded.known_by_character_ids

        # Adding again should be idempotent
        store.add_knower(secret.id, "char1")
        loaded = store.get(secret.id)
        assert loaded.known_by_character_ids.count("char1") == 1

    def test_add_faction_knower(self, tmp_path):
        """Test adding a faction that knows a secret."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create("Test secret")
        success = store.add_faction_knower(secret.id, "faction1")
        assert success is True

        loaded = store.get(secret.id)
        assert "faction1" in loaded.known_by_faction_ids

    def test_reveal_to_party(self, tmp_path):
        """Test revealing a secret to the party."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create("Test secret")
        success = store.reveal_to_party(
            secret.id,
            session_id="session1",
            turn_number=5,
            revealer="NPC Bob",
            method="Interrogation",
        )
        assert success is True

        loaded = store.get(secret.id)
        assert loaded.revealed_to_party is True
        assert loaded.revelation_event is not None
        assert loaded.revelation_event["session_id"] == "session1"
        assert loaded.revelation_event["turn_number"] == 5
        assert loaded.revelation_event["revealer"] == "NPC Bob"
        assert loaded.revelation_event["method"] == "Interrogation"

    def test_trigger_consequence(self, tmp_path):
        """Test triggering a consequence."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create(
            "Test secret",
            consequences=["Consequence 1", "Consequence 2"],
        )

        success = store.trigger_consequence(secret.id, "Consequence 1")
        assert success is True

        loaded = store.get(secret.id)
        assert "Consequence 1" in loaded.triggered_consequences
        assert "Consequence 2" not in loaded.triggered_consequences

        # Triggering again should return False
        success = store.trigger_consequence(secret.id, "Consequence 1")
        assert success is False

    def test_get_untriggered_consequences(self, tmp_path):
        """Test getting untriggered consequences."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret = store.create(
            "Test secret",
            consequences=["Consequence 1", "Consequence 2", "Consequence 3"],
        )

        store.trigger_consequence(secret.id, "Consequence 1")

        untriggered = store.get_untriggered_consequences(secret.id)
        assert len(untriggered) == 2
        assert "Consequence 1" not in untriggered
        assert "Consequence 2" in untriggered
        assert "Consequence 3" in untriggered

    def test_get_character_secrets(self, tmp_path):
        """Test getting secrets known by a character."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret1 = store.create("Secret 1")
        secret2 = store.create("Secret 2")
        secret3 = store.create("Secret 3")

        store.add_knower(secret1.id, "char1")
        store.add_knower(secret2.id, "char1")
        store.add_knower(secret3.id, "char2")

        char1_secrets = store.get_character_secrets("char1")
        assert len(char1_secrets) == 2

        char2_secrets = store.get_character_secrets("char2")
        assert len(char2_secrets) == 1

    def test_get_faction_secrets(self, tmp_path):
        """Test getting secrets known by a faction."""
        store = SecretStore("test-campaign", base_dir=tmp_path)

        secret1 = store.create("Secret 1")
        secret2 = store.create("Secret 2")

        store.add_faction_knower(secret1.id, "faction1")
        store.add_faction_knower(secret2.id, "faction1")

        faction_secrets = store.get_faction_secrets("faction1")
        assert len(faction_secrets) == 2


class TestCampaignStateSecretTools:
    """Tests for secret tools in CampaignStateServer."""

    @pytest.fixture
    def campaign_server(self, tmp_path):
        """Create CampaignStateServer with test data."""
        with patch("gm_agent.mcp.campaign_state.CAMPAIGNS_DIR", tmp_path):
            server = CampaignStateServer("test-campaign")
            yield server

    def test_create_secret_tool(self, campaign_server):
        """Test creating a secret via MCP tool."""
        result = campaign_server.call_tool(
            "create_secret",
            {
                "content": "The king is dying of poison",
                "importance": "critical",
                "consequences": "Succession crisis,War breaks out",
            }
        )

        assert result.success
        assert "The king is dying" in result.data

    def test_create_secret_defaults(self, campaign_server):
        """Test creating secret with default values."""
        result = campaign_server.call_tool(
            "create_secret",
            {"content": "Simple secret"}
        )

        assert result.success

    def test_list_secrets_all(self, campaign_server):
        """Test listing all secrets."""
        campaign_server.call_tool("create_secret", {"content": "Secret 1"})
        campaign_server.call_tool("create_secret", {"content": "Secret 2"})

        result = campaign_server.call_tool("list_secrets", {"revealed": "all"})

        assert result.success
        assert "Secret 1" in result.data
        assert "Secret 2" in result.data

    def test_list_secrets_by_status(self, campaign_server):
        """Test listing secrets filtered by revelation status."""
        # Create secrets
        result1 = campaign_server.call_tool("create_secret", {"content": "Hidden secret"})
        result2 = campaign_server.call_tool("create_secret", {"content": "To be revealed"})

        # Extract secret ID from result2 (it's in the format "Created secret (ID: secret-...) ...")
        import re
        match = re.search(r'ID: (secret-[\d-]+)', result2.data)
        secret_id = match.group(1) if match else None

        assert secret_id is not None

        # Reveal one secret
        campaign_server.call_tool("reveal_secret", {"secret_id": secret_id})

        # List hidden only
        hidden_result = campaign_server.call_tool("list_secrets", {"revealed": "false"})
        assert "Hidden secret" in hidden_result.data
        assert "To be revealed" not in hidden_result.data

        # List revealed only
        revealed_result = campaign_server.call_tool("list_secrets", {"revealed": "true"})
        assert "To be revealed" in revealed_result.data
        assert "Hidden secret" not in revealed_result.data

    def test_reveal_secret_tool(self, campaign_server, tmp_path):
        """Test revealing a secret via MCP tool."""
        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            # Create a session
            session = session_store.start("test-campaign")

            # Create secret
            result = campaign_server.call_tool("create_secret", {"content": "Test secret"})

            # Extract secret ID
            import re
            match = re.search(r'ID: (secret-[\d-]+)', result.data)
            secret_id = match.group(1) if match else None

            assert secret_id is not None

            # Reveal it
            reveal_result = campaign_server.call_tool(
                "reveal_secret",
                {
                    "secret_id": secret_id,
                    "revealer": "Mysterious Stranger",
                    "method": "Deathbed confession",
                }
            )

            assert reveal_result.success
            assert "Test secret" in reveal_result.data

    def test_get_revelation_history_empty(self, campaign_server):
        """Test getting revelation history when none exist."""
        result = campaign_server.call_tool("get_revelation_history", {})

        assert result.success
        assert "No secrets have been revealed" in result.data

    def test_get_revelation_history(self, campaign_server, tmp_path):
        """Test getting revelation history."""
        with patch("gm_agent.storage.session.CAMPAIGNS_DIR", tmp_path):
            # Create session
            session = session_store.start("test-campaign")

            # Create and reveal secrets
            result1 = campaign_server.call_tool(
                "create_secret",
                {"content": "Secret 1", "consequences": "Consequence A"}
            )
            result2 = campaign_server.call_tool(
                "create_secret",
                {"content": "Secret 2"}
            )

            # Extract secret IDs
            import re
            secret_ids = []
            for result in [result1, result2]:
                match = re.search(r'ID: (secret-[\d-]+)', result.data)
                if match:
                    secret_ids.append(match.group(1))

            # Reveal them
            for secret_id in secret_ids:
                campaign_server.call_tool("reveal_secret", {"secret_id": secret_id})

            # Get history
            history_result = campaign_server.call_tool("get_revelation_history", {})

            assert history_result.success
            assert "Secret 1" in history_result.data
            assert "Secret 2" in history_result.data
            assert "Consequence A" in history_result.data


class TestSecretPersistence:
    """Tests for secret persistence."""

    def test_secret_persists_across_loads(self, tmp_path):
        """Test that secrets are saved and loaded correctly."""
        # Create and save secret
        store1 = SecretStore("test-campaign", base_dir=tmp_path)
        secret1 = store1.create(
            content="The artifact grants immortality",
            importance="critical",
            known_by_character_ids=["npc1"],
            consequences=["Power struggle begins", "Ancient evil awakens"],
            tags=["artifact", "plot"],
        )

        # Reveal it
        store1.reveal_to_party(
            secret1.id,
            revealer="Sage",
            method="Ancient scroll",
        )

        # Trigger one consequence
        store1.trigger_consequence(secret1.id, "Power struggle begins")

        # Load in new store instance
        store2 = SecretStore("test-campaign", base_dir=tmp_path)
        loaded = store2.get(secret1.id)

        assert loaded is not None
        assert loaded.content == "The artifact grants immortality"
        assert loaded.importance == "critical"
        assert "npc1" in loaded.known_by_character_ids
        assert "artifact" in loaded.tags
        assert loaded.revealed_to_party is True
        assert loaded.revelation_event is not None
        assert loaded.revelation_event["revealer"] == "Sage"
        assert "Power struggle begins" in loaded.triggered_consequences
        assert "Ancient evil awakens" not in loaded.triggered_consequences


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
