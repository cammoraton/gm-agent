"""Cross-system propagation bus for campaign state.

Lightweight mediator that links stores together so that changes in one
system (e.g. secret revealed) propagate to others (e.g. knowledge store).
All methods are idempotent — duplicate propagation is safe.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage.characters import CharacterStore
    from .storage.factions import FactionStore
    from .storage.knowledge import KnowledgeStore
    from .storage.locations import LocationStore
    from .storage.schemas import Secret

logger = logging.getLogger(__name__)

# Importance mapping for secret → knowledge propagation
SECRET_IMPORTANCE_MAP = {
    "minor": 4,
    "major": 7,
    "critical": 10,
}


class PropagationBus:
    """Direct-call mediator linking campaign stores.

    Not an event system — methods are called explicitly after store mutations.
    """

    def __init__(
        self,
        knowledge: "KnowledgeStore",
        factions: "FactionStore | None" = None,
        locations: "LocationStore | None" = None,
        characters: "CharacterStore | None" = None,
    ):
        self.knowledge = knowledge
        self.factions = factions
        self.locations = locations
        self.characters = characters

    def on_secret_revealed(self, secret: "Secret", revealer: str | None = None) -> int:
        """Propagate a revealed secret to party and relevant NPCs/factions.

        Adds knowledge to:
        - Party (__party__)
        - Characters in secret.known_by_character_ids
        - Members of factions in secret.known_by_faction_ids

        Returns count of knowledge entries created.
        """
        source = f"revealed_by_{revealer}" if revealer else "revealed"
        importance = SECRET_IMPORTANCE_MAP.get(secret.importance, 7)
        created = 0

        # Add to party knowledge
        if not self.knowledge.has_similar_knowledge("__party__", secret.content):
            self.knowledge.add_knowledge(
                character_id="__party__",
                character_name="The Party",
                content=secret.content,
                knowledge_type="secret",
                sharing_condition="free",
                source=source,
                importance=importance,
            )
            created += 1

        # Propagate to individual NPCs listed on the secret
        if self.characters:
            for char_id in secret.known_by_character_ids:
                char = self.characters.get(char_id)
                char_name = char.name if char else char_id
                if not self.knowledge.has_similar_knowledge(char_id, secret.content):
                    self.knowledge.add_knowledge(
                        character_id=char_id,
                        character_name=char_name,
                        content=secret.content,
                        knowledge_type="secret",
                        sharing_condition="free",
                        source=source,
                        importance=importance,
                    )
                    created += 1

        # Propagate to faction members
        if self.factions and self.characters:
            for faction_id in secret.known_by_faction_ids:
                member_ids = self.factions.get_members(faction_id)
                for member_id in member_ids:
                    char = self.characters.get(member_id)
                    char_name = char.name if char else member_id
                    if not self.knowledge.has_similar_knowledge(member_id, secret.content):
                        self.knowledge.add_knowledge(
                            character_id=member_id,
                            character_name=char_name,
                            content=secret.content,
                            knowledge_type="secret",
                            sharing_condition="free",
                            source=source,
                            importance=importance,
                        )
                        created += 1

        logger.info("Secret propagated: %d knowledge entries created", created)
        return created

    def on_faction_knowledge_added(self, faction_id: str, knowledge_id: int) -> int:
        """Push a knowledge entry to all members of a faction.

        Returns count of knowledge entries created.
        """
        if not self.factions or not self.characters:
            return 0

        member_ids = self.factions.get_members(faction_id)
        created = 0

        for member_id in member_ids:
            char = self.characters.get(member_id)
            char_name = char.name if char else member_id
            result = self.knowledge.copy_knowledge(
                source_id=knowledge_id,
                target_character_id=member_id,
                target_character_name=char_name,
                new_source=f"faction:{faction_id}",
            )
            if result:
                created += 1

        return created

    def on_npc_joins_faction(
        self, character_id: str, character_name: str, faction_id: str,
    ) -> int:
        """New faction member inherits all shared_knowledge entries.

        Returns count of knowledge entries created.
        """
        if not self.factions:
            return 0

        faction = self.factions.get(faction_id)
        if not faction:
            return 0

        created = 0
        for kid_str in faction.shared_knowledge:
            try:
                kid = int(kid_str)
            except (ValueError, TypeError):
                continue

            result = self.knowledge.copy_knowledge(
                source_id=kid,
                target_character_id=character_id,
                target_character_name=character_name,
                new_source=f"faction:{faction_id}",
            )
            if result:
                created += 1

        return created

    def propagate_location_knowledge(
        self, character_id: str, character_name: str, location_id: str,
    ) -> int:
        """Give an NPC a location's common_knowledge entries (pull model).

        Returns count of knowledge entries created.
        """
        if not self.locations:
            return 0

        knowledge_ids = self.locations.get_common_knowledge(location_id)
        created = 0

        for kid_str in knowledge_ids:
            try:
                kid = int(kid_str)
            except (ValueError, TypeError):
                continue

            result = self.knowledge.copy_knowledge(
                source_id=kid,
                target_character_id=character_id,
                target_character_name=character_name,
                new_source=f"location:{location_id}",
            )
            if result:
                created += 1

        return created
