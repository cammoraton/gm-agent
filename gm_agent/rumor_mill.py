"""Rumor propagation engine for information flow across the campaign world."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import CAMPAIGNS_DIR
from .storage.schemas import Rumor
from .storage.locations import LocationStore
from .storage.characters import CharacterStore

if TYPE_CHECKING:
    pass


class RumorEngine:
    """Engine for rumor propagation across campaign locations.

    Rumors spread based on:
    - Location connectivity
    - Location isolation level
    - NPC chattiness
    - Hub NPCs (accelerate spread)
    - Spread rate (slow/medium/fast)

    Accuracy degrades with each hop based on distortion rate.
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        """Initialize rumor engine.

        Args:
            campaign_id: The campaign ID
            base_dir: Base directory for campaigns (for testing)
        """
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.rumors_dir = self.base_dir / campaign_id / "rumors"
        self.rumors_dir.mkdir(parents=True, exist_ok=True)

        self.locations = LocationStore(campaign_id, base_dir=base_dir)
        self.characters = CharacterStore(campaign_id, base_dir=base_dir)

    def seed_rumor(
        self,
        content: str,
        source_type: str = "pc_seeded",
        starting_locations: list[str] | None = None,
        spread_rate: str = "medium",
        distortion_rate: float = 0.05,
        tags: list[str] | None = None,
    ) -> Rumor:
        """Seed a new rumor.

        Args:
            content: The rumor content
            source_type: "pc_seeded", "event", or "npc_created"
            starting_locations: Location IDs where rumor starts (defaults to all)
            spread_rate: "slow", "medium", or "fast"
            distortion_rate: How much accuracy decreases per hop (0.0 to 1.0)
            tags: Optional tags for categorization

        Returns:
            The created Rumor
        """
        # Generate unique ID
        rumor_id = f"rumor-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000, 9999)}"

        # Default to all locations if not specified
        if starting_locations is None:
            all_locations = self.locations.list_all()
            starting_locations = [loc.id for loc in all_locations]

        rumor = Rumor(
            id=rumor_id,
            campaign_id=self.campaign_id,
            content=content,
            source_type=source_type,
            accuracy=1.0,
            distortion_rate=distortion_rate,
            spread_rate=spread_rate,
            current_locations=starting_locations,
            tags=tags or [],
        )

        self._save(rumor)
        return rumor

    def propagate_rumors(self, time_delta_days: int = 1) -> dict[str, int]:
        """Propagate all active rumors based on time passed.

        Args:
            time_delta_days: Number of days to simulate

        Returns:
            Dict with propagation stats: {"rumors_spread": N, "new_locations": M}
        """
        rumors = self.list_all()
        stats = {"rumors_spread": 0, "new_locations": 0}

        for rumor in rumors:
            new_locations = self._propagate_single_rumor(rumor, time_delta_days)
            if new_locations:
                stats["rumors_spread"] += 1
                stats["new_locations"] += len(new_locations)

        return stats

    def _propagate_single_rumor(self, rumor: Rumor, time_delta_days: int) -> list[str]:
        """Propagate a single rumor to connected locations.

        Args:
            rumor: The rumor to propagate
            time_delta_days: Number of days to simulate

        Returns:
            List of new location IDs where rumor spread
        """
        # Spread rate determines how many hops per day
        hops_per_day = {
            "slow": 0.5,  # 1 hop every 2 days
            "medium": 1.0,  # 1 hop per day
            "fast": 2.0,  # 2 hops per day
        }

        base_hops = int(hops_per_day.get(rumor.spread_rate, 1.0) * time_delta_days)
        if base_hops == 0 and time_delta_days > 0:
            # Even slow rumors have a chance to spread
            if random.random() < 0.5:
                base_hops = 1

        if base_hops == 0:
            return []

        new_locations = set()
        current_wave = set(rumor.current_locations)

        for _ in range(base_hops):
            next_wave = set()

            for location_id in current_wave:
                location = self.locations.get(location_id)
                if not location:
                    continue

                # Check isolation level
                spread_multiplier = {
                    "connected": 1.0,
                    "remote": 0.5,
                    "isolated": 0.1,
                }
                spread_chance = spread_multiplier.get(location.isolation_level, 1.0)

                # Check for hub NPCs at this location
                hub_bonus = self._get_hub_bonus(location_id)
                spread_chance *= (1.0 + hub_bonus)

                # Try to spread to each connected location
                for connected_id in location.connected_locations:
                    if connected_id in rumor.current_locations:
                        continue  # Already has this rumor

                    if random.random() < spread_chance:
                        # Rumor spreads!
                        next_wave.add(connected_id)
                        new_locations.add(connected_id)

                        # Degrade accuracy
                        new_accuracy = max(0.0, rumor.accuracy - rumor.distortion_rate)

                        # Record in history
                        rumor.spread_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "from_location": location_id,
                            "to_location": connected_id,
                            "accuracy": new_accuracy,
                        })

                        rumor.accuracy = new_accuracy

            # Add new locations to current locations
            rumor.current_locations.extend(list(next_wave))
            current_wave = next_wave

            if not current_wave:
                break  # No more spreading

        # Save updated rumor
        if new_locations:
            self._save(rumor)

        return list(new_locations)

    def _get_hub_bonus(self, location_id: str) -> float:
        """Calculate spread bonus from hub NPCs at a location.

        Args:
            location_id: The location ID

        Returns:
            Bonus multiplier (0.0 to 1.0+)
        """
        all_characters = self.characters.list(character_type="npc")
        hub_count = 0

        # Check for hub NPCs (innkeepers, merchants, etc.)
        for char in all_characters:
            if char.is_hub:
                # Assume hub NPCs are at their location (simplified)
                # In a more complex system, we'd track NPC locations
                hub_count += 1

        # Each hub NPC adds 20% to spread chance
        return min(1.0, hub_count * 0.2)

    def get_rumors_at_location(self, location_id: str) -> list[Rumor]:
        """Get all rumors currently at a location.

        Args:
            location_id: The location ID

        Returns:
            List of Rumors at this location
        """
        all_rumors = self.list_all()
        return [r for r in all_rumors if location_id in r.current_locations]

    def get_character_rumors(self, character_id: str) -> list[Rumor]:
        """Get all rumors known by a character.

        Args:
            character_id: The character ID

        Returns:
            List of Rumors known by this character
        """
        all_rumors = self.list_all()
        return [r for r in all_rumors if character_id in r.known_by_characters]

    def mark_character_knows_rumor(self, rumor_id: str, character_id: str) -> bool:
        """Mark that a character knows a rumor.

        Args:
            rumor_id: The rumor ID
            character_id: The character ID

        Returns:
            True if successful
        """
        rumor = self.get(rumor_id)
        if not rumor:
            return False

        if character_id not in rumor.known_by_characters:
            rumor.known_by_characters.append(character_id)
            self._save(rumor)

        return True

    def get(self, rumor_id: str) -> Rumor | None:
        """Get a rumor by ID.

        Args:
            rumor_id: The rumor ID

        Returns:
            The Rumor or None if not found
        """
        rumor_file = self.rumors_dir / f"{rumor_id}.json"
        if not rumor_file.exists():
            return None

        with open(rumor_file, encoding="utf-8") as f:
            data = json.load(f)
            return Rumor(**data)

    def list_all(self) -> list[Rumor]:
        """List all rumors in the campaign.

        Returns:
            List of all Rumors
        """
        rumors = []
        for rumor_file in self.rumors_dir.glob("*.json"):
            with open(rumor_file, encoding="utf-8") as f:
                data = json.load(f)
                rumors.append(Rumor(**data))

        return sorted(rumors, key=lambda r: r.created_at, reverse=True)

    def delete(self, rumor_id: str) -> bool:
        """Delete a rumor.

        Args:
            rumor_id: The rumor ID

        Returns:
            True if deleted, False if not found
        """
        rumor_file = self.rumors_dir / f"{rumor_id}.json"
        if not rumor_file.exists():
            return False

        rumor_file.unlink()
        return True

    def _save(self, rumor: Rumor) -> None:
        """Save rumor to disk.

        Args:
            rumor: The rumor to save
        """
        rumor_file = self.rumors_dir / f"{rumor.id}.json"
        with open(rumor_file, "w", encoding="utf-8") as f:
            json.dump(rumor.model_dump(mode="json"), f, indent=2, default=str)
