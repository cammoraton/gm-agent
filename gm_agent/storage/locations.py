"""Location storage and management."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import Location
from ..config import CAMPAIGNS_DIR

if TYPE_CHECKING:
    pass


class LocationStore:
    """Storage for campaign locations.

    Locations are stored as JSON files in the campaign directory.
    Each location represents a place in the campaign world with common
    knowledge, recent events, and connectivity information.
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        """Initialize location store.

        Args:
            campaign_id: The campaign ID
            base_dir: Base directory for campaigns (for testing)
        """
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.locations_dir = self.base_dir / campaign_id / "locations"
        self.locations_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        description: str = "",
        common_knowledge: list[str] | None = None,
        recent_events: list[str] | None = None,
        isolation_level: str = "connected",
        connected_locations: list[str] | None = None,
    ) -> Location:
        """Create a new location.

        Args:
            name: Location name
            description: Location description
            common_knowledge: List of knowledge IDs everyone here knows
            recent_events: List of recent events at this location
            isolation_level: "connected", "remote", or "isolated"
            connected_locations: List of location IDs connected to this one

        Returns:
            The created Location
        """
        # Generate ID from name (lowercase, replace spaces with hyphens)
        location_id = name.lower().replace(" ", "-").replace("'", "")

        location = Location(
            id=location_id,
            campaign_id=self.campaign_id,
            name=name,
            description=description,
            common_knowledge=common_knowledge or [],
            recent_events=recent_events or [],
            isolation_level=isolation_level,
            connected_locations=connected_locations or [],
        )

        self._save(location)
        return location

    def get(self, location_id: str) -> Location | None:
        """Get a location by ID.

        Args:
            location_id: The location ID

        Returns:
            The Location or None if not found
        """
        file_path = self.locations_dir / f"{location_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return Location(**data)

    def get_by_name(self, name: str) -> Location | None:
        """Get a location by name (case-insensitive).

        Args:
            name: The location name

        Returns:
            The Location or None if not found
        """
        # Try exact ID match first
        location_id = name.lower().replace(" ", "-").replace("'", "")
        location = self.get(location_id)
        if location:
            return location

        # Fall back to case-insensitive name search
        name_lower = name.lower()
        for location in self.list_all():
            if location.name.lower() == name_lower:
                return location

        return None

    def update(self, location: Location) -> Location:
        """Update a location.

        Args:
            location: The location to update

        Returns:
            The updated Location
        """
        location.updated_at = datetime.now()
        self._save(location)
        return location

    def delete(self, location_id: str) -> bool:
        """Delete a location.

        Args:
            location_id: The location ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.locations_dir / f"{location_id}.json"
        if not file_path.exists():
            return False

        file_path.unlink()
        return True

    def list_all(self) -> list[Location]:
        """List all locations in the campaign.

        Returns:
            List of all Locations
        """
        locations = []
        for file_path in self.locations_dir.glob("*.json"):
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                locations.append(Location(**data))

        return sorted(locations, key=lambda loc: loc.name)

    def add_common_knowledge(self, location_id: str, knowledge_id: str) -> bool:
        """Add a knowledge entry to location's common knowledge.

        Args:
            location_id: The location ID
            knowledge_id: The knowledge ID to add

        Returns:
            True if successful, False if location not found
        """
        location = self.get(location_id)
        if not location:
            return False

        knowledge_id_str = str(knowledge_id)
        if knowledge_id_str not in location.common_knowledge:
            location.common_knowledge.append(knowledge_id_str)
            self.update(location)

        return True

    def remove_common_knowledge(self, location_id: str, knowledge_id: str) -> bool:
        """Remove a knowledge entry from location's common knowledge.

        Args:
            location_id: The location ID
            knowledge_id: The knowledge ID to remove

        Returns:
            True if successful, False if location not found
        """
        location = self.get(location_id)
        if not location:
            return False

        knowledge_id_str = str(knowledge_id)
        if knowledge_id_str in location.common_knowledge:
            location.common_knowledge.remove(knowledge_id_str)
            self.update(location)

        return True

    def get_common_knowledge(self, location_id: str) -> list[str]:
        """Get all common knowledge IDs for a location.

        Args:
            location_id: The location ID

        Returns:
            List of knowledge IDs, empty if location not found
        """
        location = self.get(location_id)
        if not location:
            return []

        return list(location.common_knowledge)

    def add_event(self, location_id: str, event: str) -> bool:
        """Add a recent event to a location.

        Args:
            location_id: The location ID
            event: The event description

        Returns:
            True if successful, False if location not found
        """
        location = self.get(location_id)
        if not location:
            return False

        if event not in location.recent_events:
            location.recent_events.append(event)
            self.update(location)

        return True

    def clear_events(self, location_id: str) -> bool:
        """Clear recent events from a location.

        Args:
            location_id: The location ID

        Returns:
            True if successful, False if location not found
        """
        location = self.get(location_id)
        if not location:
            return False

        location.recent_events = []
        self.update(location)
        return True

    def connect_locations(self, location_id: str, other_location_id: str) -> bool:
        """Create a bidirectional connection between two locations.

        Args:
            location_id: The first location ID
            other_location_id: The second location ID

        Returns:
            True if successful, False if either location not found
        """
        location = self.get(location_id)
        other_location = self.get(other_location_id)

        if not location or not other_location:
            return False

        # Add bidirectional connections
        if other_location_id not in location.connected_locations:
            location.connected_locations.append(other_location_id)
            self.update(location)

        if location_id not in other_location.connected_locations:
            other_location.connected_locations.append(location_id)
            self.update(other_location)

        return True

    def disconnect_locations(self, location_id: str, other_location_id: str) -> bool:
        """Remove bidirectional connection between two locations.

        Args:
            location_id: The first location ID
            other_location_id: The second location ID

        Returns:
            True if successful, False if either location not found
        """
        location = self.get(location_id)
        other_location = self.get(other_location_id)

        if not location or not other_location:
            return False

        # Remove bidirectional connections
        if other_location_id in location.connected_locations:
            location.connected_locations.remove(other_location_id)
            self.update(location)

        if location_id in other_location.connected_locations:
            other_location.connected_locations.remove(location_id)
            self.update(other_location)

        return True

    def set_isolation_level(
        self, location_id: str, isolation_level: str
    ) -> bool:
        """Set the isolation level of a location.

        Args:
            location_id: The location ID
            isolation_level: "connected", "remote", or "isolated"

        Returns:
            True if successful, False if location not found
        """
        location = self.get(location_id)
        if not location:
            return False

        location.isolation_level = isolation_level
        self.update(location)
        return True

    def _save(self, location: Location) -> None:
        """Save location to disk.

        Args:
            location: The location to save
        """
        file_path = self.locations_dir / f"{location.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(location.model_dump(mode="json"), f, indent=2, default=str)
