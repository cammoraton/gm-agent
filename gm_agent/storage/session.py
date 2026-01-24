"""Session storage and management."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..config import CAMPAIGNS_DIR
from .schemas import Session, Turn, SceneState, ToolCallRecord, TurnMetadata


class SessionStore:
    """Manages session persistence."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or CAMPAIGNS_DIR

    def _sessions_dir(self, campaign_id: str) -> Path:
        return self.base_dir / campaign_id / "sessions"

    def _session_file(self, campaign_id: str, session_id: str) -> Path:
        return self._sessions_dir(campaign_id) / f"{session_id}.json"

    def _current_session_file(self, campaign_id: str) -> Path:
        return self._sessions_dir(campaign_id) / "current.json"

    def start(self, campaign_id: str) -> Session:
        """Start a new session for a campaign."""
        sessions_dir = self._sessions_dir(campaign_id)
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # End any current session first
        current = self.get_current(campaign_id)
        if current:
            self.end(campaign_id)

        session = Session(
            id=str(uuid4())[:8],
            campaign_id=campaign_id,
        )

        self._save_current(session)
        return session

    def get_current(self, campaign_id: str) -> Session | None:
        """Get the current active session for a campaign."""
        current_file = self._current_session_file(campaign_id)
        if not current_file.exists():
            return None

        with open(current_file) as f:
            data = json.load(f)
        return Session.model_validate(data)

    def get_or_start(self, campaign_id: str) -> Session:
        """Get current session or start a new one."""
        session = self.get_current(campaign_id)
        if session:
            return session
        return self.start(campaign_id)

    def add_turn(
        self,
        campaign_id: str,
        player_input: str,
        gm_response: str,
        tool_calls: list[ToolCallRecord] | None = None,
        scene_state: SceneState | None = None,
        metadata: TurnMetadata | None = None,
    ) -> Turn:
        """Add a turn to the current session.

        Args:
            campaign_id: The campaign ID
            player_input: The player's input text
            gm_response: The GM's response text
            tool_calls: List of tool calls made during this turn
            scene_state: Updated scene state after this turn
            metadata: Optional metadata for analytics/fine-tuning
                (source, timing, player info, etc.)
        """
        session = self.get_current(campaign_id)
        if not session:
            raise ValueError(f"No active session for campaign '{campaign_id}'")

        turn = Turn(
            player_input=player_input,
            gm_response=gm_response,
            tool_calls=tool_calls or [],
            scene_state=scene_state,
            metadata=metadata or TurnMetadata(),
        )

        session.turns.append(turn)

        # Update scene state if provided
        if scene_state:
            session.scene_state = scene_state

        self._save_current(session)
        return turn

    def update_scene(self, campaign_id: str, scene_state: SceneState) -> None:
        """Update the current scene state."""
        session = self.get_current(campaign_id)
        if not session:
            raise ValueError(f"No active session for campaign '{campaign_id}'")

        session.scene_state = scene_state
        self._save_current(session)

    def end(self, campaign_id: str, summary: str = "") -> Session | None:
        """End the current session and archive it."""
        session = self.get_current(campaign_id)
        if not session:
            return None

        session.ended_at = datetime.now()
        session.summary = summary

        # Archive the session
        archive_file = self._session_file(campaign_id, session.id)
        with open(archive_file, "w") as f:
            json.dump(session.model_dump(mode="json"), f, indent=2, default=str)

        # Remove current session file
        self._current_session_file(campaign_id).unlink()

        return session

    def list(self, campaign_id: str) -> list[Session]:
        """List all archived sessions for a campaign."""
        sessions_dir = self._sessions_dir(campaign_id)
        if not sessions_dir.exists():
            return []

        sessions = []
        for session_file in sessions_dir.glob("*.json"):
            if session_file.name == "current.json":
                continue
            with open(session_file) as f:
                data = json.load(f)
            sessions.append(Session.model_validate(data))

        return sorted(sessions, key=lambda s: s.started_at)

    def get(self, campaign_id: str, session_id: str) -> Session | None:
        """Get a specific session by ID."""
        session_file = self._session_file(campaign_id, session_id)
        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)
        return Session.model_validate(data)

    def _save_current(self, session: Session) -> None:
        """Save the current session."""
        with open(self._current_session_file(session.campaign_id), "w") as f:
            json.dump(session.model_dump(mode="json"), f, indent=2, default=str)


# Global instance
session_store = SessionStore()
