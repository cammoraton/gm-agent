"""Session replay functionality for debugging and model comparison."""

import logging
import time
from datetime import datetime
from typing import Iterator

from .agent import GMAgent
from .models.base import LLMBackend
from .models.factory import get_backend
from .storage.campaign import campaign_store
from .storage.session import session_store
from .storage.schemas import Session, Turn, TurnMetadata

logger = logging.getLogger(__name__)


class ReplayResult:
    """Result of replaying a session with a specific model."""

    def __init__(self, model_name: str, session_id: str):
        self.model_name = model_name
        self.session_id = session_id
        self.turns: list[dict] = []
        self.total_time_ms: float = 0.0
        self.total_tokens: int = 0
        self.errors: list[str] = []

    def add_turn(
        self,
        turn_number: int,
        player_input: str,
        gm_response: str,
        processing_time_ms: float,
        metadata: TurnMetadata,
    ):
        """Add a turn result."""
        self.turns.append(
            {
                "turn_number": turn_number,
                "player_input": player_input,
                "gm_response": gm_response,
                "processing_time_ms": processing_time_ms,
                "metadata": metadata.model_dump(),
            }
        )
        self.total_time_ms += processing_time_ms
        if metadata.model:
            # Track token usage if available
            usage = getattr(metadata, "usage", {})
            self.total_tokens += usage.get("total_tokens", 0)

    def add_error(self, turn_number: int, error: str):
        """Add an error that occurred during replay."""
        self.errors.append(f"Turn {turn_number}: {error}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "session_id": self.session_id,
            "turns": self.turns,
            "total_time_ms": self.total_time_ms,
            "total_tokens": self.total_tokens,
            "total_turns": len(self.turns),
            "errors": self.errors,
        }


class SessionReplayer:
    """Replays recorded sessions for debugging and model comparison."""

    def __init__(self, campaign_id: str):
        """Initialize replayer for a campaign.

        Args:
            campaign_id: ID of the campaign to replay sessions from
        """
        self.campaign = campaign_store.get(campaign_id)
        if not self.campaign:
            raise ValueError(f"Campaign '{campaign_id}' not found")
        self.campaign_id = campaign_id

    def load_session(self, session_id: str) -> Session:
        """Load a session by ID.

        Args:
            session_id: ID of the session to load

        Returns:
            Session object

        Raises:
            ValueError: If session not found
        """
        session = session_store.get(self.campaign_id, session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        return session

    def replay(
        self,
        session_id: str,
        speed_multiplier: float = 1.0,
        llm: LLMBackend | None = None,
        verbose: bool = False,
    ) -> Iterator[dict]:
        """Replay a session at a specified speed.

        Yields turn-by-turn results as they are replayed. This allows
        monitoring progress and streaming results.

        Args:
            session_id: ID of the session to replay
            speed_multiplier: Speed multiplier (1.0 = real-time, 10.0 = 10x faster, 0.0 = instant)
            llm: Optional LLM backend to use (defaults to campaign's configured backend)
            verbose: Whether to print verbose output

        Yields:
            Dictionary with turn results as they complete

        Example:
            ```python
            replayer = SessionReplayer(campaign_id)
            for result in replayer.replay(session_id, speed_multiplier=10.0):
                print(f"Turn {result['turn_number']}: {result['gm_response']}")
            ```
        """
        session = self.load_session(session_id)

        if verbose:
            logger.info(
                f"Replaying session {session_id} with {len(session.turns)} turns "
                f"at {speed_multiplier}x speed"
            )

        # Create agent with specified or default backend
        backend = llm or get_backend()
        agent = GMAgent(
            campaign_id=self.campaign_id,
            llm=backend,
            verbose=verbose,
            auto_summarize=False,  # Don't summarize during replay
        )

        try:
            # Replay each turn
            for i, turn in enumerate(session.turns):
                turn_number = i + 1

                if verbose:
                    logger.info(f"Replaying turn {turn_number}/{len(session.turns)}")

                # Calculate delay based on original timing
                if speed_multiplier > 0 and i > 0:
                    # Get time since previous turn
                    prev_turn = session.turns[i - 1]
                    time_delta = (turn.timestamp - prev_turn.timestamp).total_seconds()
                    delay = time_delta / speed_multiplier
                    if delay > 0:
                        time.sleep(delay)

                # Process the turn
                start_time = time.time()
                try:
                    # Create metadata for this replay turn
                    replay_metadata = TurnMetadata(
                        source="replay",
                        event_type=turn.metadata.event_type,
                        player_id=turn.metadata.player_id,
                        actor_name=turn.metadata.actor_name,
                    )

                    response = agent.process_turn(turn.player_input, metadata=replay_metadata)
                    processing_time_ms = (time.time() - start_time) * 1000

                    # Get the recorded turn metadata
                    session_after = session_store.get_current(self.campaign_id)
                    latest_turn = session_after.turns[-1]

                    yield {
                        "turn_number": turn_number,
                        "player_input": turn.player_input,
                        "original_response": turn.gm_response,
                        "replayed_response": response,
                        "processing_time_ms": processing_time_ms,
                        "metadata": latest_turn.metadata.model_dump(),
                    }

                except Exception as e:
                    logger.error(f"Error replaying turn {turn_number}: {e}")
                    yield {
                        "turn_number": turn_number,
                        "player_input": turn.player_input,
                        "error": str(e),
                    }

        finally:
            agent.close()

    def compare_models(
        self,
        session_id: str,
        model_backends: list[tuple[str, LLMBackend]],
        verbose: bool = False,
    ) -> dict:
        """Compare different models by replaying the same session.

        Args:
            session_id: ID of the session to replay
            model_backends: List of (name, backend) tuples to compare
            verbose: Whether to print verbose output

        Returns:
            Dictionary with comparison results for each model

        Example:
            ```python
            from gm_agent.models.ollama import OllamaBackend
            from gm_agent.models.openai import OpenAIBackend

            replayer = SessionReplayer(campaign_id)
            results = replayer.compare_models(
                session_id,
                [
                    ("llama3", OllamaBackend(model="llama3")),
                    ("gpt-4", OpenAIBackend(model="gpt-4")),
                ]
            )
            ```
        """
        session = self.load_session(session_id)

        if verbose:
            logger.info(
                f"Comparing {len(model_backends)} models on session {session_id} "
                f"({len(session.turns)} turns)"
            )

        results = {}

        for model_name, backend in model_backends:
            if verbose:
                logger.info(f"Testing model: {model_name}")

            result = ReplayResult(model_name=model_name, session_id=session_id)

            # Replay with this model
            try:
                for turn_result in self.replay(
                    session_id, speed_multiplier=0.0, llm=backend, verbose=verbose
                ):
                    if "error" in turn_result:
                        result.add_error(turn_result["turn_number"], turn_result["error"])
                    else:
                        result.add_turn(
                            turn_number=turn_result["turn_number"],
                            player_input=turn_result["player_input"],
                            gm_response=turn_result["replayed_response"],
                            processing_time_ms=turn_result["processing_time_ms"],
                            metadata=TurnMetadata(**turn_result["metadata"]),
                        )

            except Exception as e:
                logger.error(f"Error comparing model {model_name}: {e}")
                result.add_error(0, str(e))

            results[model_name] = result.to_dict()

        # Add comparison summary
        results["_summary"] = self._generate_comparison_summary(results)

        return results

    def _generate_comparison_summary(self, results: dict) -> dict:
        """Generate comparison summary statistics."""
        model_names = [k for k in results.keys() if not k.startswith("_")]

        summary = {
            "models_compared": len(model_names),
            "performance": {},
            "errors": {},
        }

        for model_name in model_names:
            model_result = results[model_name]
            summary["performance"][model_name] = {
                "total_time_ms": model_result["total_time_ms"],
                "total_tokens": model_result["total_tokens"],
                "avg_time_per_turn_ms": (
                    model_result["total_time_ms"] / model_result["total_turns"]
                    if model_result["total_turns"] > 0
                    else 0
                ),
                "turns_completed": model_result["total_turns"],
            }
            summary["errors"][model_name] = len(model_result["errors"])

        return summary


def replay_session_instant(campaign_id: str, session_id: str, llm: LLMBackend | None = None) -> list[dict]:
    """Convenience function to replay a session instantly and return all results.

    Args:
        campaign_id: ID of the campaign
        session_id: ID of the session to replay
        llm: Optional LLM backend to use

    Returns:
        List of turn results
    """
    replayer = SessionReplayer(campaign_id)
    return list(replayer.replay(session_id, speed_multiplier=0.0, llm=llm))
