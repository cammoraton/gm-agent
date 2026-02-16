"""Structured JSONL logging for prep pipeline training data."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ..config import CAMPAIGNS_DIR


@dataclass
class PrepLogEntry:
    """A single prep step logged for training data export."""

    step: str  # "party_knowledge", "npc_knowledge", "world_context"
    campaign_id: str
    book: str
    entity: str | None  # NPC name, or None for batch
    input_context: str  # Raw entity/page text fed to LLM
    system_prompt: str  # Synthesis prompt used
    thinking: str | None  # LLM reasoning chain
    output: list[dict]  # Knowledge entries produced
    model: str
    duration_ms: float
    token_usage: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_json(self) -> str:
        """Serialize to JSON string for JSONL output."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "PrepLogEntry":
        """Deserialize from a JSON string."""
        d = json.loads(line)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class PrepLogger:
    """Append-only JSONL logger for prep pipeline steps.

    Each campaign stores its prep log at campaigns/{id}/prep_log.jsonl.
    """

    def __init__(self, campaign_id: str, base_dir: Path | None = None):
        self.campaign_id = campaign_id
        self.base_dir = base_dir or CAMPAIGNS_DIR
        self.log_path = self.base_dir / campaign_id / "prep_log.jsonl"

    def log(self, entry: PrepLogEntry) -> None:
        """Append a log entry."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def read(self) -> list[PrepLogEntry]:
        """Read back all entries."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(PrepLogEntry.from_json(line))
        return entries
