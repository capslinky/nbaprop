"""Run manifest for reproducibility."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import uuid


@dataclass
class RunManifest:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: datetime = field(default_factory=datetime.utcnow)
    git_sha: Optional[str] = None
    config_hash: Optional[str] = None
    source_versions: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "source_versions": dict(self.source_versions),
            "outputs": dict(self.outputs),
        }
