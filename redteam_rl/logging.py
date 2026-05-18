"""Experiment logging helpers for multi-turn red-team episodes."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from redteam_rl.types import EpisodeState


class JsonlEpisodeLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(self, episode: EpisodeState, metadata: dict[str, Any] | None = None) -> None:
        record = {
            "episode": asdict(episode),
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

