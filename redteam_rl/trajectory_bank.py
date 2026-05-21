"""Trajectory bank utilities for accumulating attack episodes."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from redteam_rl.actions import AttackAction
from redteam_rl.types import DialogueTurn, EpisodeState


DEFAULT_TRAJECTORY_BANK = Path("outputs/trajectory_bank/episodes.jsonl")


def append_episode(
    episode: EpisodeState,
    path: str | Path = DEFAULT_TRAJECTORY_BANK,
    metadata: dict[str, Any] | None = None,
) -> None:
    record = {
        "episode": asdict(episode),
        "metadata": metadata or {},
    }
    append_record(record, path)


def append_episode_result(
    result: dict[str, Any],
    path: str | Path = DEFAULT_TRAJECTORY_BANK,
    metadata: dict[str, Any] | None = None,
) -> None:
    record_metadata = dict(result.get("metadata", {}))
    if metadata:
        record_metadata.update(metadata)
    record = {
        "episode": {
            "seed_prompt": result["seed_prompt"],
            "turns": result.get("turns", []),
        },
        "metadata": record_metadata,
    }
    append_record(record, path)


def append_record(record: dict[str, Any], path: str | Path = DEFAULT_TRAJECTORY_BANK) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_episodes(
    episodes: list[EpisodeState],
    path: str | Path = DEFAULT_TRAJECTORY_BANK,
    metadata: dict[str, Any] | None = None,
) -> None:
    for episode in episodes:
        append_episode(episode, path=path, metadata=metadata)


def load_episodes(
    path: str | Path = DEFAULT_TRAJECTORY_BANK,
    limit: int | None = None,
) -> list[EpisodeState]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if limit is not None and limit > 0:
        records = records[-limit:]
    return [episode_from_record(record) for record in records]


def load_records(
    path: str | Path = DEFAULT_TRAJECTORY_BANK,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if limit is not None and limit > 0:
        records = records[-limit:]
    return records


def episode_from_record(record: dict[str, Any]) -> EpisodeState:
    raw_episode = record.get("episode", record)
    state = EpisodeState(seed_prompt=str(raw_episode["seed_prompt"]))
    for raw_turn in raw_episode.get("turns", []):
        action = raw_turn.get("action")
        state.turns.append(
            DialogueTurn(
                user_message=str(raw_turn.get("user_message", "")),
                victim_response=str(raw_turn.get("victim_response", "")),
                action=AttackAction(action) if action else None,
                reward=raw_turn.get("reward"),
                metadata=dict(raw_turn.get("metadata", {})),
            )
        )
    return state
