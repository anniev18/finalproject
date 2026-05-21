"""Helpers for labeling model versions in saved trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def adapter_version(role: str, adapter_path: str | Path | None) -> str:
    """Return a compact version label for a base model plus optional adapter."""

    if adapter_path is None:
        return "base"

    name = Path(adapter_path).name
    if name.startswith(f"{role}_"):
        return name
    if name.startswith(("round_", "dry_round_")):
        return f"{role}_{name}"
    return f"{role}_{name}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def build_model_metadata(
    *,
    victim_model: str,
    attacker_model: str,
    judge_model: str,
    victim_adapter_path: str | Path | None = None,
    attacker_adapter_path: str | Path | None = None,
    run_id: str | None = None,
    round_index: int | None = None,
    trajectory_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build consistent metadata for an episode or trajectory-bank record."""

    metadata: dict[str, Any] = {
        "run_id": run_id or new_run_id(),
        "created_at": now_iso(),
        "victim_model": victim_model,
        "victim_version": adapter_version("victim", victim_adapter_path),
        "victim_adapter_path": str(victim_adapter_path) if victim_adapter_path else None,
        "attacker_model": attacker_model,
        "attacker_version": adapter_version("attacker", attacker_adapter_path),
        "attacker_adapter_path": str(attacker_adapter_path) if attacker_adapter_path else None,
        "judge_model": judge_model,
    }
    if round_index is not None:
        metadata["round_index"] = round_index
    if trajectory_source is not None:
        metadata["trajectory_source"] = trajectory_source
    if extra:
        metadata.update(extra)
    return metadata


def annotate_turn_metadata(
    turns: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Copy version labels onto each turn for easier filtering/debugging."""

    per_turn_keys = (
        "victim_model",
        "victim_version",
        "victim_adapter_path",
        "attacker_model",
        "attacker_version",
        "attacker_adapter_path",
        "judge_model",
        "run_id",
        "round_index",
    )
    for turn in turns:
        turn_metadata = dict(turn.get("metadata", {}))
        for key in per_turn_keys:
            if key in metadata:
                turn_metadata.setdefault(key, metadata[key])
        turn["metadata"] = turn_metadata

