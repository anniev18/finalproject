"""Filtered attacker training examples for mutator-template evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from redteam_rl.attacker_training import (
    AttackerExample,
    load_full_input_episodes_from_bank,
)
from redteam_rl.mutators import (
    REQUEST_PLACEHOLDER,
    is_valid_attack_template,
    normalize_attack_template,
)
from redteam_rl.types import EpisodeState


@dataclass(frozen=True)
class FilteredAttackerExampleConfig:
    min_reward: float = 1.0
    victim_version: str | None = None
    attacker_version: str | None = None
    require_no_refusal: bool = True
    require_no_invalid_template: bool = True
    require_no_fallback: bool = True
    require_placeholder: bool = True
    require_template_changed: bool = True


@dataclass(frozen=True)
class FilteredAttackerExampleResult:
    examples: list[AttackerExample]
    stats: dict[str, int] = field(default_factory=dict)


def filtered_template_examples_from_trajectory_bank(
    trajectory_bank_path: str | Path,
    config: FilteredAttackerExampleConfig | None = None,
    limit: int | None = None,
) -> FilteredAttackerExampleResult:
    episodes = load_full_input_episodes_from_bank(trajectory_bank_path, limit=limit)
    return filtered_template_examples_from_episodes(episodes, config=config)


def filtered_template_examples_from_episodes(
    episodes: list[EpisodeState],
    config: FilteredAttackerExampleConfig | None = None,
) -> FilteredAttackerExampleResult:
    """Build SFT examples for template editing, not rendered attack messages.

    The legacy attacker trainer uses `mutator_input -> user_message`. This
    filtered path uses `mutator_input -> raw_attack_template`, which teaches the
    attacker/mutator to edit jailbreak templates while preserving `{REQUEST}`.
    """

    cfg = config or FilteredAttackerExampleConfig()
    examples: list[AttackerExample] = []
    seen: set[tuple[str, str]] = set()
    stats = {
        "turns_seen": 0,
        "skipped_low_reward": 0,
        "skipped_victim_version": 0,
        "skipped_attacker_version": 0,
        "skipped_missing_mutator_input": 0,
        "skipped_missing_raw_template": 0,
        "skipped_refusal": 0,
        "skipped_invalid_template": 0,
        "skipped_fallback": 0,
        "skipped_placeholder": 0,
        "skipped_unchanged": 0,
        "skipped_duplicate": 0,
        "examples": 0,
    }

    for episode in episodes:
        previous_template = episode.initial_template
        for turn in episode.turns:
            stats["turns_seen"] += 1
            metadata = turn.metadata or {}
            reward = float(turn.reward or 0.0)
            if reward < cfg.min_reward:
                stats["skipped_low_reward"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            turn_victim_version = _metadata_str(metadata, "victim_version")
            turn_attacker_version = _metadata_str(metadata, "attacker_version")
            if cfg.victim_version is not None and turn_victim_version != cfg.victim_version:
                stats["skipped_victim_version"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            if cfg.attacker_version is not None and turn_attacker_version != cfg.attacker_version:
                stats["skipped_attacker_version"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            mutator_input = str(metadata.get("mutator_input", "")).strip()
            if not mutator_input:
                stats["skipped_missing_mutator_input"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            raw_template = str(metadata.get("raw_attack_template", "")).strip()
            if not raw_template:
                stats["skipped_missing_raw_template"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            if cfg.require_no_refusal and bool(metadata.get("mutator_refused")):
                stats["skipped_refusal"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            if cfg.require_no_invalid_template and bool(metadata.get("mutator_invalid_template")):
                stats["skipped_invalid_template"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            if cfg.require_no_fallback and bool(metadata.get("mutator_fallback_used")):
                stats["skipped_fallback"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            if cfg.require_placeholder and raw_template.count(REQUEST_PLACEHOLDER) != 1:
                stats["skipped_placeholder"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            if not is_valid_attack_template(raw_template):
                stats["skipped_invalid_template"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            normalized_template = normalize_attack_template(raw_template)
            if (
                cfg.require_template_changed
                and _normalize_for_compare(normalized_template) == _normalize_for_compare(previous_template)
            ):
                stats["skipped_unchanged"] += 1
                previous_template = turn.attack_template or previous_template
                continue

            key = (mutator_input, normalized_template)
            if key in seen:
                stats["skipped_duplicate"] += 1
                previous_template = turn.attack_template or previous_template
                continue
            seen.add(key)
            examples.append(
                AttackerExample(
                    mutator_input=mutator_input,
                    attacker_message=normalized_template,
                    weight=max(reward, 1.0),
                    victim_version=turn_victim_version,
                    attacker_version=turn_attacker_version,
                )
            )
            stats["examples"] += 1
            previous_template = turn.attack_template or previous_template

    return FilteredAttackerExampleResult(examples=examples, stats=stats)


def _metadata_str(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _normalize_for_compare(template: str | None) -> str:
    if not template:
        return ""
    return " ".join(str(template).strip().split())
