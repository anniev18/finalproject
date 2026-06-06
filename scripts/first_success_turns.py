"""Compute first successful turn statistics from eval episode logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize queries until first UNSAFE turn.")
    parser.add_argument("path", help="Eval summary JSON or *_episodes.jsonl file.")
    parser.add_argument("--policy", default=None, help="Optional policy label filter, e.g. epoch_59.")
    args = parser.parse_args()

    episodes = load_episodes(Path(args.path))
    if args.policy:
        episodes = [episode for episode in episodes if str(episode.get("policy")) == args.policy]
    if not episodes:
        raise SystemExit("No episodes found for the requested input/filter.")

    rows = []
    for episode in episodes:
        first_turn = first_success_turn(episode)
        rows.append(
            {
                "policy": episode.get("policy"),
                "success": first_turn is not None,
                "first_success_turn": first_turn,
                "num_turns": int(episode.get("num_turns", len(episode.get("turns", [])))),
                "episode_return": float(episode.get("episode_return", 0.0)),
            }
        )

    successes = [row for row in rows if row["success"]]
    histogram = Counter(row["first_success_turn"] for row in successes)
    total = len(rows)

    print("First-success-turn summary")
    print("=" * 96)
    print(f"episodes: {total}")
    print(f"successes: {len(successes)}")
    print(f"success_rate: {len(successes) / total:.3f}")
    if successes:
        mean_first = sum(int(row["first_success_turn"]) for row in successes) / len(successes)
        print(f"mean_queries_until_first_success: {mean_first:.3f}")
        print(f"median_queries_until_first_success: {median([int(row['first_success_turn']) for row in successes]):.3f}")
    else:
        print("mean_queries_until_first_success: n/a")
        print("median_queries_until_first_success: n/a")
    print(f"first_success_turn_histogram: {dict(sorted(histogram.items()))}")
    print(f"mean_episode_return: {sum(row['episode_return'] for row in rows) / total:.3f}")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        episodes = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    episodes.append(json.loads(stripped))
        return episodes

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = []
    episodes.extend(data.get("random_episodes", []))
    for grouped_episodes in data.get("checkpoint_episodes", {}).values():
        episodes.extend(grouped_episodes)
    episodes.extend(data.get("trailblazer_episodes", []))
    return episodes


def first_success_turn(episode: dict[str, Any]) -> int | None:
    for index, turn in enumerate(episode.get("turns", []), start=1):
        label = str(turn.get("judge_label") or turn.get("metadata", {}).get("judge_label") or "").upper()
        reward = float(turn.get("reward") or 0.0)
        if label == "UNSAFE" or reward >= 1.0:
            return index
    return None


def median(values: list[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


if __name__ == "__main__":
    main()
