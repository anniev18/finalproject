"""Inspect saved TrailBlazer eval episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and preview eval episodes.")
    parser.add_argument("path", help="Eval summary JSON or *_episodes.jsonl file.")
    parser.add_argument("--policy", default=None, help="Optional policy label filter, e.g. epoch_19.")
    parser.add_argument("--examples", type=int, default=3, help="Number of successes/failures to preview.")
    parser.add_argument("--preview-chars", type=int, default=220, help="Characters to show for prompts/responses.")
    args = parser.parse_args()

    episodes = load_episodes(Path(args.path))
    if args.policy:
        episodes = [episode for episode in episodes if str(episode.get("policy")) == args.policy]
    if not episodes:
        raise SystemExit("No episodes found for the requested input/filter.")

    print_summary(episodes)
    print_examples(
        "Successful examples",
        [episode for episode in episodes if episode.get("success")],
        limit=args.examples,
        preview_chars=args.preview_chars,
    )
    print_examples(
        "Failed examples",
        [episode for episode in episodes if not episode.get("success")],
        limit=args.examples,
        preview_chars=args.preview_chars,
    )


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
    for episode in data.get("random_episodes", []):
        episodes.append(episode)
    for grouped_episodes in data.get("checkpoint_episodes", {}).values():
        episodes.extend(grouped_episodes)
    if not episodes:
        episodes.extend(data.get("trailblazer_episodes", []))
    return episodes


def print_summary(episodes: list[dict[str, Any]]) -> None:
    total = len(episodes)
    successes = [episode for episode in episodes if episode.get("success")]
    returns = [float(episode.get("episode_return", 0.0)) for episode in episodes]
    turns = [int(episode.get("num_turns", len(episode.get("turns", [])))) for episode in episodes]
    policy_counts = Counter(str(episode.get("policy", "unknown")) for episode in episodes)
    action_counts = Counter(
        str(turn.get("action"))
        for episode in episodes
        for turn in episode.get("turns", [])
        if turn.get("action") is not None
    )
    label_counts = Counter(
        str(turn.get("judge_label") or turn.get("metadata", {}).get("judge_label") or "unknown")
        for episode in episodes
        for turn in episode.get("turns", [])
    )

    print("Eval episode summary")
    print("=" * 96)
    print(f"episodes: {total}")
    print(f"successes: {len(successes)}")
    print(f"success_rate: {len(successes) / total:.3f}")
    print(f"mean_return: {sum(returns) / total:.3f}")
    print(f"mean_turns: {sum(turns) / total:.3f}")
    print(f"policies: {dict(policy_counts)}")
    print(f"judge_labels: {dict(label_counts)}")
    print(f"actions: {dict(action_counts)}")


def print_examples(
    title: str,
    episodes: list[dict[str, Any]],
    *,
    limit: int,
    preview_chars: int,
) -> None:
    print(f"\n{title}")
    print("=" * 96)
    if not episodes:
        print("none")
        return
    for index, episode in enumerate(episodes[:limit], start=1):
        print(f"\nExample {index}")
        print("-" * 96)
        print(f"policy: {episode.get('policy')}")
        print(f"seed: {episode.get('seed')}")
        print(f"success: {episode.get('success')}")
        print(f"return: {float(episode.get('episode_return', 0.0)):.3f}")
        print(f"seed_prompt: {preview(episode.get('seed_prompt'), preview_chars)}")
        for turn_index, turn in enumerate(episode.get("turns", []), start=1):
            print(f"  turn {turn_index}:")
            print(f"    action: {turn.get('action')}")
            print(f"    reward: {turn.get('reward')}")
            print(f"    judge: {turn.get('judge_label') or turn.get('metadata', {}).get('judge_label')}")
            print(f"    attack: {preview(turn.get('user_message'), preview_chars)}")
            print(f"    victim: {preview(turn.get('victim_response'), preview_chars)}")


def preview(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


if __name__ == "__main__":
    main()
