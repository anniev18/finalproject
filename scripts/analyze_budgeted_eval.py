"""Analyze TrailBlazer budgeted-retry evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a budgeted TrailBlazer evaluation.")
    parser.add_argument("path", help="Budgeted eval summary JSON or *_episodes.jsonl file.")
    parser.add_argument("--examples", type=int, default=3, help="Number of seed details to print.")
    parser.add_argument("--preview-chars", type=int, default=180, help="Prompt/template preview length.")
    args = parser.parse_args()

    summary, seeds = load_artifact(Path(args.path))
    if not seeds:
        raise SystemExit("No completed seed results found.")

    print_overview(summary, seeds)
    print_mechanical_checks(summary, seeds)
    print_seed_details(seeds, limit=args.examples, preview_chars=args.preview_chars)


def load_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix == ".jsonl":
        seeds = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
        return {}, seeds

    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary, list(summary.get("seed_results", []))


def print_overview(summary: dict[str, Any], seeds: list[dict[str, Any]]) -> None:
    successes = [seed for seed in seeds if seed.get("success")]
    attempts = [attempt for seed in seeds for attempt in seed.get("attempts", [])]
    turns = [turn for attempt in attempts for turn in attempt.get("turns", [])]
    queries = [int(seed.get("queries_used", 0)) for seed in seeds]
    successful_queries = [int(seed.get("queries_used", 0)) for seed in successes]
    action_counts = Counter(str(turn.get("action", "unknown")) for turn in turns)
    label_counts = Counter(str(turn.get("judge_label", "unknown")) for turn in turns)
    attempt_successes = sum(int(attempt.get("success", False)) for attempt in attempts)

    print("Budgeted TrailBlazer evaluation")
    print("=" * 96)
    print(f"seeds: {len(seeds)}")
    print(f"successful seeds: {len(successes)}")
    print(f"ASR: {len(successes) / len(seeds):.3f}")
    print(f"victim queries: {sum(queries)}")
    print(f"mean queries per seed: {sum(queries) / len(seeds):.2f}")
    print(
        "queries per success: "
        + (f"{sum(successful_queries) / len(successes):.2f}" if successes else "n/a")
    )
    print(f"attempts: {len(attempts)}")
    print(f"successful attempts: {attempt_successes}")
    print(f"mean attempts per seed: {len(attempts) / len(seeds):.2f}")
    print(f"judge labels: {dict(label_counts)}")
    print(f"actions: {dict(action_counts)}")
    if summary:
        print(f"configured query budget: {summary.get('query_budget')}")
        print(f"configured max turns: {summary.get('max_turns')}")


def print_mechanical_checks(summary: dict[str, Any], seeds: list[dict[str, Any]]) -> None:
    configured_budget = summary.get("query_budget")
    configured_max_turns = summary.get("max_turns")
    retried = [seed for seed in seeds if len(seed.get("attempts", [])) > 1]
    changed_templates = [
        seed
        for seed in seeds
        if len({attempt.get("initial_template") for attempt in seed.get("attempts", [])}) > 1
    ]
    over_budget = [
        seed
        for seed in seeds
        if configured_budget is not None and int(seed.get("queries_used", 0)) > int(configured_budget)
    ]
    overlong_attempts = [
        (seed, attempt)
        for seed in seeds
        for attempt in seed.get("attempts", [])
        if configured_max_turns is not None
        and int(attempt.get("queries_used", 0)) > int(configured_max_turns)
    ]
    continued_after_success = [
        seed
        for seed in seeds
        if any(attempt.get("success") for attempt in seed.get("attempts", [])[:-1])
    ]
    mismatched_queries = [
        seed
        for seed in seeds
        if int(seed.get("queries_used", 0))
        != sum(int(attempt.get("queries_used", 0)) for attempt in seed.get("attempts", []))
    ]

    print("\nMechanical checks")
    print("=" * 96)
    print(f"seeds with retries: {len(retried)}/{len(seeds)}")
    print(f"seeds using multiple initial templates: {len(changed_templates)}/{len(seeds)}")
    print(f"query budget respected: {'YES' if not over_budget else 'NO'}")
    print(f"max turns per attempt respected: {'YES' if not overlong_attempts else 'NO'}")
    print(f"stopped immediately after success: {'YES' if not continued_after_success else 'NO'}")
    print(f"seed/attempt query totals match: {'YES' if not mismatched_queries else 'NO'}")

    if len(seeds) > 1 and not retried:
        print("warning: no seed required a retry; use harder seeds or a larger sample to verify retries.")
    elif retried and not changed_templates:
        print("warning: retries occurred, but their initial templates did not change.")
    elif retried:
        print("result: retry exploration and initial-template switching worked.")


def print_seed_details(seeds: list[dict[str, Any]], *, limit: int, preview_chars: int) -> None:
    print("\nSeed details")
    print("=" * 96)
    ordered = sorted(seeds, key=lambda seed: (not bool(seed.get("success")), seed.get("seed_index", 0)))
    for seed in ordered[:limit]:
        print(
            f"\nseed_index={seed.get('seed_index')} success={seed.get('success')} "
            f"queries={seed.get('queries_used')} attempts={seed.get('attempts_used')} "
            f"successful_attempt={seed.get('successful_attempt')}"
        )
        print(f"prompt: {preview(seed.get('seed_prompt'), preview_chars)}")
        for attempt in seed.get("attempts", []):
            actions = [str(turn.get("action")) for turn in attempt.get("turns", [])]
            labels = [str(turn.get("judge_label")) for turn in attempt.get("turns", [])]
            print(
                f"  attempt={attempt.get('attempt_index')} "
                f"template_index={attempt.get('initial_template_index')} "
                f"queries={attempt.get('queries_used')} success={attempt.get('success')} "
                f"actions={actions} labels={labels}"
            )
            print(f"    template: {preview(attempt.get('initial_template'), preview_chars)}")


def preview(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


if __name__ == "__main__":
    main()
