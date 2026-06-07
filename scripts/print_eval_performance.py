"""Print compact performance metrics from a TrailBlazer eval summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Print eval performance metrics.")
    parser.add_argument("summary_json", help="Path to modal_eval_trailblazer summary JSON.")
    parser.add_argument("--title", default=None, help="Optional title for the policy row.")
    args = parser.parse_args()

    with Path(args.summary_json).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    title = args.title or _default_title(summary)
    print(title)
    print("=" * 72)
    _print_row("trailblazer", summary.get("trailblazer", {}))
    if summary.get("random"):
        _print_row("random", summary.get("random", {}))

    rows = summary.get("checkpoint_results", [])
    if rows:
        print("\nCheckpoint details")
        print("=" * 72)
        for row in rows:
            label = str(row.get("label", "checkpoint"))
            _print_row(label, row)


def _print_row(label: str, metrics: dict[str, Any]) -> None:
    print(f"{label}:")
    print(f"  success_rate: {float(metrics.get('success_rate', 0.0)):.3f}")
    print(f"  mean_return: {float(metrics.get('mean_return', 0.0)):.3f}")
    print(f"  mean_turns: {float(metrics.get('mean_turns', 0.0)):.3f}")
    qps = metrics.get("queries_per_success")
    print(f"  queries_per_success: {qps if qps is not None else 'n/a'}")
    first = metrics.get("mean_first_success_turn")
    if first is not None:
        print(f"  mean_first_success_turn: {float(first):.3f}")
    median = metrics.get("median_first_success_turn")
    if median is not None:
        print(f"  median_first_success_turn: {float(median):.3f}")
    histogram = metrics.get("first_success_turn_histogram")
    if histogram:
        print(f"  first_success_turn_histogram: {histogram}")


def _default_title(summary: dict[str, Any]) -> str:
    config = summary.get("config", {})
    checkpoint = config.get("trailblazer_checkpoint")
    victim = config.get("victim_lora_adapter") or "base victim"
    reward = config.get("reward_backend")
    pieces = ["Eval performance"]
    if checkpoint:
        pieces.append(str(checkpoint).split("/")[-1])
    if victim:
        pieces.append(f"victim={str(victim).split('/')[-1]}")
    if reward:
        pieces.append(f"reward={reward}")
    return " | ".join(pieces)


if __name__ == "__main__":
    main()
