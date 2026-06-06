"""Select the best TrailBlazer checkpoint from an eval summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a ranked checkpoint table and select the best checkpoint."
    )
    parser.add_argument("eval_summary", help="Path to modal_eval_trailblazer summary JSON.")
    parser.add_argument(
        "--epochs",
        default=None,
        help="Optional comma-separated epoch filter, e.g. 0,9,19,29.",
    )
    args = parser.parse_args()

    summary_path = Path(args.eval_summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = _checkpoint_rows(summary)
    if args.epochs:
        allowed_labels = {f"epoch_{epoch.strip()}" for epoch in args.epochs.split(",") if epoch.strip()}
        rows = [row for row in rows if row["label"] in allowed_labels]

    if not rows:
        raise SystemExit("No checkpoint rows found for the requested filter.")

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("success_rate", 0.0)),
            float(row.get("mean_return", 0.0)),
            _epoch_number(str(row.get("label", ""))),
        ),
        reverse=True,
    )
    best = ranked[0]

    print("Checkpoint ranking")
    print("=" * 96)
    print(
        f"{'rank':<5} {'label':<12} {'success':<10} {'mean_return':<12} "
        f"{'mean_turns':<11} {'q/success':<10} checkpoint"
    )
    for index, row in enumerate(ranked, start=1):
        print(
            f"{index:<5} "
            f"{row.get('label', ''):<12} "
            f"{float(row.get('success_rate', 0.0)):<10.3f} "
            f"{float(row.get('mean_return', 0.0)):<12.3f} "
            f"{float(row.get('mean_turns', 0.0)):<11.3f} "
            f"{_format_optional(row.get('queries_per_success')):<10} "
            f"{row.get('checkpoint')}"
        )

    tied_success = [
        row for row in ranked
        if float(row.get("success_rate", 0.0)) == float(best.get("success_rate", 0.0))
    ]
    tied_return = [
        row for row in tied_success
        if float(row.get("mean_return", 0.0)) == float(best.get("mean_return", 0.0))
    ]

    print("\nSelected checkpoint")
    print("=" * 96)
    print(f"label: {best.get('label')}")
    print(f"checkpoint: {best.get('checkpoint')}")
    print(f"success_rate: {float(best.get('success_rate', 0.0)):.3f}")
    print(f"mean_return: {float(best.get('mean_return', 0.0)):.3f}")
    print(f"mean_turns: {float(best.get('mean_turns', 0.0)):.3f}")
    print(f"queries_per_success: {_format_optional(best.get('queries_per_success'))}")
    if best.get("first_success_turn_histogram"):
        print(f"first_success_turn_histogram: {best.get('first_success_turn_histogram')}")
    print("reason:", _selection_reason(best, tied_success, tied_return))


def _checkpoint_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("checkpoint_ranking") or summary.get("checkpoint_results") or []
    return [row for row in rows if isinstance(row, dict)]


def _epoch_number(label: str) -> int:
    if label.startswith("epoch_"):
        try:
            return int(label.removeprefix("epoch_"))
        except ValueError:
            return -1
    return -1


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _selection_reason(
    best: dict[str, Any],
    tied_success: list[dict[str, Any]],
    tied_return: list[dict[str, Any]],
) -> str:
    if len(tied_success) == 1:
        return "highest success rate."
    if len(tied_return) == 1:
        return "tied on success rate, then won by higher mean return."
    labels = ", ".join(str(row.get("label")) for row in tied_return)
    return f"tied on success rate and mean return with {labels}; selected latest epoch."


if __name__ == "__main__":
    main()
