"""Merge worker outputs from modal_eval_sweep_parallel.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel sweep worker outputs into one summary JSON.")
    parser.add_argument("sweep_dir", help="Local sweep directory containing worker_*.json or worker_*_partial.json files.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary path. Defaults to <sweep_dir>/sweep_summary_from_partials.json.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    worker_paths = _worker_paths(sweep_dir)
    if not worker_paths:
        raise SystemExit(f"No worker files found in {sweep_dir}")

    completed = []
    worker_partials = []
    for path in worker_paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = [row for row in payload.get("completed", []) if isinstance(row, dict)]
        completed.extend(rows)
        worker_partials.append(
            {
                "path": str(path),
                "worker_index": payload.get("worker_index"),
                "assigned_checkpoints": payload.get("assigned_checkpoints", []),
                "completed": len(rows),
            }
        )

    ranking = sorted(
        completed,
        key=lambda row: (
            float(row.get("success_rate", 0.0)),
            float(row.get("mean_return", 0.0)),
            _epoch_number(str(row.get("label", ""))),
        ),
        reverse=True,
    )
    summary = {
        "source": "worker_partials",
        "sweep_dir": str(sweep_dir),
        "worker_partials": worker_partials,
        "checkpoints_completed": len(completed),
        "checkpoint_results": completed,
        "checkpoint_ranking": ranking,
    }
    output_path = Path(args.output) if args.output else sweep_dir / "sweep_summary_from_partials.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"merged {len(completed)} completed checkpoint results from {len(worker_paths)} worker files")
    print(f"wrote {output_path}")
    if ranking:
        best = ranking[0]
        print(
            "best:",
            best.get("label"),
            "success=",
            f"{float(best.get('success_rate', 0.0)):.3f}",
            "mean_return=",
            f"{float(best.get('mean_return', 0.0)):.3f}",
        )


def _epoch_number(label: str) -> int:
    if label.startswith("epoch_"):
        try:
            return int(label.removeprefix("epoch_"))
        except ValueError:
            return -1
    return -1


def _worker_paths(sweep_dir: Path) -> list[Path]:
    paths_by_worker: dict[str, Path] = {}
    for path in sorted(sweep_dir.glob("worker_*.json")):
        if path.name.endswith("_partial.json"):
            worker_key = path.name.removesuffix("_partial.json")
            paths_by_worker.setdefault(worker_key, path)
            continue
        worker_key = path.name.removesuffix(".json")
        paths_by_worker[worker_key] = path
    return [paths_by_worker[key] for key in sorted(paths_by_worker)]


if __name__ == "__main__":
    main()
