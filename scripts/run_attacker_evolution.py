"""Train or dry-run an attacker mutator LoRA from saved trajectories."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.attacker_training import (
    attacker_examples_from_trajectory_bank,
    create_dry_run_attacker_adapter,
    fine_tune_attacker_lora,
)
from redteam_rl.config import load_config
from redteam_rl.orchestration import verify_adapter_update
from redteam_rl.versioning import new_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--trajectory-bank", type=str, default="outputs/trajectory_bank/episodes.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/attacker_rounds")
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--victim-version", type=str, default=None)
    parser.add_argument("--attacker-version", type=str, default=None)
    parser.add_argument("--max-bank-episodes", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-adapter-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_id = new_run_id("attacker_evolution")

    examples = attacker_examples_from_trajectory_bank(
        args.trajectory_bank,
        min_reward=args.min_reward,
        victim_version=args.victim_version,
        attacker_version=args.attacker_version,
        limit=args.max_bank_episodes,
    )

    adapter_path = None
    if examples:
        if args.dry_run:
            adapter_path = create_dry_run_attacker_adapter(args.output_dir, args.round_index, examples)
        else:
            adapter_path = fine_tune_attacker_lora(
                cfg.attacker_finetune_config(
                    output_dir=Path(args.output_dir) / f"round_{args.round_index:03d}"
                ),
                examples,
            )

    verification = (
        verify_adapter_update(adapter_path)
        if args.verify_adapter_update and adapter_path is not None
        else None
    )
    summary = {
        "run_id": run_id,
        "round_index": args.round_index,
        "trajectory_bank": args.trajectory_bank,
        "training_examples": len(examples),
        "min_reward": args.min_reward,
        "victim_version_filter": args.victim_version,
        "attacker_version_filter": args.attacker_version,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_verification": verification,
        "dry_run": args.dry_run,
        "examples": [asdict(example) for example in examples[:10]],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"attacker_round_{args.round_index:03d}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote attacker evolution summary to {summary_path}")


if __name__ == "__main__":
    main()

