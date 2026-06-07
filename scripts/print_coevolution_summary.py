"""Print compact stats from a co-evolution run summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize modal_coevolve_full.py output.")
    parser.add_argument("summary_json", help="Path to coevolution summary.json.")
    parser.add_argument("--title", default=None, help="Optional printed title.")
    parser.add_argument("--top-epochs", type=int, default=5, help="Number of best PPO epochs to print.")
    args = parser.parse_args()

    with Path(args.summary_json).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    title = args.title or Path(args.summary_json).parent.name
    cycles = summary.get("cycles", [])
    print(title)
    print("=" * 112)
    print(f"cycles_completed: {len(cycles)} / {summary.get('config', {}).get('cycles', 'unknown')}")
    print(f"initial_policy: {summary.get('initial_policy_checkpoint')}")
    print(f"final_policy:   {summary.get('final_policy_checkpoint')}")
    print(f"final_attacker: {summary.get('final_attacker_adapter') or 'none'}")
    print(f"final_victim:   {summary.get('final_victim_adapter') or 'none'}")

    print("\nCycle Summary")
    print("=" * 112)
    header = (
        "cycle  epochs     ppo_sr  ppo_ret  q/succ  victim_ex  victim_ok  "
        "attacker_ex  attacker_ok  victim_version"
    )
    print(header)
    print("-" * len(header))
    for cycle in cycles:
        cycle_index = cycle.get("cycle_index")
        policy = cycle.get("policy", {})
        victim = cycle.get("victim", {})
        attacker = cycle.get("attacker", {})
        last_epoch = _last_epoch(policy)
        rollout = last_epoch.get("rollout_metrics", {})
        epochs = _epoch_span(policy)
        print(
            f"{cycle_index:<5}  "
            f"{epochs:<9}  "
            f"{_fmt(last_epoch.get('success_rate')):<6}  "
            f"{_fmt(rollout.get('episode_return_mean')):<7}  "
            f"{_fmt(last_epoch.get('queries_per_success')):<6}  "
            f"{_int(victim.get('training_examples')):<9}  "
            f"{str(bool(victim.get('adapter_valid'))):<9}  "
            f"{_int(attacker.get('training_examples')):<11}  "
            f"{str(bool(attacker.get('adapter_valid'))):<11}  "
            f"{victim.get('output_victim_version') or _short(victim.get('active_adapter_path'))}"
        )

    all_epochs = _all_epochs(cycles)
    if all_epochs:
        print("\nBest PPO Epochs By Rollout Success")
        print("=" * 112)
        ranked = sorted(
            all_epochs,
            key=lambda row: (
                float(row["epoch"].get("success_rate") or 0.0),
                float((row["epoch"].get("rollout_metrics") or {}).get("episode_return_mean") or 0.0),
            ),
            reverse=True,
        )
        print("rank  cycle  epoch  success  mean_return  q/success  checkpoint")
        print("-" * 112)
        for rank, row in enumerate(ranked[: args.top_epochs], start=1):
            epoch = row["epoch"]
            rollout = epoch.get("rollout_metrics", {})
            print(
                f"{rank:<4}  "
                f"{row['cycle']:<5}  "
                f"{epoch.get('epoch'):<5}  "
                f"{_fmt(epoch.get('success_rate')):<7}  "
                f"{_fmt(rollout.get('episode_return_mean')):<11}  "
                f"{_fmt(epoch.get('queries_per_success')):<9}  "
                f"{epoch.get('checkpoint_path')}"
            )

    print("\nTakeaways")
    print("=" * 112)
    if cycles:
        first = _last_epoch(cycles[0].get("policy", {}))
        last = _last_epoch(cycles[-1].get("policy", {}))
        print(
            "final-epoch PPO success changed "
            f"{_fmt(first.get('success_rate'))} -> {_fmt(last.get('success_rate'))} "
            f"from cycle {cycles[0].get('cycle_index')} to cycle {cycles[-1].get('cycle_index')}."
        )
    victim_updates = sum(1 for cycle in cycles if cycle.get("victim", {}).get("adapter_valid"))
    attacker_updates = sum(1 for cycle in cycles if cycle.get("attacker", {}).get("adapter_valid"))
    print(f"valid victim updates: {victim_updates}")
    print(f"valid attacker updates: {attacker_updates}")
    total_victim_examples = sum(int(cycle.get("victim", {}).get("training_examples") or 0) for cycle in cycles)
    total_attacker_examples = sum(int(cycle.get("attacker", {}).get("training_examples") or 0) for cycle in cycles)
    print(f"total victim training examples counted across cycles: {total_victim_examples}")
    print(f"total attacker training examples counted across cycles: {total_attacker_examples}")


def _last_epoch(policy: dict[str, Any]) -> dict[str, Any]:
    epochs = policy.get("epochs") or []
    return epochs[-1] if epochs else {}


def _epoch_span(policy: dict[str, Any]) -> str:
    start = policy.get("policy_start_epoch")
    end = policy.get("policy_end_epoch")
    if start is None or end is None:
        return "n/a"
    return f"{start}-{end}"


def _all_epochs(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cycle in cycles:
        for epoch in cycle.get("policy", {}).get("epochs") or []:
            rows.append({"cycle": cycle.get("cycle_index"), "epoch": epoch})
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> str:
    if value is None:
        return "0"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _short(value: Any) -> str:
    if not value:
        return "none"
    return str(value).rstrip("/").split("/")[-1]


if __name__ == "__main__":
    main()
