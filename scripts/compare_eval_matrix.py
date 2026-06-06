"""Print the final fixed-vs-coevolved evaluation matrix.

This combines the already-run fixed-policy/base-victim eval with the matrix
summary for the remaining cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed and coevolved PPO eval results.")
    parser.add_argument(
        "--fixed-base-summary",
        default="outputs/eval/fixed_continued_epoch34_base_victim_test_50.json",
        help="modal_eval_trailblazer JSON for fixed policy vs base victim.",
    )
    parser.add_argument(
        "--matrix-summary",
        default="outputs/eval_matrix/final_matrix_missing_3_test_50/matrix_summary.json",
        help="modal_eval_matrix JSON containing the remaining cells.",
    )
    parser.add_argument("--title", default="Final PPO Evaluation Matrix")
    args = parser.parse_args()

    fixed_base = _load_json(Path(args.fixed_base_summary))
    matrix = _load_json(Path(args.matrix_summary))

    rows = {
        ("fixed", "base"): _metrics_from_eval_summary(fixed_base),
    }
    for row in matrix.get("table", []):
        policy = _normalize_policy(row.get("policy"))
        victim = _normalize_victim(row.get("victim"))
        if policy and victim:
            rows[(policy, victim)] = row

    print(args.title)
    print("=" * 96)
    _print_table(rows)
    print()
    _print_notes(rows, fixed_base, matrix)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_from_eval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("trailblazer", {})
    return {
        "policy": "fixed",
        "victim": "base",
        "success_rate": metrics.get("success_rate"),
        "mean_return": metrics.get("mean_return"),
        "mean_turns": metrics.get("mean_turns"),
        "queries_per_success": metrics.get("queries_per_success"),
    }


def _normalize_policy(policy: Any) -> str | None:
    value = str(policy or "").lower()
    if value in {"fixed", "fixed_continued", "fixed_policy"}:
        return "fixed"
    if value in {"coevolved", "co_evolved", "coevolved_policy"}:
        return "coevolved"
    return value or None


def _normalize_victim(victim: Any) -> str | None:
    value = str(victim or "").lower()
    if value in {"base", "base_victim"}:
        return "base"
    if value in {"evolved", "evolved_victim", "victim3", "v3"}:
        return "evolved"
    return value or None


def _print_table(rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    header = f"{'policy':<12} {'victim':<10} {'success':>8} {'mean_return':>12} {'mean_turns':>11} {'q/success':>10}"
    print(header)
    print("-" * len(header))
    for policy, victim in [
        ("fixed", "base"),
        ("fixed", "evolved"),
        ("coevolved", "base"),
        ("coevolved", "evolved"),
    ]:
        metrics = rows.get((policy, victim), {})
        print(
            f"{policy:<12} "
            f"{victim:<10} "
            f"{_fmt_float(metrics.get('success_rate')):>8} "
            f"{_fmt_float(metrics.get('mean_return')):>12} "
            f"{_fmt_float(metrics.get('mean_turns')):>11} "
            f"{_fmt_value(metrics.get('queries_per_success')):>10}"
        )


def _print_notes(rows: dict[tuple[str, str], dict[str, Any]], fixed_base: dict[str, Any], matrix: dict[str, Any]) -> None:
    print("Notes")
    print("=" * 96)
    fixed_base_config = fixed_base.get("config", {})
    matrix_config = matrix.get("config", {})
    print(f"fixed/base source: {fixed_base_config.get('trailblazer_checkpoint')}")
    print(f"matrix fixed policy: {matrix_config.get('fixed_policy_checkpoint')}")
    print(f"matrix coevolved policy: {matrix_config.get('coevolved_policy_checkpoint')}")
    print(f"matrix evolved victim: {matrix_config.get('evolved_victim_adapter')}")
    print(f"seed file: {matrix_config.get('seed_prompt_file') or fixed_base_config.get('seed_prompt_file')}")
    print(f"num episodes: {matrix_config.get('num_episodes') or fixed_base_config.get('num_episodes')}")
    print(f"max turns: {matrix_config.get('max_turns') or fixed_base_config.get('max_turns')}")
    skipped = matrix_config.get("skipped_cells") or []
    if skipped:
        print(f"skipped cells in matrix run: {', '.join(skipped)}")

    fixed_base_success = _as_float(rows.get(("fixed", "base"), {}).get("success_rate"))
    fixed_evolved_success = _as_float(rows.get(("fixed", "evolved"), {}).get("success_rate"))
    co_base_success = _as_float(rows.get(("coevolved", "base"), {}).get("success_rate"))
    co_evolved_success = _as_float(rows.get(("coevolved", "evolved"), {}).get("success_rate"))
    if fixed_base_success is not None and fixed_evolved_success is not None:
        print(f"fixed PPO success drop base->evolved: {fixed_base_success:.3f} -> {fixed_evolved_success:.3f}")
    if co_base_success is not None and co_evolved_success is not None:
        print(f"coevolved PPO success drop base->evolved: {co_base_success:.3f} -> {co_evolved_success:.3f}")


def _fmt_float(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.3f}"


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
