#!/usr/bin/env python3
"""Download AdvBench/HarmBench prompt files and build seed_prompts JSON.

This script intentionally uses only the Python standard library so it can run
before optional ML dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_benchmark_prompts import load_records, prompt_from_record


@dataclass(frozen=True)
class BenchmarkFile:
    benchmark: str
    split: str
    url: str
    local_path: Path


BENCHMARK_FILES: dict[str, tuple[BenchmarkFile, ...]] = {
    "advbench": (
        BenchmarkFile(
            benchmark="advbench",
            split="all",
            url=(
                "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
                "data/advbench/harmful_behaviors.csv"
            ),
            local_path=Path("data/benchmarks/advbench/harmful_behaviors.csv"),
        ),
    ),
    "harmbench": (
        BenchmarkFile(
            benchmark="harmbench",
            split="val",
            url=(
                "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                "data/behavior_datasets/harmbench_behaviors_text_val.csv"
            ),
            local_path=Path("data/benchmarks/harmbench/harmbench_behaviors_text_val.csv"),
        ),
        BenchmarkFile(
            benchmark="harmbench",
            split="test",
            url=(
                "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                "data/behavior_datasets/harmbench_behaviors_text_test.csv"
            ),
            local_path=Path("data/benchmarks/harmbench/harmbench_behaviors_text_test.csv"),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(BENCHMARK_FILES),
        help="Benchmark to download. Can be passed more than once. Defaults to both.",
    )
    parser.add_argument(
        "--output",
        default="data/seed_prompts.json",
        help="Combined all-prompts JSON output.",
    )
    parser.add_argument(
        "--split-output-dir",
        default="data",
        help="Directory for seed_prompts_train/val/test.json split outputs.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/benchmarks",
        help="Directory for downloaded raw benchmark files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Only download raw files; do not write seed prompt JSON.",
    )
    parser.add_argument(
        "--extra-train-file",
        action="append",
        default=[],
        help="Additional local CSV/JSON/JSONL prompt file to merge into the training split only.",
    )
    parser.add_argument(
        "--extra-split-file",
        action="append",
        default=[],
        help=(
            "Additional local CSV/JSON/JSONL prompt file to split deterministically "
            "across train/val/test. Defaults to data/seed_prompts_legacy.json when present."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=224, help="Seed for deterministic AdvBench split.")
    parser.add_argument(
        "--advbench-split-mode",
        choices=["train_only", "deterministic"],
        default="train_only",
        help=(
            "How to use AdvBench. train_only puts all AdvBench prompts in train. "
            "deterministic makes an 80/10/10 train/val/test split."
        ),
    )
    parser.add_argument("--advbench-train-frac", type=float, default=0.8)
    parser.add_argument("--advbench-val-frac", type=float, default=0.1)
    parser.add_argument("--extra-train-frac", type=float, default=0.8)
    parser.add_argument("--extra-val-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_names = args.benchmark or sorted(BENCHMARK_FILES)
    files = [
        with_raw_dir(benchmark_file, Path(args.raw_dir))
        for name in benchmark_names
        for benchmark_file in BENCHMARK_FILES[name]
    ]

    for benchmark_file in files:
        download_file(benchmark_file, force=args.force)

    if args.no_convert:
        return

    splits = build_seed_prompt_splits(
        files,
        extra_train_files=[Path(path) for path in args.extra_train_file],
        extra_split_files=default_extra_split_files(args.extra_split_file),
        split_seed=args.split_seed,
        advbench_split_mode=args.advbench_split_mode,
        advbench_train_frac=args.advbench_train_frac,
        advbench_val_frac=args.advbench_val_frac,
        extra_train_frac=args.extra_train_frac,
        extra_val_frac=args.extra_val_frac,
    )
    records = [record for split in ("train", "val", "test") for record in splits[split]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    print(f"wrote {len(records)} prompts to {output_path}")

    split_output_dir = Path(args.split_output_dir)
    split_output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_records in splits.items():
        split_path = split_output_dir / f"seed_prompts_{split}.json"
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(split_records, f, indent=2)
            f.write("\n")
        print(f"wrote {len(split_records)} {split} prompts to {split_path}")


def with_raw_dir(benchmark_file: BenchmarkFile, raw_dir: Path) -> BenchmarkFile:
    relative_name = benchmark_file.local_path.name
    return BenchmarkFile(
        benchmark=benchmark_file.benchmark,
        split=benchmark_file.split,
        url=benchmark_file.url,
        local_path=raw_dir / benchmark_file.benchmark / relative_name,
    )


def download_file(benchmark_file: BenchmarkFile, force: bool = False) -> None:
    path = benchmark_file.local_path
    if path.exists() and not force:
        print(f"exists {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        benchmark_file.url,
        headers={"User-Agent": "cs224r-redteam-rl/benchmark-downloader"},
    )
    print(f"downloading {benchmark_file.url}")
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())
    print(f"wrote {path}")


def build_seed_prompt_splits(
    files: list[BenchmarkFile],
    extra_train_files: list[Path],
    extra_split_files: list[Path],
    split_seed: int,
    advbench_split_mode: str,
    advbench_train_frac: float,
    advbench_val_frac: float,
    extra_train_frac: float,
    extra_val_frac: float,
) -> dict[str, list[dict[str, object]]]:
    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    seen_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    seen_global: set[str] = set()
    for benchmark_file in files:
        rows = load_records(benchmark_file.local_path)
        if benchmark_file.benchmark == "advbench":
            records = [
                make_seed_prompt_record(
                    row=row,
                    benchmark_file=benchmark_file,
                    source_index=index,
                    split="unsplit",
                )
                for index, row in enumerate(rows)
            ]
            records = dedupe_records(records)
            if advbench_split_mode == "train_only":
                split_groups = (("train", records),)
            elif advbench_split_mode == "deterministic":
                rng = random.Random(split_seed)
                rng.shuffle(records)
                train_end = int(len(records) * advbench_train_frac)
                val_end = train_end + int(len(records) * advbench_val_frac)
                split_groups = (
                    ("train", records[:train_end]),
                    ("val", records[train_end:val_end]),
                    ("test", records[val_end:]),
                )
            else:
                raise ValueError(f"Unsupported AdvBench split mode: {advbench_split_mode}")
            for split, split_records in split_groups:
                for record in split_records:
                    add_record(splits, seen_by_split, seen_global, with_split(record, split))
        elif benchmark_file.benchmark == "harmbench":
            if benchmark_file.split not in {"val", "test"}:
                raise ValueError(f"Unexpected HarmBench split: {benchmark_file.split}")
            for index, row in enumerate(rows):
                add_record(
                    splits,
                    seen_by_split,
                    seen_global,
                    make_seed_prompt_record(
                        row=row,
                        benchmark_file=benchmark_file,
                        source_index=index,
                        split=benchmark_file.split,
                    ),
                )
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark_file.benchmark}")

    for extra_path in extra_train_files:
        rows = load_records(extra_path)
        for index, row in enumerate(rows):
            add_record(
                splits,
                seen_by_split,
                seen_global,
                {
                    "instruction": prompt_from_record(row),
                    "source": extra_path.stem,
                    "split": "train",
                    "source_file": str(extra_path),
                    "source_index": index,
                },
            )

    for extra_path in extra_split_files:
        records = []
        for index, row in enumerate(load_records(extra_path)):
            prompt = prompt_from_record(row)
            if not prompt or prompt in seen_global:
                continue
            records.append(
                {
                    "instruction": prompt,
                    "source": extra_path.stem,
                    "split": "unsplit",
                    "source_file": str(extra_path),
                    "source_index": index,
                }
            )
        records = dedupe_records(records)
        rng = random.Random(split_seed)
        rng.shuffle(records)
        train_end = int(len(records) * extra_train_frac)
        val_end = train_end + int(len(records) * extra_val_frac)
        for split, split_records in (
            ("train", records[:train_end]),
            ("val", records[train_end:val_end]),
            ("test", records[val_end:]),
        ):
            for record in split_records:
                add_record(splits, seen_by_split, seen_global, with_split(record, split))
    return splits


def default_extra_split_files(paths: list[str]) -> list[Path]:
    selected = [Path(path) for path in paths]
    default_legacy_path = Path("data/seed_prompts_legacy.json")
    if default_legacy_path.exists() and default_legacy_path not in selected:
        selected.append(default_legacy_path)
    return selected


def make_seed_prompt_record(
    row: dict,
    benchmark_file: BenchmarkFile,
    source_index: int,
    split: str,
) -> dict[str, object]:
    return {
        "instruction": prompt_from_record(row),
        "source": benchmark_file.benchmark,
        "split": split,
        "source_file": str(benchmark_file.local_path),
        "source_index": source_index,
    }


def with_split(record: dict[str, object], split: str) -> dict[str, object]:
    updated = dict(record)
    updated["split"] = split
    return updated


def add_record(
    splits: dict[str, list[dict[str, object]]],
    seen_by_split: dict[str, set[str]],
    seen_global: set[str],
    record: dict[str, object],
) -> None:
    prompt = str(record["instruction"])
    split = str(record["split"])
    if not prompt or prompt in seen_by_split[split] or prompt in seen_global:
        return
    seen_by_split[split].add(prompt)
    seen_global.add(prompt)
    splits[split].append(record)


def dedupe_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    seen = set()
    for record in records:
        prompt = str(record["instruction"])
        if prompt and prompt not in seen:
            seen.add(prompt)
            output.append(record)
    return output


if __name__ == "__main__":
    main()
