#!/usr/bin/env python3
"""Create deterministic AdvBench splits with paired unaligned responses."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="rlbreaker_reference/datasets/advbench.csv",
        help="RLbreaker AdvBench CSV containing question/response pairs.",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    records = load_pairs(Path(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(records)

    train_end = int(len(records) * args.train_frac)
    val_end = train_end + int(len(records) * args.val_frac)
    split_rows = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }
    splits = {
        split: [{**record, "split": split} for record in split_records]
        for split, split_records in split_rows.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [record for split in ("train", "val", "test") for record in splits[split]]
    write_json(output_dir / "advbench_reference_all.json", all_records)
    for split, split_records in splits.items():
        write_json(output_dir / f"advbench_reference_{split}.json", split_records)

    all_prompts = {record["instruction"] for record in all_records}
    assert len(all_prompts) == len(all_records)
    assert sum(len(split) for split in splits.values()) == len(all_records)
    assert all(record["reference_response"] for record in all_records)
    assert not ({record["instruction"] for record in splits["train"]} & {record["instruction"] for record in splits["val"]})
    assert not ({record["instruction"] for record in splits["train"]} & {record["instruction"] for record in splits["test"]})
    assert not ({record["instruction"] for record in splits["val"]} & {record["instruction"] for record in splits["test"]})

    print(f"paired AdvBench records: {len(all_records)}")
    for split, split_records in splits.items():
        print(f"{split}: {len(split_records)} prompts, {len(split_records)} reference responses")


def load_pairs(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    seen = set()
    for source_index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        response = str(row.get("response", "")).strip()
        if not question or not response:
            raise ValueError(f"Missing question/response at source row {source_index}.")
        if question in seen:
            raise ValueError(f"Duplicate AdvBench question: {question}")
        seen.add(question)
        records.append(
            {
                "instruction": question,
                "reference_response": response,
                "source": "rlbreaker_advbench_unaligned",
                "source_file": str(path),
                "source_index": source_index,
            }
        )
    return records


def write_json(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
