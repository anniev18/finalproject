"""Convert AdvBench/HarmBench-style files into seed prompt JSON."""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.seed_prompts import PROMPT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="CSV or JSON/JSONL benchmark file. Can be passed more than once.",
    )
    parser.add_argument("--output", required=True, help="Output JSON seed prompt file.")
    parser.add_argument("--source", required=True, choices=["advbench", "harmbench", "custom"])
    parser.add_argument("--split", default=None, help="Optional split label to attach to records.")
    parser.add_argument(
        "--field",
        default=None,
        help="Optional explicit field name to read instead of auto-detecting common benchmark fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = []
    seen = set()
    for input_path in args.input:
        records = load_records(Path(input_path))
        for index, record in enumerate(records):
            prompt = prompt_from_record(record, field=args.field)
            if not prompt or prompt in seen:
                continue
            seen.add(prompt)
            output.append(
                {
                    "instruction": prompt,
                    "source": args.source,
                    "split": args.split,
                    "source_file": input_path,
                    "source_index": index,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"wrote {len(output)} prompts to {output_path}")


def load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [record if isinstance(record, dict) else {"instruction": str(record)} for record in data]
    for key in ("data", "behaviors", "records", "items"):
        if isinstance(data.get(key), list):
            return [record if isinstance(record, dict) else {"instruction": str(record)} for record in data[key]]
    raise ValueError(f"Unsupported JSON shape in {path}")


def prompt_from_record(record: dict, field: str | None = None) -> str:
    if field is not None:
        value = record.get(field)
        if value:
            return str(value).strip()
        raise KeyError(f"Could not find requested field {field!r} in {record.keys()}")
    for field in PROMPT_FIELDS:
        value = record.get(field)
        if value:
            return str(value).strip()
    raise KeyError(f"Could not find prompt field in {record.keys()}")


if __name__ == "__main__":
    main()
