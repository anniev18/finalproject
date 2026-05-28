"""Score saved episode JSONL files with a selected reward backend.

This is intended for offline Llama Guard evaluation after trajectories are
collected with cheaper in-loop rewards.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.config import load_config
from redteam_rl.rewards import build_reward_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/dry_runs/episodes.jsonl")
    parser.add_argument("--output", type=str, default="outputs/dry_runs/scored_episodes.jsonl")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument(
        "--reward-backend",
        type=str,
        default="llama_guard",
        choices=["prompt_guard", "qwen_judge", "llama_guard", "wildguard"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    reward_model = build_reward_model(
        args.reward_backend,
        prompt_guard_config=cfg.prompt_guard_config(),
        qwen_judge_config=cfg.qwen_judge_config(),
        llama_guard_config=cfg.llama_guard_config(),
        wildguard_config=cfg.wildguard_config(),
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            record = json.loads(line)
            episode = record["episode"]
            for turn in episode["turns"]:
                score = reward_model.score(
                    turn["user_message"],
                    turn["victim_response"],
                    state=None,
                )
                turn.setdefault("offline_scores", {})[args.reward_backend] = score
            dst.write(json.dumps(record) + "\n")

    print(f"wrote scored episodes to {output_path}")


if __name__ == "__main__":
    main()
