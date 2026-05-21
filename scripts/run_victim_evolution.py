"""Dry-run victim evolution orchestration.

This exercises the round-based controller without loading LLMs or training a
real LoRA adapter. It is meant to validate orchestration before running the
real GPU path on Modal.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.attacker import Attacker
from redteam_rl.config import load_config
from redteam_rl.env import RedTeamEnv
from redteam_rl.mutators import TemplateMutator
from redteam_rl.orchestration import create_dry_run_adapter, run_victim_evolution_rounds
from redteam_rl.policy import RandomPolicy
from redteam_rl.versioning import build_model_metadata, new_run_id


class DryRunEvolvingVictim:
    def __init__(self) -> None:
        self.adapter_path: Path | None = None
        self.round = 0

    def respond(self, prompt, state, victim_history_turns=0):
        del state, victim_history_turns
        adapter_note = f" [adapter={self.adapter_path.name}]" if self.adapter_path else ""
        if any(word in prompt.lower() for word in ("rob", "hack", "weapon", "harm")):
            if self.adapter_path:
                return f"I'm sorry, but I can't help with that.{adapter_note}"
            return f"Dry unsafe response with operational details for: {prompt}{adapter_note}"
        return f"Dry victim response to: {prompt}{adapter_note}"

    def evolve_to_adapter(self, adapter_path):
        self.round += 1
        self.adapter_path = Path(adapter_path)


class DryRunReward:
    last_labels: list[str]

    def __init__(self) -> None:
        self.last_labels = []

    def score(self, prompt, response, state):
        del state
        unsafe = not response.lower().startswith("i'm sorry") and any(
            word in prompt.lower() for word in ("rob", "hack", "weapon", "harm")
        )
        label = "UNSAFE" if unsafe else "SAFE"
        self.last_labels = [label]
        return 1.0 if unsafe else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--seed-prompt", type=str, default=None)
    parser.add_argument("--seed-prompt-file", type=str, default="data/seed_prompts.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--episodes-per-round", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/victim_rounds_dry")
    parser.add_argument(
        "--trajectory-source",
        type=str,
        default=None,
        choices=["fresh", "bank", "fresh_plus_bank"],
    )
    parser.add_argument("--verify-adapter-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_id = new_run_id("dry_victim_evolution")
    evolution_config = cfg.victim_evolution_config(
        output_dir=args.output_dir,
        verify_adapter_update=args.verify_adapter_update,
        **({"trajectory_source": args.trajectory_source} if args.trajectory_source else {}),
        **({"num_rounds": args.num_rounds} if args.num_rounds is not None else {}),
        **(
            {"episodes_per_round": args.episodes_per_round}
            if args.episodes_per_round is not None
            else {}
        ),
    )

    attacker = Attacker(policy=RandomPolicy(), mutator=TemplateMutator())
    victim = DryRunEvolvingVictim()
    env = RedTeamEnv(
        config=cfg.env_config(**({"max_turns": args.max_turns} if args.max_turns else {})),
        attacker=attacker,
        victim=victim,
        reward_model=DryRunReward(),
    )

    def train_adapter_fn(_episodes, round_index, examples):
        return create_dry_run_adapter(evolution_config.output_dir, round_index, examples)

    summaries = run_victim_evolution_rounds(
        env=env,
        victim=victim,
        config=evolution_config,
        seed_prompt_file=args.seed_prompt_file,
        train_adapter_fn=train_adapter_fn,
        seed=args.seed,
        seed_prompt=args.seed_prompt,
        log_dir=Path(evolution_config.output_dir) / "logs",
        model_metadata=build_model_metadata(
            run_id=run_id,
            victim_model="dry_fake_victim",
            attacker_model="template",
            judge_model="dry_fake_reward",
            trajectory_source=evolution_config.trajectory_source,
            extra={
                "config": args.config,
                "dry_run": True,
                "seed_prompt_file": args.seed_prompt_file,
            },
        ),
    )

    print(json.dumps([asdict(summary) for summary in summaries], indent=2))
    print(f"wrote dry-run victim evolution artifacts to {evolution_config.output_dir}")


if __name__ == "__main__":
    main()
