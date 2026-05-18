"""Run one multi-turn red-team episode.

Use `--dry-run` to test environment wiring without loading model weights.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.attacker import Attacker
from redteam_rl.config import load_config
from redteam_rl.env import EnvConfig, RedTeamEnv
from redteam_rl.logging import JsonlEpisodeLogger
from redteam_rl.mutators import LLMMutator, TemplateMutator
from redteam_rl.policy import RandomPolicy
from redteam_rl.rewards import build_reward_model
from redteam_rl.seed_prompts import sample_seed_prompt
from redteam_rl.victims import EvolvingVictim, VLLMVictim


class FakeVictim:
    def respond(self, prompt, state):
        del state
        return f"Fake victim response to: {prompt}"


class FakeReward:
    def score(self, prompt, response, state):
        del response
        return float(len(prompt) + len(state.turns)) / 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--seed-prompt", type=str, default=None)
    parser.add_argument("--seed-prompt-file", type=str, default="data/seed_prompts.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--log_file", type=str, default="outputs/episodes.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-template-mutator", action="store_true")
    parser.add_argument(
        "--reward-backend",
        type=str,
        default=None,
        choices=["fake", "prompt_guard", "qwen_judge", "qwen_safety_judge", "llama_guard"],
    )
    parser.add_argument(
        "--aux-reward-backend",
        action="append",
        default=[],
        choices=["prompt_guard", "qwen_judge", "llama_guard"],
        help="Optional auxiliary scores to log without changing the main reward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_prompt = args.seed_prompt or sample_seed_prompt(args.seed_prompt_file, seed=args.seed)

    policy = RandomPolicy()
    if args.dry_run or args.use_template_mutator:
        mutator = TemplateMutator()
    else:
        mutator = LLMMutator(cfg.mutator_config())

    attacker = Attacker(policy=policy, mutator=mutator)

    if args.dry_run:
        victim = FakeVictim()
        reward_model = FakeReward()
    else:
        victim = EvolvingVictim(VLLMVictim(cfg.victim_config()))
        reward_backend = args.reward_backend or cfg.reward_backend()
        reward_model = build_reward_model(
            reward_backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
        )
    auxiliary_reward_models = {
        backend: build_reward_model(
            backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
        )
        for backend in args.aux_reward_backend
    }

    env = RedTeamEnv(
        config=EnvConfig(max_turns=args.max_turns),
        attacker=attacker,
        victim=victim,
        reward_model=reward_model,
        auxiliary_reward_models=auxiliary_reward_models,
    )
    logger = JsonlEpisodeLogger(args.log_file)

    state = env.reset(seed_prompt)
    done = False
    while not done:
        state, reward, done, _ = env.step()
        turn = state.turns[-1]
        print(f"turn={len(state.turns)} action={turn.action.value} reward={reward:.4f}")
        print(f"user: {turn.user_message}")
        print(f"victim: {turn.victim_response}")
        print()

    logger.log_episode(
        state,
        metadata={
            "config": args.config,
            "dry_run": args.dry_run,
            "max_turns": args.max_turns,
            "seed_prompt_file": args.seed_prompt_file,
        },
    )
    print(f"logged episode to {args.log_file}")


if __name__ == "__main__":
    main()
