"""Run one multi-turn red-team episode.

Use `--dry-run` to test environment wiring without loading model weights.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.attacker import Attacker, EvolvingAttacker
from redteam_rl.config import load_config
from redteam_rl.env import RedTeamEnv
from redteam_rl.logging import JsonlEpisodeLogger
from redteam_rl.mutators import LLMMutator, TemplateMutator
from redteam_rl.policy import RandomPolicy, TrailBlazerPolicy
from redteam_rl.rewards import build_reward_model
from redteam_rl.seed_prompts import sample_seed_prompt
from redteam_rl.trajectory_bank import append_episode
from redteam_rl.versioning import build_model_metadata, new_run_id
from redteam_rl.victims import EvolvingVictim, VLLMVictim


class FakeVictim:
    def respond(self, prompt, state, victim_history_turns=0):
        del state, victim_history_turns
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
    parser.add_argument("--max_turns", type=int, default=None)
    parser.add_argument("--victim-history-turns", type=int, default=None)
    parser.add_argument("--log_file", type=str, default="outputs/dry_runs/episodes.jsonl")
    parser.add_argument("--trajectory-bank", type=str, default="outputs/trajectory_bank/episodes.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-template-mutator", action="store_true")
    parser.add_argument("--show-mutator-input", action="store_true")
    parser.add_argument("--show-victim-input", action="store_true")
    parser.add_argument("--attacker-lora-adapter", type=str, default=None)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
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
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional Weights & Biases project name to log to")
    return parser.parse_args()


def _judge_model_name(cfg, reward_backend: str) -> str:
    if reward_backend in {"qwen_judge", "qwen_safety_judge"}:
        return cfg.models.qwen_safety_judge
    if reward_backend == "prompt_guard":
        return cfg.models.prompt_guard
    if reward_backend == "llama_guard":
        return cfg.models.llama_guard
    return "fake"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_id = new_run_id("local_episode")
    # Optional Weights & Biases integration
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=run_id)
            wandb_run = wandb
        except Exception as e:
            print(f"wandb init failed: {e}")
    seed_prompt = args.seed_prompt or sample_seed_prompt(args.seed_prompt_file, seed=args.seed)
    if args.attacker_lora_adapter and (args.dry_run or args.use_template_mutator):
        raise ValueError("--attacker-lora-adapter requires the LLM mutator.")

    policy = TrailBlazerPolicy.from_checkpoint(args.policy_checkpoint) if args.policy_checkpoint else RandomPolicy()
    if args.dry_run or args.use_template_mutator:
        mutator = TemplateMutator()
    else:
        config_initial_attacker_adapter = cfg.attacker_evolution.get("initial_adapter_path")
        use_attacker_lora = args.attacker_lora_adapter is not None or config_initial_attacker_adapter is not None
        mutator = LLMMutator(
            cfg.mutator_config(enable_lora=use_attacker_lora or cfg.mutator.get("enable_lora", False)),
            capture_debug_prompt=args.show_mutator_input,
            lora_adapter_path=args.attacker_lora_adapter,
        )

    attacker = (
        EvolvingAttacker(
            policy=policy,
            mutator=mutator,
            config=cfg.attacker_evolution_config(
                initial_adapter_path=args.attacker_lora_adapter
                or cfg.attacker_evolution.get("initial_adapter_path")
            ),
        )
        if cfg.attacker_evolution.get("enabled", False) or args.attacker_lora_adapter
        else Attacker(policy=policy, mutator=mutator)
    )

    if args.dry_run:
        victim = FakeVictim()
        reward_model = FakeReward()
        reward_backend = "fake"
    else:
        victim = EvolvingVictim(
            VLLMVictim(cfg.victim_config(), capture_debug_prompt=args.show_victim_input)
        )
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
        config=cfg.env_config(
            **({"max_turns": args.max_turns} if args.max_turns is not None else {}),
            **(
                {"victim_history_turns": args.victim_history_turns}
                if args.victim_history_turns is not None
                else {}
            ),
        ),
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
        # Log per-turn metrics to W&B if enabled
        if wandb_run:
            try:
                metrics = {"turn": len(state.turns), "reward": float(reward)}
                jl = None
                if isinstance(turn.metadata, dict):
                    jl = turn.metadata.get("judge_label")
                if jl is not None:
                    metrics["judge_label"] = jl
                wandb_run.log(metrics, step=len(state.turns))
            except Exception:
                pass

    run_metadata = build_model_metadata(
        run_id=run_id,
        victim_model="fake" if args.dry_run else cfg.models.victim,
        victim_adapter_path=None,
        attacker_model="template" if args.dry_run or args.use_template_mutator else cfg.models.mutator,
        attacker_adapter_path=args.attacker_lora_adapter,
        policy_type="trailblazer" if args.policy_checkpoint else "random",
        policy_checkpoint=args.policy_checkpoint,
        judge_model=_judge_model_name(cfg, reward_backend),
        extra={
            "config": args.config,
            "dry_run": args.dry_run,
            "max_turns": env.config.max_turns,
            "victim_history_turns": env.config.victim_history_turns,
            "seed_prompt_file": args.seed_prompt_file,
        },
    )
    for turn in state.turns:
        turn.metadata.update(
            {
                "run_id": run_metadata["run_id"],
                "victim_model": run_metadata["victim_model"],
                "victim_version": run_metadata["victim_version"],
                "victim_adapter_path": run_metadata["victim_adapter_path"],
                "attacker_model": run_metadata["attacker_model"],
                "attacker_version": run_metadata["attacker_version"],
                "attacker_adapter_path": run_metadata["attacker_adapter_path"],
                "judge_model": run_metadata["judge_model"],
                "policy_type": run_metadata.get("policy_type"),
                "policy_checkpoint": run_metadata.get("policy_checkpoint"),
            }
        )

    logger.log_episode(
        state,
        metadata=run_metadata,
    )
    append_episode(
        state,
        path=args.trajectory_bank,
        metadata={
            **run_metadata,
            "source": "scripts/run_episode.py",
        },
    )
    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass
    print(f"logged episode to {args.log_file}")
    print(f"appended episode to {args.trajectory_bank}")


if __name__ == "__main__":
    main()
