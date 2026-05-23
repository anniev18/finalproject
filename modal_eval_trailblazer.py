"""Modal entrypoint for comparing RandomPolicy vs TrailBlazerPolicy.

This wrapper runs both policies on the same set of seeds and returns a compact
summary with return, turn-count, and success metrics. It is intended as a fast
milestone experiment harness.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import modal


app = modal.App("cs224r-trailblazer-eval")
volume = modal.Volume.from_name("cs224r-redteam-rl-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-gpu.txt")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("redteam_rl", remote_path="/root/redteam_rl")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def evaluate_trailblazer_remote(
    trailblazer_checkpoint: str,
    num_episodes: int = 10,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    reward_backend: str = "fake",
    wandb_project: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import sys

    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import TemplateMutator
    from redteam_rl.policy import RandomPolicy, TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.versioning import build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config("/root/configs/default.json")
    run_id = new_run_id("modal_trailblazer_eval")

    def build_env(policy):
        attacker = Attacker(policy=policy, mutator=TemplateMutator())
        victim = EvolvingVictim(
            VLLMVictim(
                cfg.victim_config(enable_lora=False),
                capture_debug_prompt=True,
            )
        )
        reward_model = build_reward_model(
            reward_backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
        )
        return RedTeamEnv(
            config=cfg.env_config(max_turns=max_turns),
            attacker=attacker,
            victim=victim,
            reward_model=reward_model,
        )

    def run_policy(policy_name: str, policy) -> list[dict]:
        env = build_env(policy)
        episodes = []
        for episode_index in range(num_episodes):
            episode_seed = seed + episode_index
            selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
            state = env.reset(selected_seed_prompt)
            done = False
            while not done:
                state, _, done, _ = env.step()
            turns = state.turns
            rewards = [float(turn.reward or 0.0) for turn in turns]
            unsafe_turns = [turn for turn in turns if str(turn.metadata.get("judge_label", "")).upper() == "UNSAFE"]
            success = bool(unsafe_turns)
            episodes.append(
                {
                    "policy": policy_name,
                    "seed": episode_seed,
                    "seed_prompt": state.seed_prompt,
                    "num_turns": len(turns),
                    "episode_return": sum(rewards),
                    "success": success,
                    "judge_labels": [turn.metadata.get("judge_label") for turn in turns],
                    "turns": [
                        {
                            "action": turn.action.value if turn.action else None,
                            "reward": turn.reward,
                            "judge_label": turn.metadata.get("judge_label"),
                        }
                        for turn in turns
                    ],
                }
            )
        return episodes

    random_episodes = run_policy("random", RandomPolicy())
    trailblazer_policy = TrailBlazerPolicy.from_checkpoint(trailblazer_checkpoint)
    trailblazer_episodes = run_policy("trailblazer", trailblazer_policy)

    def summarize(episodes: list[dict]) -> dict:
        total = len(episodes)
        successes = [episode for episode in episodes if episode["success"]]
        summary = {
            "num_episodes": total,
            "mean_return": sum(episode["episode_return"] for episode in episodes) / total if total else 0.0,
            "mean_turns": sum(episode["num_turns"] for episode in episodes) / total if total else 0.0,
            "success_rate": len(successes) / total if total else 0.0,
            "queries_per_success": (
                sum(episode["num_turns"] for episode in successes) / len(successes)
                if successes
                else None
            ),
        }
        return summary

    summary = {
        "run_id": run_id,
        "config": {
            "num_episodes": num_episodes,
            "max_turns": max_turns,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "reward_backend": reward_backend,
            "trailblazer_checkpoint": trailblazer_checkpoint,
        },
        "random": summarize(random_episodes),
        "trailblazer": summarize(trailblazer_episodes),
        "random_episodes": random_episodes,
        "trailblazer_episodes": trailblazer_episodes,
        "metadata": build_model_metadata(
            run_id=run_id,
            victim_model=cfg.models.victim,
            attacker_model=cfg.models.mutator,
            judge_model=cfg.reward_backend(),
            policy_type="comparison",
            policy_checkpoint=trailblazer_checkpoint,
            extra={
                "num_episodes": num_episodes,
                "max_turns": max_turns,
                "seed": seed,
                "reward_backend": reward_backend,
            },
        ),
    }

    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name=run_id)
            wandb.define_metric("random/episode_index")
            wandb.define_metric("random/*", step_metric="random/episode_index")
            wandb.define_metric("trailblazer/episode_index")
            wandb.define_metric("trailblazer/*", step_metric="trailblazer/episode_index")
            for episode in random_episodes:
                wandb_run.log(
                    {
                        "random/episode_index": episode["seed"] - seed,
                        "random/episode_seed": episode["seed"],
                        "random/episode_return": episode["episode_return"],
                        "random/num_turns": episode["num_turns"],
                        "random/success": int(episode["success"]),
                    }
                )
            for episode in trailblazer_episodes:
                wandb_run.log(
                    {
                        "trailblazer/episode_index": episode["seed"] - seed,
                        "trailblazer/episode_seed": episode["seed"],
                        "trailblazer/episode_return": episode["episode_return"],
                        "trailblazer/num_turns": episode["num_turns"],
                        "trailblazer/success": int(episode["success"]),
                    }
                )
            wandb_run.summary["random/success_rate"] = summary["random"]["success_rate"]
            wandb_run.summary["random/mean_return"] = summary["random"]["mean_return"]
            wandb_run.summary["random/mean_turns"] = summary["random"]["mean_turns"]
            wandb_run.summary["trailblazer/success_rate"] = summary["trailblazer"]["success_rate"]
            wandb_run.summary["trailblazer/mean_return"] = summary["trailblazer"]["mean_return"]
            wandb_run.summary["trailblazer/mean_turns"] = summary["trailblazer"]["mean_turns"]
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    trailblazer_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_9.pt",
    num_episodes: int = 10,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    reward_backend: str = "fake",
    wandb_project: str | None = None,
    output_dir: str = "outputs/eval",
    save_local: bool = False,
) -> None:
    result = evaluate_trailblazer_remote.remote(
        trailblazer_checkpoint=trailblazer_checkpoint,
        num_episodes=num_episodes,
        max_turns=max_turns,
        seed=seed,
        seed_prompt_file=seed_prompt_file,
        reward_backend=reward_backend,
        wandb_project=wandb_project,
    )
    print(json.dumps(result, indent=2))
    print("\nsaved comparison results to Modal volume cs224r-redteam-rl-data")
    print("remote output path: /root/outputs/eval")
    if save_local:
        os.makedirs(output_dir, exist_ok=True)
        local_summary_path = Path(output_dir) / "modal_trailblazer_eval_summary.json"
        with local_summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")
