"""Modal entrypoint for training the TrailBlazer PPO policy.

This trains against the same red-team environment stack used by episode runs:
policy -> mutator -> victim -> reward model. For cheaper experiments, pass
``--use-template-mutator`` or ``--reward-backend=fake``.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import modal


app = modal.App("cs224r-trailblazer-train")
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
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_trailblazer_remote(
    save_dir: str = "/root/outputs/policies/trailblazer_ppo",
    epochs: int = 10,
    episodes_per_batch: int = 8,
    max_turns: int = 3,
    lr: float = 3e-4,
    config_path: str = "/root/configs/default.json",
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    resume_checkpoint: str | None = None,
    start_epoch: int | None = None,
    wandb_project: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import sys

    sys.path.insert(0, "/root")

    import torch
    import torch.optim as optim

    from redteam_rl.actions import ACTIONS
    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.types import EpisodeState
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    selected_reward_backend = reward_backend or cfg.reward_backend()

    def build_training_env(policy: TrailBlazerPolicy) -> RedTeamEnv:
        if attacker_lora_adapter and use_template_mutator:
            raise ValueError("attacker_lora_adapter requires the LLM mutator; remove --use-template-mutator.")

        config_initial_attacker_adapter = cfg.attacker_evolution.get("initial_adapter_path")
        use_attacker_lora = attacker_lora_adapter is not None or config_initial_attacker_adapter is not None
        mutator = (
            TemplateMutator()
            if use_template_mutator
            else LLMMutator(
                cfg.mutator_config(enable_lora=use_attacker_lora or cfg.mutator.get("enable_lora", False)),
                capture_debug_prompt=True,
                lora_adapter_path=attacker_lora_adapter,
            )
        )
        attacker = (
            EvolvingAttacker(
                policy=policy,
                mutator=mutator,
                config=cfg.attacker_evolution_config(
                    initial_adapter_path=attacker_lora_adapter or config_initial_attacker_adapter
                ),
            )
            if cfg.attacker_evolution.get("enabled", False) or attacker_lora_adapter
            else Attacker(policy=policy, mutator=mutator)
        )
        victim = EvolvingVictim(
            VLLMVictim(
                cfg.victim_config(enable_lora=victim_lora_adapter is not None or cfg.victim.get("enable_lora", False)),
                lora_adapter_path=victim_lora_adapter,
                capture_debug_prompt=True,
            )
        )
        reward_model = build_reward_model(
            selected_reward_backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
            wildguard_config=cfg.wildguard_config(),
        )
        return RedTeamEnv(
            config=cfg.env_config(
                max_turns=max_turns,
                **(
                    {"victim_history_turns": victim_history_turns}
                    if victim_history_turns is not None
                    else {}
                ),
            ),
            attacker=attacker,
            victim=victim,
            reward_model=reward_model,
        )

    def collect_episodes(env: RedTeamEnv, n: int, epoch: int) -> list[dict]:
        batch = []
        for episode_index in range(n):
            episode_seed = None if seed is None else seed + epoch * n + episode_index
            selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
            state = env.reset(selected_seed_prompt)
            done = False
            episode = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": []}
            while not done:
                episode["states"].append(copy.deepcopy(state))
                state, reward_value, done, info = env.step()
                latest_turn = state.turns[-1]
                episode["actions"].append(ACTIONS.index(latest_turn.action))
                episode["log_probs"].append(float(info.get("policy_log_prob", 0.0)))
                episode["values"].append(float(info.get("policy_value", 0.0)))
                episode["rewards"].append(float(reward_value))
            batch.append(episode)
        return batch

    def compute_gae(rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        values = values + [0.0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def ppo_update(policy: TrailBlazerPolicy, batch, optimizer, clip_eps=0.2, value_coef=0.5, ent_coef=0.01):
        device = policy.device
        all_states = []
        all_actions = []
        old_log_probs = []
        returns = []
        advantages = []

        for episode in batch:
            episode_returns, episode_advantages = compute_gae(episode["rewards"], episode["values"])
            returns.extend(episode_returns)
            advantages.extend(episode_advantages)
            all_actions.extend(episode["actions"])
            old_log_probs.extend(episode["log_probs"])
            all_states.extend(episode["states"])

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        entropies = []

        for index, state in enumerate(all_states):
            logits, value, _ = policy.forward_from_state(state)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)

            action = torch.tensor(all_actions[index], device=device)
            new_log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()

            ratio = torch.exp(new_log_prob - torch.tensor(old_log_probs[index], device=device))
            advantage = advantages_tensor[index]
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_losses.append(-torch.min(surrogate_1, surrogate_2))
            value_losses.append((returns_tensor[index] - value.squeeze(0)) ** 2)
            entropies.append(entropy)

        loss = (
            torch.stack(policy_losses).mean()
            + value_coef * torch.stack(value_losses).mean()
            - ent_coef * torch.stack(entropies).mean()
        )
        loss.backward()
        optimizer.step()
        return float(loss.item())

    os.makedirs(save_dir, exist_ok=True)
    policy = (
        TrailBlazerPolicy.from_checkpoint(resume_checkpoint, cfg=cfg.state_feature_config())
        if resume_checkpoint
        else TrailBlazerPolicy(cfg=cfg.state_feature_config())
    )
    if not resume_checkpoint:
        policy.select_action(EpisodeState(seed_prompt="dummy"))
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    resume_epoch = None
    resumed_optimizer = False
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=policy.device)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            resumed_optimizer = True
        resume_epoch = _checkpoint_epoch(checkpoint, resume_checkpoint)
    env = build_training_env(policy)

    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name="trailblazer_modal_train")
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    run_summary = {
        "epochs": [],
        "save_dir": save_dir,
        "config_path": config_path,
        "seed_prompt_file": seed_prompt_file,
        "reward_backend": selected_reward_backend,
        "use_template_mutator": use_template_mutator,
        "victim_lora_adapter": victim_lora_adapter,
        "attacker_lora_adapter": attacker_lora_adapter,
        "resume_checkpoint": resume_checkpoint,
        "resume_epoch": resume_epoch,
        "resumed_optimizer": resumed_optimizer,
    }
    first_epoch = start_epoch if start_epoch is not None else ((resume_epoch + 1) if resume_epoch is not None else 0)
    for epoch in range(first_epoch, first_epoch + epochs):
        batch = collect_episodes(env, episodes_per_batch, epoch)
        loss_value = ppo_update(policy, batch, optimizer)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        policy.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint.update(
            {
                "epoch": epoch,
                "optimizer_state": optimizer.state_dict(),
                "trainer_state": {
                    "lr": lr,
                    "epochs_requested": epochs,
                    "episodes_per_batch": episodes_per_batch,
                    "max_turns": max_turns,
                    "config_path": config_path,
                    "seed_prompt_file": seed_prompt_file,
                    "seed": seed,
                    "victim_history_turns": victim_history_turns,
                    "reward_backend": selected_reward_backend,
                    "use_template_mutator": use_template_mutator,
                    "attacker_lora_adapter": attacker_lora_adapter,
                    "victim_lora_adapter": victim_lora_adapter,
                    "resume_checkpoint": resume_checkpoint,
                    "resumed_optimizer": resumed_optimizer,
                },
            }
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"epoch={epoch} loss={loss_value:.6f} checkpoint={checkpoint_path}")
        if wandb_run is not None:
            try:
                wandb_run.log({"epoch": epoch, "loss": loss_value, "checkpoint_path": checkpoint_path}, step=epoch)
            except Exception:
                pass
        run_summary["epochs"].append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "checkpoint_path": checkpoint_path,
            }
        )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    volume.commit()
    return run_summary


def _checkpoint_epoch(checkpoint: dict, checkpoint_path: str) -> int | None:
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        return int(epoch)
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


@app.local_entrypoint()
def main(
    save_dir: str = "outputs/policies/trailblazer_ppo",
    remote_save_dir: str = "/root/outputs/policies/trailblazer_ppo",
    epochs: int = 10,
    episodes_per_batch: int = 8,
    max_turns: int = 3,
    lr: float = 3e-4,
    seed: int | None = None,
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    resume_checkpoint: str | None = None,
    start_epoch: int | None = None,
    wandb_project: str | None = None,
) -> None:
    result = train_trailblazer_remote.remote(
        save_dir=remote_save_dir,
        epochs=epochs,
        episodes_per_batch=episodes_per_batch,
        max_turns=max_turns,
        lr=lr,
        config_path="/root/configs/default.json",
        seed_prompt_file="/root/data/seed_prompts.json",
        seed=seed,
        victim_history_turns=victim_history_turns,
        reward_backend=reward_backend,
        use_template_mutator=use_template_mutator,
        attacker_lora_adapter=attacker_lora_adapter,
        victim_lora_adapter=victim_lora_adapter,
        resume_checkpoint=resume_checkpoint,
        start_epoch=start_epoch,
        wandb_project=wandb_project,
    )
    print(json.dumps(result, indent=2))
    print("\nsaved TrailBlazer checkpoints to Modal volume cs224r-redteam-rl-data")
    print(f"remote checkpoint path: {remote_save_dir}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        local_summary_path = Path(save_dir) / "modal_trailblazer_train_summary.json"
        with local_summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")
