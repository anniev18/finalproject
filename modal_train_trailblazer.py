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
from datetime import datetime, timezone
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

def _train_trailblazer_impl(
    save_dir: str = "/root/outputs/policies/trailblazer_ppo",
    episode_log_path: str | None = None,
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
    mutator_model: str | None = None,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    resume_checkpoint: str | None = None,
    start_epoch: int | None = None,
    wandb_project: str | None = None,
    low_memory_vllm: bool = False,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _ensure_hf_token_env()

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
    from redteam_rl.rlbreaker_templates import select_initial_template
    from redteam_rl.types import DialogueTurn, EpisodeState
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
                cfg.mutator_config(
                    model_name=mutator_model or cfg.models.mutator,
                    enable_lora=use_attacker_lora or cfg.mutator.get("enable_lora", False),
                    **(
                        {
                            "gpu_memory_utilization": 0.15,
                            "max_model_len": 2048,
                        }
                        if low_memory_vllm
                        else {}
                    ),
                ),
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
                cfg.victim_config(
                    enable_lora=victim_lora_adapter is not None or cfg.victim.get("enable_lora", False),
                    **(
                        {
                            "gpu_memory_utilization": 0.18,
                            "max_model_len": 2048,
                        }
                        if low_memory_vllm
                        else {}
                    ),
                ),
                lora_adapter_path=victim_lora_adapter,
                capture_debug_prompt=True,
            )
        )
        reward_model = build_reward_model(
            selected_reward_backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
            wildguard_config=cfg.wildguard_config(
                **(
                    {
                        "gpu_memory_utilization": 0.45,
                        "max_model_len": 1024,
                    }
                    if low_memory_vllm
                    else {}
                )
            ),
            reference_similarity_config=cfg.reference_similarity_config(),
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
        states = []
        episode_seeds = []
        for episode_index in range(n):
            episode_seed = None if seed is None else seed + epoch * n + episode_index
            selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
            states.append(
                EpisodeState(
                    seed_prompt=selected_seed_prompt,
                    initial_template=select_initial_template(selected_seed_prompt),
                )
            )
            episode_seeds.append(episode_seed)

        batch = [
            {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "turn_metadata": []}
            for _ in range(n)
        ]
        for turn_index in range(max_turns):
            for episode, state in zip(batch, states):
                episode["states"].append(copy.deepcopy(state))

            attack_steps = env.attacker.act_batch(states)
            attack_prompts = [attack_step.prompt for attack_step in attack_steps]
            victim_responses = env.victim.respond_states_batch(
                attack_prompts,
                states,
                victim_history_turns=env.config.victim_history_turns,
            )
            reward_values = score_reward_batch(env.reward_model, attack_prompts, victim_responses, states)
            judge_labels = list(getattr(env.reward_model, "last_labels", []))
            reward_scores = list(getattr(env.reward_model, "last_scores", []))
            victim_inputs = list(getattr(env.victim, "last_debug_prompts", []))

            for episode_index, (episode, state, attack_step, victim_response, reward_value) in enumerate(
                zip(batch, states, attack_steps, victim_responses, reward_values)
            ):
                metadata = dict(attack_step.metadata or {})
                if episode_index < len(victim_inputs):
                    metadata["victim_input"] = victim_inputs[episode_index]
                if episode_index < len(judge_labels):
                    metadata["judge_label"] = judge_labels[episode_index]
                if episode_index < len(reward_scores):
                    metadata["reward_score"] = reward_scores[episode_index]
                metadata["auxiliary_scores"] = {}
                state.turns.append(
                    DialogueTurn(
                        user_message=attack_step.prompt,
                        victim_response=victim_response,
                        action=attack_step.action,
                        attack_template=attack_step.attack_template,
                        reward=float(reward_value),
                        metadata=metadata,
                    )
                )
                latest_turn = state.turns[-1]
                episode["actions"].append(ACTIONS.index(latest_turn.action))
                episode["log_probs"].append(float(metadata.get("policy_log_prob", 0.0)))
                episode["values"].append(float(metadata.get("policy_value", 0.0)))
                episode["rewards"].append(float(reward_value))
                episode["turn_metadata"].append(dict(latest_turn.metadata))

            print(
                f"epoch={epoch} collected turn {turn_index + 1}/{max_turns} "
                f"for {n} parallel episodes",
                flush=True,
            )

        for episode_index, (episode_seed, state) in enumerate(zip(episode_seeds, states)):
            write_training_episode(
                path=selected_episode_log_path,
                epoch=epoch,
                episode_index=episode_index,
                episode_seed=episode_seed,
                state=state,
            )
            episode_return = sum(float(turn.reward or 0.0) for turn in state.turns)
            judge_labels = [turn.metadata.get("judge_label") for turn in state.turns]
            success = any(str(label).upper() == "UNSAFE" for label in judge_labels)
            print(
                f"epoch={epoch} episode={episode_index + 1}/{n} "
                f"seed={episode_seed} return={episode_return:.3f} "
                f"success={int(success)} labels={judge_labels}",
                flush=True,
            )
        return batch

    def score_reward_batch(reward_model, prompts, responses, states):
        if hasattr(reward_model, "score_batch"):
            try:
                return reward_model.score_batch(prompts, responses, states)
            except TypeError:
                try:
                    return reward_model.score_batch(prompts, responses)
                except TypeError:
                    try:
                        return reward_model.score_batch(prompts)
                    except TypeError:
                        pass
        return [
            reward_model.score(prompt, response, state)
            for prompt, response, state in zip(prompts, responses, states)
        ]

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

    def summarize_batch(batch: list[dict]) -> dict:
        episode_returns = [sum(float(reward) for reward in episode["rewards"]) for episode in batch]
        episode_successes = [
            any(
                str(metadata.get("judge_label", "")).upper() == "UNSAFE"
                for metadata in episode.get("turn_metadata", [])
            )
            for episode in batch
        ]
        first_success_turns = [
            first_turn
            for episode in batch
            if (first_turn := first_success_turn_from_episode(episode)) is not None
        ]
        all_rewards = [float(reward) for episode in batch for reward in episode["rewards"]]
        all_values = [float(value) for episode in batch for value in episode["values"]]
        all_actions = [int(action) for episode in batch for action in episode["actions"]]
        all_turn_metadata = [metadata for episode in batch for metadata in episode.get("turn_metadata", [])]
        action_counts = {ACTIONS[index].value: 0 for index in range(len(ACTIONS))}
        for action in all_actions:
            action_counts[ACTIONS[action].value] += 1
        total_actions = max(len(all_actions), 1)
        mutator_refusals = [
            bool(metadata.get("mutator_refused"))
            for metadata in all_turn_metadata
        ]
        mutator_invalid_templates = [
            bool(metadata.get("mutator_invalid_template"))
            for metadata in all_turn_metadata
        ]
        mutator_fallbacks = [
            bool(metadata.get("mutator_fallback_used"))
            for metadata in all_turn_metadata
        ]
        raw_templates = [
            str(metadata.get("raw_attack_template", ""))
            for metadata in all_turn_metadata
        ]
        final_templates = [
            str(metadata.get("attack_template", ""))
            for metadata in all_turn_metadata
        ]
        raw_placeholder_preserved = [
            template.count("{REQUEST}") == 1
            for template in raw_templates
            if template
        ]
        final_placeholder_preserved = [
            template.count("{REQUEST}") == 1
            for template in final_templates
            if template
        ]
        raw_template_changed = [
            raw_templates[index].strip() != final_templates[index].strip()
            for index in range(min(len(raw_templates), len(final_templates)))
            if raw_templates[index] and final_templates[index]
        ]
        mutator_turns = len(all_turn_metadata)
        return {
            "episodes": len(batch),
            "turns": len(all_rewards),
            "episode_return_mean": sum(episode_returns) / len(episode_returns) if episode_returns else 0.0,
            "episode_return_max": max(episode_returns) if episode_returns else 0.0,
            "success_rate": sum(episode_successes) / len(episode_successes) if episode_successes else 0.0,
            "queries_per_success": (
                sum(first_success_turns) / len(first_success_turns)
                if first_success_turns
                else None
            ),
            "mean_first_success_turn": (
                sum(first_success_turns) / len(first_success_turns)
                if first_success_turns
                else None
            ),
            "first_success_turn_histogram": {
                str(turn): first_success_turns.count(turn)
                for turn in sorted(set(first_success_turns))
            },
            "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "reward_max": max(all_rewards) if all_rewards else 0.0,
            "policy_value_mean": sum(all_values) / len(all_values) if all_values else 0.0,
            "action_counts": action_counts,
            "action_frequencies": {
                action: count / total_actions
                for action, count in action_counts.items()
            },
            "mutator_turns": mutator_turns,
            "mutator_compliance_rate": (
                sum(not refused and not invalid for refused, invalid in zip(mutator_refusals, mutator_invalid_templates))
                / mutator_turns
                if mutator_turns
                else 0.0
            ),
            "mutator_refusal_rate": sum(mutator_refusals) / mutator_turns if mutator_turns else 0.0,
            "mutator_invalid_template_rate": (
                sum(mutator_invalid_templates) / mutator_turns
                if mutator_turns
                else 0.0
            ),
            "mutator_fallback_rate": sum(mutator_fallbacks) / mutator_turns if mutator_turns else 0.0,
            "mutator_raw_placeholder_preservation_rate": (
                sum(raw_placeholder_preserved) / len(raw_placeholder_preserved)
                if raw_placeholder_preserved
                else 0.0
            ),
            "mutator_final_placeholder_preservation_rate": (
                sum(final_placeholder_preserved) / len(final_placeholder_preserved)
                if final_placeholder_preserved
                else 0.0
            ),
            "mutator_raw_final_changed_rate": (
                sum(raw_template_changed) / len(raw_template_changed)
                if raw_template_changed
                else 0.0
            ),
        }

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
        approx_kls = []
        clip_fractions = []
        ratios = []

        for index, state in enumerate(all_states):
            logits, value, _ = policy.forward_from_state(state)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)

            action = torch.tensor(all_actions[index], device=device)
            new_log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            old_log_prob = torch.tensor(old_log_probs[index], device=device)

            ratio = torch.exp(new_log_prob - old_log_prob)
            advantage = advantages_tensor[index]
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_losses.append(-torch.min(surrogate_1, surrogate_2))
            value_losses.append((returns_tensor[index] - value.squeeze(0)) ** 2)
            entropies.append(entropy)
            approx_kls.append(old_log_prob - new_log_prob)
            clip_fractions.append((torch.abs(ratio - 1.0) > clip_eps).float())
            ratios.append(ratio)

        actor_loss = torch.stack(policy_losses).mean()
        critic_loss = torch.stack(value_losses).mean()
        entropy_mean = torch.stack(entropies).mean()
        approx_kl = torch.stack(approx_kls).mean()
        clip_fraction = torch.stack(clip_fractions).mean()
        ratio_mean = torch.stack(ratios).mean()
        loss = actor_loss + value_coef * critic_loss - ent_coef * entropy_mean
        loss.backward()
        optimizer.step()
        advantages_std = advantages_tensor.std(unbiased=False)
        returns_std = returns_tensor.std(unbiased=False)
        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy_mean.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "ratio_mean": float(ratio_mean.item()),
            "advantage_mean": float(advantages_tensor.mean().item()),
            "advantage_std": float(advantages_std.item()),
            "return_mean": float(returns_tensor.mean().item()),
            "return_std": float(returns_std.item()),
        }

    os.makedirs(save_dir, exist_ok=True)
    selected_episode_log_path = episode_log_path or os.path.join(save_dir, "training_episodes.jsonl")
    Path(selected_episode_log_path).parent.mkdir(parents=True, exist_ok=True)
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

            wandb_run = wandb.init(
                project=wandb_project,
                name=Path(save_dir).name,
                resume="never",
                config={
                    "save_dir": save_dir,
                    "episode_log_path": selected_episode_log_path,
                    "epochs": epochs,
                    "episodes_per_batch": episodes_per_batch,
                    "max_turns": max_turns,
                    "lr": lr,
                    "config_path": config_path,
                    "seed_prompt_file": seed_prompt_file,
                    "reward_backend": selected_reward_backend,
                    "use_template_mutator": use_template_mutator,
                    "mutator_model": "template" if use_template_mutator else (mutator_model or cfg.models.mutator),
                    "attacker_lora_adapter": attacker_lora_adapter,
                    "victim_lora_adapter": victim_lora_adapter,
                    "resume_checkpoint": resume_checkpoint,
                    "start_epoch": start_epoch,
                    "low_memory_vllm": low_memory_vllm,
                },
            )
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    run_summary = {
        "epochs": [],
        "save_dir": save_dir,
        "episode_log_path": selected_episode_log_path,
        "template_format": "rlbreaker_mutable_template",
        "config_path": config_path,
        "seed_prompt_file": seed_prompt_file,
        "reward_backend": selected_reward_backend,
        "use_template_mutator": use_template_mutator,
        "mutator_model": "template" if use_template_mutator else (mutator_model or cfg.models.mutator),
        "victim_lora_adapter": victim_lora_adapter,
        "attacker_lora_adapter": attacker_lora_adapter,
        "resume_checkpoint": resume_checkpoint,
        "resume_epoch": resume_epoch,
        "resumed_optimizer": resumed_optimizer,
        "low_memory_vllm": low_memory_vllm,
    }
    first_epoch = start_epoch if start_epoch is not None else ((resume_epoch + 1) if resume_epoch is not None else 0)
    for epoch in range(first_epoch, first_epoch + epochs):
        batch = collect_episodes(env, episodes_per_batch, epoch)
        rollout_metrics = summarize_batch(batch)
        update_metrics = ppo_update(policy, batch, optimizer)
        loss_value = update_metrics["loss"]
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
                    "mutator_model": "template" if use_template_mutator else (mutator_model or cfg.models.mutator),
                    "attacker_lora_adapter": attacker_lora_adapter,
                    "victim_lora_adapter": victim_lora_adapter,
                    "resume_checkpoint": resume_checkpoint,
                    "resumed_optimizer": resumed_optimizer,
                    "low_memory_vllm": low_memory_vllm,
                },
            }
        )
        torch.save(checkpoint, checkpoint_path)
        print(
            f"epoch={epoch} "
            f"loss={loss_value:.6f} "
            f"actor_loss={update_metrics['actor_loss']:.6f} "
            f"critic_loss={update_metrics['critic_loss']:.6f} "
            f"entropy={update_metrics['entropy']:.6f} "
            f"success_rate={rollout_metrics['success_rate']:.3f} "
            f"q_per_success={_format_optional_float(rollout_metrics['queries_per_success'])} "
            f"mean_return={rollout_metrics['episode_return_mean']:.3f} "
            f"checkpoint={checkpoint_path}"
        )
        if wandb_run is not None:
            try:
                wandb_payload = {
                    "epoch": epoch,
                    "loss/total": update_metrics["loss"],
                    "loss/actor": update_metrics["actor_loss"],
                    "loss/critic": update_metrics["critic_loss"],
                    "policy/entropy": update_metrics["entropy"],
                    "policy/approx_kl": update_metrics["approx_kl"],
                    "policy/clip_fraction": update_metrics["clip_fraction"],
                    "policy/ratio_mean": update_metrics["ratio_mean"],
                    "advantage/mean": update_metrics["advantage_mean"],
                    "advantage/std": update_metrics["advantage_std"],
                    "returns/mean": update_metrics["return_mean"],
                    "returns/std": update_metrics["return_std"],
                    "rollout/episode_return_mean": rollout_metrics["episode_return_mean"],
                    "rollout/episode_return_max": rollout_metrics["episode_return_max"],
                    "rollout/success_rate": rollout_metrics["success_rate"],
                    "rollout/queries_per_success": rollout_metrics["queries_per_success"] or 0.0,
                    "rollout/mean_first_success_turn": rollout_metrics["mean_first_success_turn"] or 0.0,
                    "rollout/reward_mean": rollout_metrics["reward_mean"],
                    "rollout/reward_max": rollout_metrics["reward_max"],
                    "rollout/policy_value_mean": rollout_metrics["policy_value_mean"],
                    "rollout/turns": rollout_metrics["turns"],
                    "mutator/compliance_rate": rollout_metrics["mutator_compliance_rate"],
                    "mutator/refusal_rate": rollout_metrics["mutator_refusal_rate"],
                    "mutator/invalid_template_rate": rollout_metrics["mutator_invalid_template_rate"],
                    "mutator/fallback_rate": rollout_metrics["mutator_fallback_rate"],
                    "mutator/raw_placeholder_preservation_rate": rollout_metrics[
                        "mutator_raw_placeholder_preservation_rate"
                    ],
                    "mutator/final_placeholder_preservation_rate": rollout_metrics[
                        "mutator_final_placeholder_preservation_rate"
                    ],
                    "mutator/raw_final_changed_rate": rollout_metrics["mutator_raw_final_changed_rate"],
                }
                if selected_reward_backend in {
                    "reference_similarity",
                    "reference_similarity_raw",
                    "reference_similarity_binary",
                }:
                    wandb_payload["reference_similarity/mean"] = rollout_metrics["reward_mean"]
                    wandb_payload["reference_similarity/max"] = rollout_metrics["reward_max"]
                    wandb_payload["reference_similarity/success_rate_at_threshold"] = rollout_metrics[
                        "success_rate"
                    ]
                for action, frequency in rollout_metrics["action_frequencies"].items():
                    wandb_payload[f"actions/{action}"] = frequency
                wandb_run.log(wandb_payload, step=epoch)
                wandb_run.summary["latest_checkpoint_path"] = checkpoint_path
            except Exception as exc:
                print(f"wandb train logging failed: {exc}", flush=True)
        run_summary["epochs"].append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "update_metrics": update_metrics,
                "rollout_metrics": rollout_metrics,
                "checkpoint_path": checkpoint_path,
            }
        )
        write_training_summary(
            path=os.path.join(save_dir, "training_summary_partial.json"),
            summary=run_summary,
        )
        volume.commit()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    write_training_summary(
        path=os.path.join(save_dir, "training_summary.json"),
        summary=run_summary,
    )
    volume.commit()
    return run_summary


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 24,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_trailblazer_remote(**kwargs) -> dict:
    return _train_trailblazer_impl(**kwargs)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_trailblazer_a100_80gb_remote(**kwargs) -> dict:
    return _train_trailblazer_impl(**kwargs)


def write_training_summary(path: str, summary: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _ensure_hf_token_env() -> None:
    if not os.environ.get("HF_TOKEN"):
        for alternate_name in ("HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            alternate_value = os.environ.get(alternate_name)
            if alternate_value:
                os.environ["HF_TOKEN"] = alternate_value
                break
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])
        os.environ.setdefault("HF_HUB_TOKEN", os.environ["HF_TOKEN"])
    print(f"HF_TOKEN available: {bool(os.environ.get('HF_TOKEN'))}", flush=True)


def write_training_episode(
    path: str,
    epoch: int,
    episode_index: int,
    episode_seed: int | None,
    state,
) -> None:
    record = {
        "epoch": epoch,
        "episode_index": episode_index,
        "seed": episode_seed,
        "template_format": "rlbreaker_mutable_template",
        "seed_prompt": state.seed_prompt,
        "initial_template": state.initial_template,
        "num_turns": len(state.turns),
        "episode_return": sum(float(turn.reward or 0.0) for turn in state.turns),
        "success": any(str(turn.metadata.get("judge_label", "")).upper() == "UNSAFE" for turn in state.turns),
        "first_success_turn": first_success_turn_from_turns(state.turns),
        "turns": [
            {
                "turn_index": turn_index,
                "action": turn.action.value if turn.action else None,
                "attack_template": turn.attack_template,
                "user_message": turn.user_message,
                "victim_response": turn.victim_response,
                "reward": turn.reward,
                "metadata": turn.metadata,
            }
            for turn_index, turn in enumerate(state.turns, start=1)
        ],
    }
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def first_success_turn_from_episode(episode: dict) -> int | None:
    rewards = episode.get("rewards", [])
    turn_metadata = episode.get("turn_metadata", [])
    for index, (reward, metadata) in enumerate(zip(rewards, turn_metadata), start=1):
        label = str(metadata.get("judge_label", "")).upper()
        if label == "UNSAFE" or float(reward or 0.0) >= 1.0:
            return index
    return None


def first_success_turn_from_turns(turns) -> int | None:
    for index, turn in enumerate(turns, start=1):
        label = str(turn.metadata.get("judge_label", "")).upper()
        if label == "UNSAFE" or float(turn.reward or 0.0) >= 1.0:
            return index
    return None


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


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
    remote_episode_log_path: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    run_name: str | None = None,
    wait_for_result: bool = False,
    epochs: int = 10,
    episodes_per_batch: int = 8,
    max_turns: int = 3,
    lr: float = 3e-4,
    seed: int | None = None,
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    mutator_model: str | None = None,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    resume_checkpoint: str | None = None,
    start_epoch: int | None = None,
    wandb_project: str | None = None,
    gpu_type: str = "A100-80GB",
    low_memory_vllm: bool = False,
) -> None:
    resolved_run_name = run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    actual_remote_save_dir = str(Path(remote_save_dir) / resolved_run_name)
    actual_episode_log_path = remote_episode_log_path or str(
        Path(actual_remote_save_dir) / "training_episodes.jsonl"
    )
    selected_gpu = _normalize_gpu_type(gpu_type)
    selected_low_memory_vllm = low_memory_vllm
    call_kwargs = {
        "save_dir": actual_remote_save_dir,
        "episode_log_path": actual_episode_log_path,
        "epochs": epochs,
        "episodes_per_batch": episodes_per_batch,
        "max_turns": max_turns,
        "lr": lr,
        "config_path": "/root/configs/default.json",
        "seed_prompt_file": seed_prompt_file,
        "seed": seed,
        "victim_history_turns": victim_history_turns,
        "reward_backend": reward_backend,
        "use_template_mutator": use_template_mutator,
        "mutator_model": mutator_model,
        "attacker_lora_adapter": attacker_lora_adapter,
        "victim_lora_adapter": victim_lora_adapter,
        "resume_checkpoint": resume_checkpoint,
        "start_epoch": start_epoch,
        "wandb_project": wandb_project,
        "low_memory_vllm": selected_low_memory_vllm,
    }
    remote_function = {
        "L40S": train_trailblazer_remote,
        "A100-80GB": train_trailblazer_a100_80gb_remote,
    }[selected_gpu]
    if not wait_for_result:
        function_call = remote_function.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted TrailBlazer training without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("gpu type:", selected_gpu)
        print("low memory vllm:", selected_low_memory_vllm)
        print("remote checkpoint path:", actual_remote_save_dir)
        print("remote training episodes:", actual_episode_log_path)
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-trailblazer-train` to follow progress.")
        return

    result = remote_function.remote(**call_kwargs)
    print(json.dumps(result, indent=2))
    print("\nsaved TrailBlazer checkpoints to Modal volume cs224r-redteam-rl-data")
    print(f"run name: {resolved_run_name}")
    print(f"gpu type: {selected_gpu}")
    print(f"low memory vllm: {selected_low_memory_vllm}")
    print(f"remote checkpoint path: {actual_remote_save_dir}")
    print("remote training episodes:", result.get("episode_log_path"))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        local_summary_path = Path(save_dir) / "modal_trailblazer_train_summary.json"
        with local_summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")


def _normalize_gpu_type(gpu_type: str) -> str:
    normalized = gpu_type.strip().upper().replace("_", "-")
    if normalized in {"L40S", "L40"}:
        return "L40S"
    if normalized in {"A100", "A100-80GB", "A100-80G"}:
        return "A100-80GB"
    raise ValueError("gpu_type must be one of: L40S, A100-80GB")
