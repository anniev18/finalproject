"""Full Modal orchestration for PPO/attacker/victim co-evolution.

This script keeps staged execution isolated by Modal function, and adds an
optional attacker LoRA update stage between PPO and victim updates. Use flags to run
ablations with PPO only, PPO+victim, or full PPO+attacker+victim co-evolution.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-coevolution-full")
volume = modal.Volume.from_name("cs224r-redteam-rl-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-gpu.txt")
    .env({
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("redteam_rl", remote_path="/root/redteam_rl")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_policy_cycle_remote(
    run_dir: str,
    cycle_index: int,
    policy_checkpoint: str,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    start_epoch: int | None = None,
    ppo_epochs: int = 5,
    episodes_per_batch: int = 16,
    rollout_parallelism: int = 5,
    max_turns: int = 3,
    lr: float = 3e-4,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    config_path: str = "/root/configs/default.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    wandb_project: str | None = None,
    launch_command: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _ensure_hf_token_env()
    import sys

    sys.path.insert(0, "/root")
    _print_launch_context(
        "PPO policy cycle",
        run_dir=run_dir,
        launch_command=launch_command,
        extra={
            "cycle_index": cycle_index,
            "policy_checkpoint": policy_checkpoint,
            "attacker_lora_adapter": attacker_lora_adapter,
            "victim_lora_adapter": victim_lora_adapter,
        },
    )

    import torch
    import torch.optim as optim

    from redteam_rl.actions import ACTIONS
    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.orchestration import annotate_episode_state
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.rlbreaker_templates import select_initial_template
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.trajectory_bank import append_episodes
    from redteam_rl.types import DialogueTurn, EpisodeState
    from redteam_rl.versioning import build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    selected_reward_backend = reward_backend or cfg.reward_backend()
    run_path = Path(run_dir)
    cycle_dir = run_path / f"cycle_{cycle_index:03d}"
    policy_dir = cycle_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = cycle_dir / "training_episodes.jsonl"
    trajectory_bank_path = run_path / "trajectory_bank" / "episodes.jsonl"
    run_id = new_run_id(f"full_coevolve_cycle_{cycle_index:03d}")

    policy = TrailBlazerPolicy.from_checkpoint(policy_checkpoint, cfg=cfg.state_feature_config())
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    checkpoint = torch.load(policy_checkpoint, map_location=policy.device)
    if checkpoint.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError:
            pass
    inferred_epoch = _checkpoint_epoch(checkpoint, policy_checkpoint)
    first_epoch = start_epoch if start_epoch is not None else ((inferred_epoch + 1) if inferred_epoch is not None else 0)
    requested_final_epoch = first_epoch + ppo_epochs - 1

    config_initial_attacker_adapter = cfg.attacker_evolution.get("initial_adapter_path")
    requested_attacker_adapter = attacker_lora_adapter or config_initial_attacker_adapter
    active_attacker_adapter = _valid_lora_adapter_path(requested_attacker_adapter)
    if requested_attacker_adapter and not active_attacker_adapter:
        print(
            f"warning: ignoring invalid attacker adapter path for PPO cycle {cycle_index}: "
            f"{requested_attacker_adapter}",
            flush=True,
        )
    use_attacker_lora = active_attacker_adapter is not None
    mutator = (
        TemplateMutator()
        if use_template_mutator
        else LLMMutator(
            cfg.mutator_config(enable_lora=use_attacker_lora or cfg.mutator.get("enable_lora", False)),
            capture_debug_prompt=True,
            lora_adapter_path=active_attacker_adapter,
        )
    )
    attacker = (
        EvolvingAttacker(
            policy=policy,
            mutator=mutator,
            config=cfg.attacker_evolution_config(initial_adapter_path=active_attacker_adapter),
        )
        if cfg.attacker_evolution.get("enabled", False) or active_attacker_adapter
        else Attacker(policy=policy, mutator=mutator)
    )
    active_victim_adapter = _valid_lora_adapter_path(victim_lora_adapter)
    if victim_lora_adapter and not active_victim_adapter:
        print(
            f"warning: ignoring invalid victim adapter path for PPO cycle {cycle_index}: "
            f"{victim_lora_adapter}",
            flush=True,
        )

    victim = EvolvingVictim(
        VLLMVictim(
            cfg.victim_config(enable_lora=active_victim_adapter is not None or cfg.victim.get("enable_lora", False)),
            lora_adapter_path=active_victim_adapter,
            capture_debug_prompt=True,
        )
    )
    reward_model = build_reward_model(
        selected_reward_backend,
        prompt_guard_config=cfg.prompt_guard_config(),
        qwen_judge_config=cfg.qwen_judge_config(),
        llama_guard_config=cfg.llama_guard_config(),
        wildguard_config=cfg.wildguard_config(),
        reference_similarity_config=cfg.reference_similarity_config(),
    )
    env = RedTeamEnv(
        config=cfg.env_config(
            max_turns=max_turns,
            **({"victim_history_turns": victim_history_turns} if victim_history_turns is not None else {}),
        ),
        attacker=attacker,
        victim=victim,
        reward_model=reward_model,
    )

    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=f"{Path(run_dir).name}_cycle_{cycle_index:03d}_ppo",
                resume="never",
                config={
                    "run_dir": run_dir,
                    "cycle_index": cycle_index,
                    "policy_checkpoint": policy_checkpoint,
                    "requested_victim_lora_adapter": victim_lora_adapter,
                    "active_victim_lora_adapter": active_victim_adapter,
                    "ppo_epochs": ppo_epochs,
                    "episodes_per_batch": episodes_per_batch,
                    "rollout_parallelism": rollout_parallelism,
                    "max_turns": max_turns,
                    "reward_backend": selected_reward_backend,
                },
            )
            wandb.define_metric("epoch")
            wandb.define_metric("loss/*", step_metric="epoch")
            wandb.define_metric("policy/*", step_metric="epoch")
            wandb.define_metric("advantage/*", step_metric="epoch")
            wandb.define_metric("returns/*", step_metric="epoch")
            wandb.define_metric("rollout/*", step_metric="epoch")
            wandb.define_metric("mutator/*", step_metric="epoch")
            wandb.define_metric("actions/*", step_metric="epoch")
            wandb.define_metric("coev/*", step_metric="epoch")
        except Exception as exc:
            print(f"wandb init failed: {exc}", flush=True)

    all_cycle_episodes = []
    epoch_summaries = []
    rewards = []
    existing_policy_summary = _load_policy_summary(cycle_dir / "policy_summary.json")
    completed_epoch_summaries = [
        summary
        for summary in existing_policy_summary
        if first_epoch <= int(summary.get("epoch", -1)) <= requested_final_epoch
        and Path(str(summary.get("checkpoint_path", ""))).exists()
    ]
    if completed_epoch_summaries:
        completed_epoch_summaries.sort(key=lambda row: int(row.get("epoch", -1)))
        contiguous_epoch_summaries = []
        expected_epoch = first_epoch
        for summary in completed_epoch_summaries:
            summary_epoch = int(summary.get("epoch", -1))
            if summary_epoch != expected_epoch:
                break
            contiguous_epoch_summaries.append(summary)
            expected_epoch += 1
        completed_epoch_summaries = contiguous_epoch_summaries
    if completed_epoch_summaries:
        latest_completed = completed_epoch_summaries[-1]
        latest_completed_epoch = int(latest_completed["epoch"])
        latest_checkpoint_path = str(latest_completed["checkpoint_path"])
        if latest_completed_epoch >= requested_final_epoch:
            print(
                f"cycle={cycle_index} policy stage already has completed epochs "
                f"{first_epoch}-{requested_final_epoch}; reusing {latest_checkpoint_path}",
                flush=True,
            )
            result = _finalize_policy_stage_summary(
                cycle_dir=cycle_dir,
                cycle_index=cycle_index,
                policy_checkpoint=policy_checkpoint,
                active_attacker_adapter=active_attacker_adapter,
                active_victim_adapter=active_victim_adapter,
                first_epoch=first_epoch,
                requested_final_epoch=requested_final_epoch,
                policy_dir=policy_dir,
                final_checkpoint=latest_checkpoint_path,
                episode_log_path=episode_log_path,
                trajectory_bank_path=trajectory_bank_path,
                epoch_summaries=completed_epoch_summaries,
                all_cycle_episodes=[],
                rewards=[],
            )
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            return result
        print(
            f"cycle={cycle_index} resuming policy stage from completed epoch "
            f"{latest_completed_epoch}; loading {latest_checkpoint_path}",
            flush=True,
        )
        policy = TrailBlazerPolicy.from_checkpoint(latest_checkpoint_path, cfg=cfg.state_feature_config())
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=policy.device)
        if latest_checkpoint.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(latest_checkpoint["optimizer_state"])
            except ValueError:
                pass
        epoch_summaries.extend(completed_epoch_summaries)
        first_epoch_to_run = latest_completed_epoch + 1
    else:
        first_epoch_to_run = first_epoch
    effective_rollout_parallelism = max(1, min(int(rollout_parallelism), int(episodes_per_batch)))

    def score_reward_batch(prompts, responses, states):
        if hasattr(env.reward_model, "score_batch"):
            try:
                return env.reward_model.score_batch(prompts, responses, states)
            except TypeError:
                try:
                    return env.reward_model.score_batch(prompts, responses)
                except TypeError:
                    try:
                        return env.reward_model.score_batch(prompts)
                    except TypeError:
                        pass
        return [
            env.reward_model.score(prompt, response, state)
            for prompt, response, state in zip(prompts, responses, states)
        ]

    def collect_episode_chunk(epoch: int, chunk_start: int, chunk_size: int) -> tuple[list[dict], list]:
        states = []
        episode_seeds = []
        for offset in range(chunk_size):
            episode_index = chunk_start + offset
            episode_seed = seed + epoch * episodes_per_batch + episode_index
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
            for _ in range(chunk_size)
        ]
        print(
            f"cycle={cycle_index} epoch={epoch} collecting episodes "
            f"{chunk_start + 1}-{chunk_start + chunk_size}/{episodes_per_batch} "
            f"with rollout_parallelism={chunk_size}",
            flush=True,
        )
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
            reward_values = score_reward_batch(attack_prompts, victim_responses, states)
            judge_labels = list(getattr(env.reward_model, "last_labels", []))
            reward_scores = list(getattr(env.reward_model, "last_scores", []))
            victim_inputs = list(getattr(env.victim, "last_debug_prompts", []))

            for local_index, (episode, state, attack_step, victim_response, reward_value) in enumerate(
                zip(batch, states, attack_steps, victim_responses, reward_values)
            ):
                metadata = dict(attack_step.metadata or {})
                if local_index < len(victim_inputs):
                    metadata["victim_input"] = victim_inputs[local_index]
                if local_index < len(judge_labels):
                    metadata["judge_label"] = judge_labels[local_index]
                if local_index < len(reward_scores):
                    metadata["reward_score"] = reward_scores[local_index]
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

            unsafe_count = sum(1 for label in judge_labels if str(label).upper() == "UNSAFE")
            print(
                f"cycle={cycle_index} epoch={epoch} collected turn {turn_index + 1}/{max_turns} "
                f"for episodes {chunk_start + 1}-{chunk_start + chunk_size}/{episodes_per_batch} "
                f"unsafe={unsafe_count}/{chunk_size}",
                flush=True,
            )

        for local_index, (episode, episode_seed, state) in enumerate(zip(batch, episode_seeds, states)):
            global_episode_index = chunk_start + local_index
            metadata = build_model_metadata(
                run_id=run_id,
                victim_model=cfg.models.victim,
                victim_adapter_path=active_victim_adapter,
                attacker_model="template" if use_template_mutator else cfg.models.mutator,
                attacker_adapter_path=active_attacker_adapter,
                judge_model=_judge_model_name(cfg, selected_reward_backend),
                policy_type="trailblazer_coevolution_full",
                policy_checkpoint=policy_checkpoint,
                extra={
                    "cycle_index": cycle_index,
                    "epoch": epoch,
                    "reward_backend": selected_reward_backend,
                    "rollout_parallelism": effective_rollout_parallelism,
                },
            )
            annotate_episode_state(state, metadata)
            episode["final_state"] = copy.deepcopy(state)
            _append_training_episode(
                episode_log_path,
                cycle_index=cycle_index,
                epoch=epoch,
                episode_index=global_episode_index,
                episode_seed=episode_seed,
                state=state,
            )
            episode_return = sum(float(turn.reward or 0.0) for turn in state.turns)
            labels = [turn.metadata.get("judge_label") for turn in state.turns]
            first_success_turn = _first_success_turn_from_turns(state.turns)
            print(
                f"cycle={cycle_index} epoch={epoch} episode={global_episode_index + 1}/{episodes_per_batch} done "
                f"return={episode_return:.3f} "
                f"success={int(first_success_turn is not None)} "
                f"first_success_turn={first_success_turn or 'n/a'} "
                f"labels={labels}",
                flush=True,
            )
        return batch, [copy.deepcopy(state) for state in states]

    for epoch in range(first_epoch_to_run, requested_final_epoch + 1):
        batch = []
        epoch_episodes = []
        completed_episode_records = _load_completed_policy_episode_records(
            episode_log_path,
            cycle_index=cycle_index,
            epoch=epoch,
            max_turns=max_turns,
        )
        resume_episode_start = 0
        while resume_episode_start in completed_episode_records:
            resume_episode_start += 1
        if resume_episode_start > 0:
            print(
                f"cycle={cycle_index} epoch={epoch} reusing completed rollout episodes "
                f"1-{resume_episode_start}/{episodes_per_batch}",
                flush=True,
            )
            for episode_index in range(min(resume_episode_start, episodes_per_batch)):
                episode, state = _policy_episode_from_record(completed_episode_records[episode_index])
                batch.append(episode)
                epoch_episodes.append(state)

        for chunk_start in range(resume_episode_start, episodes_per_batch, effective_rollout_parallelism):
            chunk_size = min(effective_rollout_parallelism, episodes_per_batch - chunk_start)
            chunk_batch, chunk_episodes = collect_episode_chunk(epoch, chunk_start, chunk_size)
            batch.extend(chunk_batch)
            epoch_episodes.extend(chunk_episodes)

        rollout_metrics = _summarize_policy_batch(batch)
        update_metrics = _ppo_update(policy, batch, optimizer)
        loss_value = update_metrics["loss"]
        checkpoint_path = policy_dir / f"checkpoint_epoch_{epoch}.pt"
        _save_policy_checkpoint(
            policy=policy,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            trainer_state={
                "cycle_index": cycle_index,
                "policy_checkpoint": policy_checkpoint,
                "attacker_lora_adapter": active_attacker_adapter,
                "victim_lora_adapter": active_victim_adapter,
                "lr": lr,
                "episodes_per_batch": episodes_per_batch,
                "max_turns": max_turns,
                "seed_prompt_file": seed_prompt_file,
                "reward_backend": selected_reward_backend,
            },
        )
        append_episodes(
            epoch_episodes,
            path=trajectory_bank_path,
            metadata={
                "source": "modal_coevolve_full.py",
                "run_id": run_id,
                "cycle_index": cycle_index,
                "epoch": epoch,
                "victim_adapter_path": active_victim_adapter,
            },
        )
        all_cycle_episodes.extend(epoch_episodes)
        for state in epoch_episodes:
            rewards.extend(float(turn.reward or 0.0) for turn in state.turns)
        epoch_first_success_turns = [
            first_turn
            for state in epoch_episodes
            if (first_turn := _first_success_turn_from_turns(state.turns)) is not None
        ]
        cumulative_first_success_turns = [
            first_turn
            for state in all_cycle_episodes
            if (first_turn := _first_success_turn_from_turns(state.turns)) is not None
        ]
        epoch_summary = {
            "epoch": epoch,
            "loss": loss_value,
            "update_metrics": update_metrics,
            "rollout_metrics": rollout_metrics,
            "checkpoint_path": str(checkpoint_path),
            "episodes": len(epoch_episodes),
            "success_rate": len(epoch_first_success_turns) / len(epoch_episodes) if epoch_episodes else 0.0,
            "queries_per_success": (
                sum(epoch_first_success_turns) / len(epoch_first_success_turns)
                if epoch_first_success_turns
                else None
            ),
            "mean_first_success_turn": (
                sum(epoch_first_success_turns) / len(epoch_first_success_turns)
                if epoch_first_success_turns
                else None
            ),
            "cumulative_queries_per_success": (
                sum(cumulative_first_success_turns) / len(cumulative_first_success_turns)
                if cumulative_first_success_turns
                else None
            ),
            "average_reward_so_far": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward_so_far": max(rewards) if rewards else 0.0,
        }
        epoch_summaries.append(epoch_summary)
        _write_json(cycle_dir / "policy_summary.json", {"epochs": epoch_summaries})
        volume.commit()
        print(
            f"cycle={cycle_index} epoch={epoch} loss={loss_value:.6f} "
            f"actor_loss={update_metrics['actor_loss']:.6f} "
            f"critic_loss={update_metrics['critic_loss']:.6f} "
            f"entropy={update_metrics['entropy']:.6f} "
            f"success_rate={epoch_summary['success_rate']:.3f} "
            f"q_per_success={_format_optional_float(epoch_summary['queries_per_success'])} "
            f"mean_return={rollout_metrics['episode_return_mean']:.3f} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
        if wandb_run is not None:
            try:
                wandb_payload = {
                    "cycle": cycle_index,
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
                    "coev/average_reward_so_far": epoch_summary["average_reward_so_far"],
                    "coev/max_reward_so_far": epoch_summary["max_reward_so_far"],
                }
                for action, frequency in rollout_metrics["action_frequencies"].items():
                    wandb_payload[f"actions/{action}"] = frequency
                wandb_run.log(wandb_payload, step=epoch)
            except Exception:
                pass

    final_checkpoint = str(policy_dir / f"checkpoint_epoch_{requested_final_epoch}.pt")
    result = _finalize_policy_stage_summary(
        cycle_dir=cycle_dir,
        cycle_index=cycle_index,
        policy_checkpoint=policy_checkpoint,
        active_attacker_adapter=active_attacker_adapter,
        active_victim_adapter=active_victim_adapter,
        first_epoch=first_epoch,
        requested_final_epoch=requested_final_epoch,
        policy_dir=policy_dir,
        final_checkpoint=final_checkpoint,
        episode_log_path=episode_log_path,
        trajectory_bank_path=trajectory_bank_path,
        epoch_summaries=epoch_summaries,
        all_cycle_episodes=all_cycle_episodes,
        rewards=rewards,
    )
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
)
def train_attacker_cycle_remote(
    run_dir: str,
    cycle_index: int,
    base_adapter_path: str | None = None,
    config_path: str = "/root/configs/default.json",
    min_reward_for_training: float = 1.0,
    attacker_example_source: str = "filtered_template",
    attacker_training_source: str = "cycle_only",
    max_training_episodes: int | None = None,
    dry_run_updates: bool = False,
    verify_adapter_update: bool = False,
    attacker_train_steps: int | None = None,
    attacker_train_batch_size: int = 8,
    attacker_train_grad_accum_steps: int = 4,
    attacker_train_max_length: int = 2048,
    wandb_project: str | None = None,
    launch_command: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _ensure_hf_token_env()
    import sys

    sys.path.insert(0, "/root")
    _print_launch_context(
        "attacker update cycle",
        run_dir=run_dir,
        launch_command=launch_command,
        extra={
            "cycle_index": cycle_index,
            "base_adapter_path": base_adapter_path,
            "attacker_example_source": attacker_example_source,
            "attacker_training_source": attacker_training_source,
            "dry_run_updates": dry_run_updates,
        },
    )

    from redteam_rl.attacker_training import (
        attacker_examples_from_episodes,
        create_dry_run_attacker_adapter,
        fine_tune_attacker_lora,
    )
    from redteam_rl.attacker_training_data import (
        FilteredAttackerExampleConfig,
        filtered_template_examples_from_episodes,
    )
    from redteam_rl.config import load_config
    from redteam_rl.orchestration import verify_adapter_update as verify_adapter_artifact
    from redteam_rl.trajectory_bank import load_episodes

    if attacker_example_source not in {"legacy_message", "filtered_template"}:
        raise ValueError("attacker_example_source must be legacy_message or filtered_template")
    if attacker_training_source not in {"cycle_only", "cumulative"}:
        raise ValueError("attacker_training_source must be cycle_only or cumulative")

    cfg = load_config(config_path)
    run_path = Path(run_dir)
    cycle_dir = run_path / f"cycle_{cycle_index:03d}"
    trajectory_bank_path = run_path / "trajectory_bank" / "episodes.jsonl"
    episodes = (
        _load_cycle_training_episodes(cycle_dir / "training_episodes.jsonl")
        if attacker_training_source == "cycle_only"
        else load_episodes(trajectory_bank_path, limit=max_training_episodes)
    )
    if attacker_training_source == "cycle_only" and max_training_episodes is not None and max_training_episodes > 0:
        episodes = episodes[-max_training_episodes:]

    filter_stats = None
    if attacker_example_source == "filtered_template":
        filtered_result = filtered_template_examples_from_episodes(
            episodes,
            config=FilteredAttackerExampleConfig(min_reward=min_reward_for_training),
        )
        examples = filtered_result.examples
        filter_stats = filtered_result.stats
    else:
        examples = attacker_examples_from_episodes(
            episodes,
            min_reward=min_reward_for_training,
        )

    _write_json(cycle_dir / "attacker_training_examples.json", [asdict(example) for example in examples])

    adapter_path = None
    adapter_valid = False
    verification = None
    active_adapter_path = base_adapter_path
    update_status = "no_examples"
    update_reason = "no attacker training examples passed the filter"
    if examples:
        if dry_run_updates:
            adapter_path = str(create_dry_run_attacker_adapter(cycle_dir, cycle_index, examples))
            active_adapter_path = base_adapter_path
            update_status = "dry_run"
            update_reason = "dry run wrote metadata only; active attacker adapter unchanged"
        else:
            training_overrides = {
                "base_adapter_path": base_adapter_path,
                "batch_size": attacker_train_batch_size,
                "grad_accum_steps": attacker_train_grad_accum_steps,
                "max_length": attacker_train_max_length,
            }
            if attacker_train_steps is not None:
                training_overrides["train_steps"] = attacker_train_steps
            attacker_training_config = cfg.attacker_finetune_config(
                output_dir=cycle_dir / "attacker_adapter",
                **training_overrides,
            )
            adapter_path = str(fine_tune_attacker_lora(attacker_training_config, examples))
            valid_adapter_path = _valid_lora_adapter_path(adapter_path)
            adapter_valid = valid_adapter_path is not None
            active_adapter_path = valid_adapter_path or base_adapter_path
            update_status = "updated" if adapter_valid else "invalid_adapter"
            update_reason = (
                f"created attacker adapter for cycle {cycle_index}"
                if adapter_valid
                else "attacker fine-tuning ran but did not produce a valid LoRA adapter"
            )
            if not adapter_valid:
                print(
                    f"warning: attacker update for cycle {cycle_index} produced an invalid adapter: "
                    f"{adapter_path}; keeping previous active adapter.",
                    flush=True,
                )
        if verify_adapter_update and not dry_run_updates:
            verification = verify_adapter_artifact(adapter_path)

    result = {
        "cycle_index": cycle_index,
        "stage": "attacker",
        "input_base_adapter_path": base_adapter_path,
        "trajectory_bank_path": str(trajectory_bank_path),
        "training_source": attacker_training_source,
        "example_source": attacker_example_source,
        "episodes_used": len(episodes),
        "turns_used": sum(len(episode.turns) for episode in episodes),
        "training_examples": len(examples),
        "filter_stats": filter_stats,
        "base_adapter_path": base_adapter_path,
        "adapter_path": adapter_path,
        "adapter_valid": adapter_valid,
        "active_adapter_path": active_adapter_path,
        "input_attacker_version": _adapter_version_label(base_adapter_path, "attacker"),
        "output_attacker_version": (
            f"attacker_cycle_{cycle_index:03d}" if adapter_valid else _adapter_version_label(base_adapter_path, "attacker")
        ),
        "update_status": update_status,
        "update_reason": update_reason,
        "adapter_verification": verification,
        "dry_run_updates": dry_run_updates,
        "attacker_training_overrides": {
            "train_steps": attacker_train_steps,
            "batch_size": attacker_train_batch_size,
            "grad_accum_steps": attacker_train_grad_accum_steps,
            "max_length": attacker_train_max_length,
        },
    }
    _write_json(cycle_dir / "attacker_stage_summary.json", result)
    _log_stage_wandb(
        wandb_project=wandb_project,
        run_name=f"{Path(run_dir).name}_cycle_{cycle_index:03d}_attacker",
        metrics={
            "cycle": cycle_index,
            "attacker/training_examples": len(examples),
            "attacker/episodes_used": len(episodes),
            "attacker/turns_used": sum(len(episode.turns) for episode in episodes),
            "attacker/adapter_valid": float(adapter_valid),
            "attacker/updated": float(update_status == "updated"),
            "attacker/no_examples": float(update_status == "no_examples"),
            **{
                f"attacker/filter/{key}": value
                for key, value in (filter_stats or {}).items()
                if isinstance(value, (int, float))
            },
        },
        summary=result,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
)
def train_victim_cycle_remote(
    run_dir: str,
    cycle_index: int,
    base_adapter_path: str | None = None,
    config_path: str = "/root/configs/default.json",
    min_reward_for_training: float | None = None,
    victim_training_source: str = "cumulative",
    max_training_episodes: int | None = None,
    dry_run_updates: bool = False,
    verify_adapter_update: bool = False,
    victim_train_steps: int | None = None,
    victim_train_batch_size: int = 1,
    victim_train_grad_accum_steps: int = 8,
    victim_train_max_length: int = 768,
    wandb_project: str | None = None,
    launch_command: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _ensure_hf_token_env()
    import sys

    sys.path.insert(0, "/root")
    _print_launch_context(
        "victim update cycle",
        run_dir=run_dir,
        launch_command=launch_command,
        extra={
            "cycle_index": cycle_index,
            "base_adapter_path": base_adapter_path,
            "victim_training_source": victim_training_source,
            "dry_run_updates": dry_run_updates,
        },
    )

    from redteam_rl.config import load_config
    from redteam_rl.orchestration import (
        VictimEvolutionConfig,
        build_safety_examples,
        create_dry_run_adapter,
        verify_adapter_update as verify_adapter_artifact,
    )
    from redteam_rl.trajectory_bank import load_episodes
    from redteam_rl.victim_training import train_victim_from_episodes

    cfg = load_config(config_path)
    run_path = Path(run_dir)
    cycle_dir = run_path / f"cycle_{cycle_index:03d}"
    trajectory_bank_path = run_path / "trajectory_bank" / "episodes.jsonl"
    if victim_training_source not in {"cycle_only", "cumulative"}:
        raise ValueError("victim_training_source must be cycle_only or cumulative")
    episodes = (
        _load_cycle_training_episodes(cycle_dir / "training_episodes.jsonl")
        if victim_training_source == "cycle_only"
        else load_episodes(trajectory_bank_path, limit=max_training_episodes)
    )
    if victim_training_source == "cycle_only" and max_training_episodes is not None and max_training_episodes > 0:
        episodes = episodes[-max_training_episodes:]
    selected_min_reward = (
        min_reward_for_training
        if min_reward_for_training is not None
        else cfg.victim_evolution_config().min_reward_for_training
    )
    evolution_config = VictimEvolutionConfig(
        min_reward_for_training=selected_min_reward,
        safety_target_file="/root/data/safety_target_templates.json",
        trajectory_bank_path=str(trajectory_bank_path),
        verify_adapter_update=verify_adapter_update,
    )
    examples = build_safety_examples(episodes, evolution_config)
    _write_json(cycle_dir / "victim_training_examples.json", [asdict(example) for example in examples])

    adapter_path = None
    adapter_valid = False
    verification = None
    active_adapter_path = base_adapter_path
    update_status = "no_examples"
    update_reason = "no victim safety training examples were found"
    if examples:
        if dry_run_updates:
            adapter_path = str(create_dry_run_adapter(cycle_dir, cycle_index, examples))
            active_adapter_path = base_adapter_path
            update_status = "dry_run"
            update_reason = "dry run wrote metadata only; active victim adapter unchanged"
        else:
            training_overrides = {
                "base_adapter_path": base_adapter_path,
                "batch_size": victim_train_batch_size,
                "grad_accum_steps": victim_train_grad_accum_steps,
                "max_length": victim_train_max_length,
            }
            if victim_train_steps is not None:
                training_overrides["train_steps"] = victim_train_steps
            victim_training_config = cfg.victim_finetune_config(
                output_dir=cycle_dir / "victim_adapter",
                **training_overrides,
            )
            adapter_path = str(
                train_victim_from_episodes(
                    episodes,
                    victim_training_config,
                    min_reward=selected_min_reward,
                )
            )
            valid_adapter_path = _valid_lora_adapter_path(adapter_path)
            adapter_valid = valid_adapter_path is not None
            active_adapter_path = valid_adapter_path or base_adapter_path
            update_status = "updated" if adapter_valid else "invalid_adapter"
            update_reason = (
                f"created victim adapter for cycle {cycle_index}"
                if adapter_valid
                else "victim fine-tuning ran but did not produce a valid LoRA adapter"
            )
            if not adapter_valid:
                print(
                    f"warning: victim update for cycle {cycle_index} produced an invalid adapter: "
                    f"{adapter_path}; keeping previous active adapter.",
                    flush=True,
                )
        if verify_adapter_update and not dry_run_updates:
            verification = verify_adapter_artifact(adapter_path)

    result = {
        "cycle_index": cycle_index,
        "stage": "victim",
        "input_base_adapter_path": base_adapter_path,
        "trajectory_bank_path": str(trajectory_bank_path),
        "training_source": victim_training_source,
        "cumulative_episodes": len(episodes),
        "cumulative_turns": sum(len(episode.turns) for episode in episodes),
        "training_examples": len(examples),
        "base_adapter_path": base_adapter_path,
        "adapter_path": adapter_path,
        "adapter_valid": adapter_valid,
        "active_adapter_path": active_adapter_path,
        "input_victim_version": _adapter_version_label(base_adapter_path, "victim"),
        "output_victim_version": (
            f"victim_cycle_{cycle_index:03d}" if adapter_valid else _adapter_version_label(base_adapter_path, "victim")
        ),
        "update_status": update_status,
        "update_reason": update_reason,
        "adapter_verification": verification,
        "dry_run_updates": dry_run_updates,
        "victim_training_overrides": {
            "train_steps": victim_train_steps,
            "batch_size": victim_train_batch_size,
            "grad_accum_steps": victim_train_grad_accum_steps,
            "max_length": victim_train_max_length,
        },
    }
    _write_json(cycle_dir / "victim_stage_summary.json", result)
    _log_stage_wandb(
        wandb_project=wandb_project,
        run_name=f"{Path(run_dir).name}_cycle_{cycle_index:03d}_victim",
        metrics={
            "cycle": cycle_index,
            "victim/training_examples": len(examples),
            "victim/episodes_used": len(episodes),
            "victim/turns_used": sum(len(episode.turns) for episode in episodes),
            "victim/adapter_valid": float(adapter_valid),
            "victim/updated": float(update_status == "updated"),
            "victim/no_examples": float(update_status == "no_examples"),
        },
        summary=result,
    )
    volume.commit()
    return result


@app.function(image=image, volumes={"/root/outputs": volume})
def write_run_summary_remote(run_dir: str, summary: dict) -> str:
    path = Path(run_dir) / "summary.json"
    _write_json(path, summary)
    volume.commit()
    return str(path)


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={"/root/outputs": volume},
)
def orchestrate_coevolution_remote(
    run_dir: str,
    initial_policy_checkpoint: str,
    cycles: int = 3,
    start_cycle: int = 1,
    update_ppo: bool = True,
    update_attacker: bool = False,
    update_victim: bool = True,
    initial_victim_adapter: str | None = None,
    initial_attacker_adapter: str | None = None,
    ppo_epochs_per_cycle: int = 5,
    episodes_per_batch: int = 16,
    rollout_parallelism: int = 5,
    max_turns: int = 3,
    lr: float = 3e-4,
    start_epoch: int | None = None,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    min_reward_for_victim_training: float | None = None,
    min_reward_for_attacker_training: float = 1.0,
    attacker_example_source: str = "filtered_template",
    attacker_training_source: str = "cycle_only",
    victim_training_source: str = "cumulative",
    max_training_episodes: int | None = None,
    dry_run_updates: bool = False,
    verify_adapter_update: bool = False,
    attacker_train_steps: int | None = None,
    attacker_train_batch_size: int = 8,
    attacker_train_grad_accum_steps: int = 4,
    attacker_train_max_length: int = 2048,
    victim_train_steps: int | None = None,
    victim_train_batch_size: int = 1,
    victim_train_grad_accum_steps: int = 8,
    victim_train_max_length: int = 768,
    wandb_project: str | None = None,
    launch_command: str | None = None,
) -> dict:
    _print_launch_context(
        "full co-evolution orchestrator",
        run_dir=run_dir,
        launch_command=launch_command,
        extra={
            "initial_policy_checkpoint": initial_policy_checkpoint,
            "cycles": cycles,
            "start_cycle": start_cycle,
            "update_ppo": update_ppo,
            "update_attacker": update_attacker,
            "update_victim": update_victim,
            "dry_run_updates": dry_run_updates,
        },
    )
    return _orchestrate_stages(
        run_dir=run_dir,
        initial_policy_checkpoint=initial_policy_checkpoint,
        cycles=cycles,
        start_cycle=start_cycle,
        update_ppo=update_ppo,
        update_attacker=update_attacker,
        update_victim=update_victim,
        initial_victim_adapter=initial_victim_adapter,
        initial_attacker_adapter=initial_attacker_adapter,
        ppo_epochs_per_cycle=ppo_epochs_per_cycle,
        episodes_per_batch=episodes_per_batch,
        rollout_parallelism=rollout_parallelism,
        max_turns=max_turns,
        lr=lr,
        start_epoch=start_epoch,
        seed=seed,
        seed_prompt_file=seed_prompt_file,
        victim_history_turns=victim_history_turns,
        reward_backend=reward_backend,
        use_template_mutator=use_template_mutator,
        min_reward_for_victim_training=min_reward_for_victim_training,
        min_reward_for_attacker_training=min_reward_for_attacker_training,
        attacker_example_source=attacker_example_source,
        attacker_training_source=attacker_training_source,
        victim_training_source=victim_training_source,
        max_training_episodes=max_training_episodes,
        dry_run_updates=dry_run_updates,
        verify_adapter_update=verify_adapter_update,
        attacker_train_steps=attacker_train_steps,
        attacker_train_batch_size=attacker_train_batch_size,
        attacker_train_grad_accum_steps=attacker_train_grad_accum_steps,
        attacker_train_max_length=attacker_train_max_length,
        victim_train_steps=victim_train_steps,
        victim_train_batch_size=victim_train_batch_size,
        victim_train_grad_accum_steps=victim_train_grad_accum_steps,
        victim_train_max_length=victim_train_max_length,
        wandb_project=wandb_project,
        launch_command=launch_command,
    )


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    initial_policy_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_19.pt",
    remote_output_dir: str = "/root/outputs/coevolution_full",
    cycles: int = 3,
    start_cycle: int = 1,
    update_ppo: bool = True,
    update_attacker: bool = False,
    update_victim: bool = True,
    initial_victim_adapter: str | None = None,
    initial_attacker_adapter: str | None = None,
    ppo_epochs_per_cycle: int = 5,
    episodes_per_batch: int = 16,
    rollout_parallelism: int = 5,
    max_turns: int = 3,
    lr: float = 3e-4,
    start_epoch: int | None = None,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    min_reward_for_victim_training: float | None = None,
    min_reward_for_attacker_training: float = 1.0,
    attacker_example_source: str = "filtered_template",
    attacker_training_source: str = "cycle_only",
    victim_training_source: str = "cumulative",
    max_training_episodes: int | None = None,
    dry_run_updates: bool = False,
    verify_adapter_update: bool = False,
    attacker_train_steps: int | None = None,
    attacker_train_batch_size: int = 8,
    attacker_train_grad_accum_steps: int = 4,
    attacker_train_max_length: int = 2048,
    victim_train_steps: int | None = None,
    victim_train_batch_size: int = 1,
    victim_train_grad_accum_steps: int = 8,
    victim_train_max_length: int = 768,
    victim_only_cycle: int | None = None,
    victim_base_adapter_path: str | None = None,
    wandb_project: str | None = None,
    wait_for_result: bool = False,
    local_orchestrator: bool = False,
) -> None:
    resolved_run_name = run_name or f"coevolve_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = str(Path(remote_output_dir) / resolved_run_name)
    launch_command = _format_launch_command(
        run_name=resolved_run_name,
        initial_policy_checkpoint=initial_policy_checkpoint,
        remote_output_dir=remote_output_dir,
        cycles=cycles,
        start_cycle=start_cycle,
        update_ppo=update_ppo,
        update_attacker=update_attacker,
        update_victim=update_victim,
        initial_victim_adapter=initial_victim_adapter,
        initial_attacker_adapter=initial_attacker_adapter,
        ppo_epochs_per_cycle=ppo_epochs_per_cycle,
        episodes_per_batch=episodes_per_batch,
        rollout_parallelism=rollout_parallelism,
        max_turns=max_turns,
        lr=lr,
        start_epoch=start_epoch,
        seed=seed,
        seed_prompt_file=seed_prompt_file,
        victim_history_turns=victim_history_turns,
        reward_backend=reward_backend,
        use_template_mutator=use_template_mutator,
        min_reward_for_victim_training=min_reward_for_victim_training,
        min_reward_for_attacker_training=min_reward_for_attacker_training,
        attacker_example_source=attacker_example_source,
        attacker_training_source=attacker_training_source,
        victim_training_source=victim_training_source,
        max_training_episodes=max_training_episodes,
        dry_run_updates=dry_run_updates,
        verify_adapter_update=verify_adapter_update,
        attacker_train_steps=attacker_train_steps,
        attacker_train_batch_size=attacker_train_batch_size,
        attacker_train_grad_accum_steps=attacker_train_grad_accum_steps,
        attacker_train_max_length=attacker_train_max_length,
        victim_train_steps=victim_train_steps,
        victim_train_batch_size=victim_train_batch_size,
        victim_train_grad_accum_steps=victim_train_grad_accum_steps,
        victim_train_max_length=victim_train_max_length,
        victim_only_cycle=victim_only_cycle,
        victim_base_adapter_path=victim_base_adapter_path,
        wandb_project=wandb_project,
        wait_for_result=wait_for_result,
        local_orchestrator=local_orchestrator,
    )
    call_kwargs = {
        "run_dir": run_dir,
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "cycles": cycles,
        "start_cycle": start_cycle,
        "update_ppo": update_ppo,
        "update_attacker": update_attacker,
        "update_victim": update_victim,
        "initial_victim_adapter": initial_victim_adapter,
        "initial_attacker_adapter": initial_attacker_adapter,
        "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
        "episodes_per_batch": episodes_per_batch,
        "rollout_parallelism": rollout_parallelism,
        "max_turns": max_turns,
        "lr": lr,
        "start_epoch": start_epoch,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "victim_history_turns": victim_history_turns,
        "reward_backend": reward_backend,
        "use_template_mutator": use_template_mutator,
        "min_reward_for_victim_training": min_reward_for_victim_training,
        "min_reward_for_attacker_training": min_reward_for_attacker_training,
        "attacker_example_source": attacker_example_source,
        "attacker_training_source": attacker_training_source,
        "victim_training_source": victim_training_source,
        "max_training_episodes": max_training_episodes,
        "dry_run_updates": dry_run_updates,
        "verify_adapter_update": verify_adapter_update,
        "attacker_train_steps": attacker_train_steps,
        "attacker_train_batch_size": attacker_train_batch_size,
        "attacker_train_grad_accum_steps": attacker_train_grad_accum_steps,
        "attacker_train_max_length": attacker_train_max_length,
        "victim_train_steps": victim_train_steps,
        "victim_train_batch_size": victim_train_batch_size,
        "victim_train_grad_accum_steps": victim_train_grad_accum_steps,
        "victim_train_max_length": victim_train_max_length,
        "wandb_project": wandb_project,
        "launch_command": launch_command,
    }
    if victim_only_cycle is not None:
        function_call = train_victim_cycle_remote.spawn(
            run_dir=run_dir,
            cycle_index=victim_only_cycle,
            base_adapter_path=victim_base_adapter_path,
            min_reward_for_training=min_reward_for_victim_training,
            victim_training_source=victim_training_source,
            max_training_episodes=max_training_episodes,
            dry_run_updates=dry_run_updates,
            verify_adapter_update=verify_adapter_update,
            victim_train_steps=victim_train_steps,
            victim_train_batch_size=victim_train_batch_size,
            victim_train_grad_accum_steps=victim_train_grad_accum_steps,
            victim_train_max_length=victim_train_max_length,
            wandb_project=wandb_project,
            launch_command=launch_command,
        )
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted victim-only co-evolution stage without waiting for the result")
        print("launch command:")
        print(launch_command)
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("remote run dir:", run_dir)
        print("victim cycle:", victim_only_cycle)
        print("victim summary path:", f"{run_dir}/cycle_{victim_only_cycle:03d}/victim_stage_summary.json")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-coevolution-full` to follow progress.")
        return

    if not local_orchestrator:
        if wait_for_result:
            result = orchestrate_coevolution_remote.remote(**call_kwargs)
            print(json.dumps(result, indent=2))
            return
        function_call = orchestrate_coevolution_remote.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted full co-evolution orchestrator without waiting for the result")
        print("launch command:")
        print(launch_command)
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("remote run dir:", run_dir)
        print("remote summary path:", f"{run_dir}/summary.json")
        print("remote trajectory bank:", f"{run_dir}/trajectory_bank/episodes.jsonl")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-coevolution-full` to follow progress.")
        return

    result = _orchestrate_stages(**call_kwargs)
    print(json.dumps(result, indent=2))


def _orchestrate_stages(
    *,
    run_dir: str,
    initial_policy_checkpoint: str,
    cycles: int,
    start_cycle: int,
    update_ppo: bool,
    update_attacker: bool,
    update_victim: bool,
    initial_victim_adapter: str | None,
    initial_attacker_adapter: str | None,
    ppo_epochs_per_cycle: int,
    episodes_per_batch: int,
    rollout_parallelism: int,
    max_turns: int,
    lr: float,
    start_epoch: int | None,
    seed: int,
    seed_prompt_file: str,
    victim_history_turns: int | None,
    reward_backend: str | None,
    use_template_mutator: bool,
    min_reward_for_victim_training: float | None,
    min_reward_for_attacker_training: float,
    attacker_example_source: str,
    attacker_training_source: str,
    victim_training_source: str,
    max_training_episodes: int | None,
    dry_run_updates: bool,
    verify_adapter_update: bool,
    attacker_train_steps: int | None,
    attacker_train_batch_size: int,
    attacker_train_grad_accum_steps: int,
    attacker_train_max_length: int,
    victim_train_steps: int | None,
    victim_train_batch_size: int,
    victim_train_grad_accum_steps: int,
    victim_train_max_length: int,
    wandb_project: str | None,
    launch_command: str | None,
) -> dict:
    _print_launch_context(
        "local/full co-evolution stages",
        run_dir=run_dir,
        launch_command=launch_command,
        extra={
            "initial_policy_checkpoint": initial_policy_checkpoint,
            "cycles": cycles,
            "start_cycle": start_cycle,
            "update_ppo": update_ppo,
            "update_attacker": update_attacker,
            "update_victim": update_victim,
            "dry_run_updates": dry_run_updates,
        },
    )
    current_policy_checkpoint = initial_policy_checkpoint
    current_attacker_adapter = initial_attacker_adapter
    current_victim_adapter = initial_victim_adapter
    next_epoch = start_epoch
    cycle_results = _load_existing_cycles(run_dir)
    if cycle_results:
        cycle_results = [
            cycle
            for cycle in cycle_results
            if isinstance(cycle, dict) and int(cycle.get("cycle_index", 0)) < start_cycle
        ] or [
            cycle
            for cycle in cycle_results
            if isinstance(cycle, dict) and int(cycle.get("cycle_index", 0)) <= cycles
        ]
        last_cycle = cycle_results[-1]
        current_policy_checkpoint = last_cycle.get("output_policy_checkpoint") or current_policy_checkpoint
        current_attacker_adapter = last_cycle.get("output_attacker_adapter") or current_attacker_adapter
        current_victim_adapter = last_cycle.get("output_victim_adapter") or current_victim_adapter
        if next_epoch is None:
            policy = last_cycle.get("policy", {})
            policy_end_epoch = policy.get("policy_end_epoch")
            if policy_end_epoch is not None:
                next_epoch = int(policy_end_epoch) + 1
        completed_cycle_index = int(last_cycle.get("cycle_index", 0))
        if completed_cycle_index >= start_cycle:
            print(
                f"Resuming from existing summary: cycles 1..{completed_cycle_index} "
                f"already complete; continuing at cycle {completed_cycle_index + 1}.",
                flush=True,
            )
            start_cycle = completed_cycle_index + 1

    for cycle_index in range(start_cycle, cycles + 1):
        cycle_input_policy = current_policy_checkpoint
        cycle_input_attacker = current_attacker_adapter
        cycle_input_victim = current_victim_adapter

        policy_result = {
            "cycle_index": cycle_index,
            "stage": "policy",
            "skipped": True,
            "input_policy_checkpoint": current_policy_checkpoint,
            "input_attacker_adapter": current_attacker_adapter,
            "input_victim_adapter": current_victim_adapter,
            "final_policy_checkpoint": current_policy_checkpoint,
        }
        if update_ppo:
            existing_policy_result = _load_completed_stage(run_dir, cycle_index, "policy")
            if existing_policy_result is not None:
                print(f"\nReusing completed PPO cycle {cycle_index}/{cycles}")
                policy_result = existing_policy_result
            else:
                print(f"\nStarting PPO cycle {cycle_index}/{cycles}")
                policy_result = train_policy_cycle_remote.remote(
                    run_dir=run_dir,
                    cycle_index=cycle_index,
                    policy_checkpoint=current_policy_checkpoint,
                    attacker_lora_adapter=current_attacker_adapter,
                    victim_lora_adapter=current_victim_adapter,
                    start_epoch=next_epoch,
                    ppo_epochs=ppo_epochs_per_cycle,
                    episodes_per_batch=episodes_per_batch,
                    rollout_parallelism=rollout_parallelism,
                    max_turns=max_turns,
                    lr=lr,
                    seed=seed,
                    seed_prompt_file=seed_prompt_file,
                    victim_history_turns=victim_history_turns,
                    reward_backend=reward_backend,
                    use_template_mutator=use_template_mutator,
                    wandb_project=wandb_project,
                    launch_command=launch_command,
                )
            current_policy_checkpoint = policy_result["final_policy_checkpoint"]
            next_epoch = int(policy_result["policy_end_epoch"]) + 1
        else:
            print(f"\nSkipping PPO cycle {cycle_index}/{cycles}")

        attacker_result = {
            "cycle_index": cycle_index,
            "stage": "attacker",
            "skipped": True,
            "input_base_adapter_path": current_attacker_adapter,
            "active_adapter_path": current_attacker_adapter,
        }
        if update_attacker:
            existing_attacker_result = _load_completed_stage(run_dir, cycle_index, "attacker")
            if existing_attacker_result is not None:
                print(f"\nReusing completed attacker update {cycle_index}/{cycles}")
                attacker_result = existing_attacker_result
            else:
                print(f"\nStarting attacker update {cycle_index}/{cycles}")
                attacker_result = train_attacker_cycle_remote.remote(
                    run_dir=run_dir,
                    cycle_index=cycle_index,
                    base_adapter_path=current_attacker_adapter,
                    min_reward_for_training=min_reward_for_attacker_training,
                    attacker_example_source=attacker_example_source,
                    attacker_training_source=attacker_training_source,
                    max_training_episodes=max_training_episodes,
                    dry_run_updates=dry_run_updates,
                    verify_adapter_update=verify_adapter_update,
                    attacker_train_steps=attacker_train_steps,
                    attacker_train_batch_size=attacker_train_batch_size,
                    attacker_train_grad_accum_steps=attacker_train_grad_accum_steps,
                    attacker_train_max_length=attacker_train_max_length,
                    wandb_project=wandb_project,
                    launch_command=launch_command,
                )
            current_attacker_adapter = attacker_result["active_adapter_path"]
        else:
            print(f"\nSkipping attacker update {cycle_index}/{cycles}")

        victim_result = {
            "cycle_index": cycle_index,
            "stage": "victim",
            "skipped": True,
            "input_base_adapter_path": current_victim_adapter,
            "active_adapter_path": current_victim_adapter,
        }
        if update_victim:
            existing_victim_result = _load_completed_stage(run_dir, cycle_index, "victim")
            if existing_victim_result is not None:
                print(f"\nReusing completed victim update {cycle_index}/{cycles}")
                victim_result = existing_victim_result
            else:
                print(f"\nStarting victim update {cycle_index}/{cycles}")
                victim_result = train_victim_cycle_remote.remote(
                    run_dir=run_dir,
                    cycle_index=cycle_index,
                    base_adapter_path=current_victim_adapter,
                    min_reward_for_training=min_reward_for_victim_training,
                    victim_training_source=victim_training_source,
                    max_training_episodes=max_training_episodes,
                    dry_run_updates=dry_run_updates,
                    verify_adapter_update=verify_adapter_update,
                    victim_train_steps=victim_train_steps,
                    victim_train_batch_size=victim_train_batch_size,
                    victim_train_grad_accum_steps=victim_train_grad_accum_steps,
                    victim_train_max_length=victim_train_max_length,
                    wandb_project=wandb_project,
                    launch_command=launch_command,
                )
            current_victim_adapter = victim_result["active_adapter_path"]
        else:
            print(f"\nSkipping victim update {cycle_index}/{cycles}")

        cycle_results.append(
            {
                "cycle_index": cycle_index,
                "input_policy_checkpoint": cycle_input_policy,
                "input_attacker_adapter": cycle_input_attacker,
                "input_victim_adapter": cycle_input_victim,
                "policy": policy_result,
                "attacker": attacker_result,
                "victim": victim_result,
                "output_policy_checkpoint": current_policy_checkpoint,
                "output_attacker_adapter": current_attacker_adapter,
                "output_victim_adapter": current_victim_adapter,
            }
        )
        summary = _run_summary(
            run_dir=run_dir,
            initial_policy_checkpoint=initial_policy_checkpoint,
            final_policy_checkpoint=current_policy_checkpoint,
            initial_attacker_adapter=initial_attacker_adapter,
            final_attacker_adapter=current_attacker_adapter,
            final_victim_adapter=current_victim_adapter,
            cycles=cycle_results,
            config={
                "cycles": cycles,
                "start_cycle": start_cycle,
                "update_ppo": update_ppo,
                "update_attacker": update_attacker,
                "update_victim": update_victim,
                "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
                "episodes_per_batch": episodes_per_batch,
                "rollout_parallelism": rollout_parallelism,
                "max_turns": max_turns,
                "lr": lr,
                "start_epoch": start_epoch,
                "seed": seed,
                "seed_prompt_file": seed_prompt_file,
                "victim_history_turns": victim_history_turns,
                "reward_backend": reward_backend,
                "use_template_mutator": use_template_mutator,
                "min_reward_for_victim_training": min_reward_for_victim_training,
                "min_reward_for_attacker_training": min_reward_for_attacker_training,
                "attacker_example_source": attacker_example_source,
                "attacker_training_source": attacker_training_source,
                "victim_training_source": victim_training_source,
                "max_training_episodes": max_training_episodes,
                "dry_run_updates": dry_run_updates,
                "attacker_training_overrides": {
                    "train_steps": attacker_train_steps,
                    "batch_size": attacker_train_batch_size,
                    "grad_accum_steps": attacker_train_grad_accum_steps,
                    "max_length": attacker_train_max_length,
                },
                "victim_training_overrides": {
                    "train_steps": victim_train_steps,
                    "batch_size": victim_train_batch_size,
                    "grad_accum_steps": victim_train_grad_accum_steps,
                    "max_length": victim_train_max_length,
                },
            },
        )
        summary_path = write_run_summary_remote.remote(run_dir, summary)
        print(f"cycle {cycle_index} summary written to {summary_path}")

    print("\nCo-evolution complete")
    print("=" * 80)
    print("remote run dir:", run_dir)
    print("final policy checkpoint:", current_policy_checkpoint)
    print("final attacker adapter:", current_attacker_adapter)
    print("final victim adapter:", current_victim_adapter)
    print("summary:", f"{run_dir}/summary.json")
    return summary


def _run_summary(
    *,
    run_dir: str,
    initial_policy_checkpoint: str,
    final_policy_checkpoint: str,
    initial_attacker_adapter: str | None,
    final_attacker_adapter: str | None,
    final_victim_adapter: str | None,
    cycles: list[dict],
    config: dict,
) -> dict:
    return {
        "run_dir": run_dir,
        "trajectory_bank_path": f"{run_dir}/trajectory_bank/episodes.jsonl",
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "final_policy_checkpoint": final_policy_checkpoint,
        "initial_attacker_adapter": initial_attacker_adapter,
        "final_attacker_adapter": final_attacker_adapter,
        "final_victim_adapter": final_victim_adapter,
        "cycles": cycles,
        "config": config,
    }


def _print_launch_context(
    stage: str,
    *,
    run_dir: str,
    launch_command: str | None,
    extra: dict[str, object] | None = None,
) -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"Launch context: {stage}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    if launch_command:
        print("command:", flush=True)
        print(launch_command, flush=True)
    if extra:
        print("stage inputs:", flush=True)
        for key, value in extra.items():
            print(f"  {key}: {value}", flush=True)
    print("=" * 80, flush=True)


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


def _format_launch_command(**kwargs) -> str:
    parts = ["modal", "run"]
    if not kwargs.get("wait_for_result") and not kwargs.get("local_orchestrator"):
        parts.append("--detach")
    parts.append("modal_coevolve_full.py")
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            parts.append(flag)
        else:
            parts.append(f"{flag}={shlex.quote(str(value))}")
    return " ".join(parts)


def _load_existing_cycles(run_dir: str) -> list[dict]:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return []
    cycles = summary.get("cycles", [])
    return cycles if isinstance(cycles, list) else []


def _load_completed_stage(run_dir: str, cycle_index: int, stage: str) -> dict | None:
    if stage not in {"policy", "attacker", "victim"}:
        raise ValueError(f"unknown stage: {stage}")
    path = Path(run_dir) / f"cycle_{cycle_index:03d}" / f"{stage}_stage_summary.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as exc:
        print(f"warning: could not load existing {stage} stage summary {path}: {exc}", flush=True)
        return None
    if not isinstance(result, dict):
        return None
    if stage == "policy":
        checkpoint = result.get("final_policy_checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            print(
                f"warning: ignoring incomplete PPO stage for cycle {cycle_index}; "
                f"checkpoint missing: {checkpoint}",
                flush=True,
            )
            return None
    return result


def _load_policy_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        print(f"warning: could not load policy summary {path}: {exc}", flush=True)
        return []
    epochs = summary.get("epochs", [])
    return epochs if isinstance(epochs, list) else []


def _load_completed_policy_episode_records(
    path: Path,
    *,
    cycle_index: int,
    epoch: int,
    max_turns: int,
) -> dict[int, dict]:
    if not path.exists():
        return {}
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("cycle_index", -1)) != int(cycle_index):
                continue
            if int(record.get("epoch", -1)) != int(epoch):
                continue
            if int(record.get("num_turns", 0) or 0) < int(max_turns):
                continue
            episode_index = int(record.get("episode_index", -1))
            if episode_index < 0:
                continue
            records[episode_index] = record
    return records


def _policy_episode_from_record(record: dict) -> tuple[dict, object]:
    from redteam_rl.actions import ACTIONS, AttackAction
    from redteam_rl.types import DialogueTurn, EpisodeState

    state = EpisodeState(
        seed_prompt=str(record["seed_prompt"]),
        initial_template=str(record.get("initial_template") or "{REQUEST}"),
    )
    episode = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "turn_metadata": []}
    for raw_turn in record.get("turns", []):
        episode["states"].append(copy.deepcopy(state))
        metadata = dict(raw_turn.get("metadata", {}) or {})
        action = AttackAction(raw_turn["action"]) if raw_turn.get("action") else None
        if action is None:
            raise ValueError("Cannot reconstruct PPO episode without an action.")
        reward = float(raw_turn.get("reward") or 0.0)
        turn = DialogueTurn(
            user_message=str(raw_turn.get("user_message", "")),
            victim_response=str(raw_turn.get("victim_response", "")),
            action=action,
            attack_template=raw_turn.get("attack_template") or metadata.get("attack_template"),
            reward=reward,
            metadata=metadata,
        )
        state.turns.append(turn)
        episode["actions"].append(ACTIONS.index(action))
        episode["log_probs"].append(float(metadata.get("policy_log_prob", 0.0)))
        episode["values"].append(float(metadata.get("policy_value", 0.0)))
        episode["rewards"].append(reward)
        episode["turn_metadata"].append(metadata)
    episode["final_state"] = copy.deepcopy(state)
    return episode, copy.deepcopy(state)


def _finalize_policy_stage_summary(
    *,
    cycle_dir: Path,
    cycle_index: int,
    policy_checkpoint: str,
    active_attacker_adapter: str | None,
    active_victim_adapter: str | None,
    first_epoch: int,
    requested_final_epoch: int,
    policy_dir: Path,
    final_checkpoint: str,
    episode_log_path: Path,
    trajectory_bank_path: Path,
    epoch_summaries: list[dict],
    all_cycle_episodes: list,
    rewards: list[float],
) -> dict:
    summary_episode_count = len(all_cycle_episodes) or sum(
        int(summary.get("episodes", 0) or 0)
        for summary in epoch_summaries
    )
    summary_turn_count = sum(len(state.turns) for state in all_cycle_episodes) or sum(
        int((summary.get("rollout_metrics") or {}).get("turns", 0) or 0)
        for summary in epoch_summaries
    )
    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    max_reward = max(rewards) if rewards else 0.0
    if not rewards and epoch_summaries:
        reward_means = [
            float((summary.get("rollout_metrics") or {}).get("reward_mean", 0.0) or 0.0)
            for summary in epoch_summaries
        ]
        reward_maxes = [
            float((summary.get("rollout_metrics") or {}).get("reward_max", 0.0) or 0.0)
            for summary in epoch_summaries
        ]
        average_reward = sum(reward_means) / len(reward_means) if reward_means else 0.0
        max_reward = max(reward_maxes) if reward_maxes else 0.0
    result = {
        "cycle_index": cycle_index,
        "stage": "policy",
        "input_policy_checkpoint": policy_checkpoint,
        "input_attacker_adapter": active_attacker_adapter,
        "input_victim_adapter": active_victim_adapter,
        "policy_start_epoch": first_epoch,
        "policy_end_epoch": requested_final_epoch,
        "policy_dir": str(policy_dir),
        "final_policy_checkpoint": final_checkpoint,
        "training_episodes_path": str(episode_log_path),
        "trajectory_bank_path": str(trajectory_bank_path),
        "cycle_episodes": summary_episode_count,
        "cycle_turns": summary_turn_count,
        "average_reward": average_reward,
        "max_reward": max_reward,
        "queries_per_success": _mean_first_success_turn(all_cycle_episodes),
        "mean_first_success_turn": _mean_first_success_turn(all_cycle_episodes),
        "first_success_turn_histogram": _first_success_turn_histogram(all_cycle_episodes),
        "epochs": epoch_summaries,
    }
    _write_json(cycle_dir / "policy_stage_summary.json", result)
    volume.commit()
    return result


def _valid_lora_adapter_path(adapter_path: str | Path | None) -> str | None:
    if not adapter_path:
        return None
    path = Path(adapter_path)
    if not path.exists() or not path.is_dir():
        return None
    if not (path / "adapter_config.json").exists():
        return None
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
        return str(path)
    return None


def _load_cycle_training_episodes(path: Path):
    from redteam_rl.trajectory_bank import episode_from_record

    if not path.exists():
        return []
    episodes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(episode_from_record(json.loads(line)))
    return episodes


def _first_success_turn_from_turns(turns) -> int | None:
    for index, turn in enumerate(turns, start=1):
        label = str(turn.metadata.get("judge_label", "")).upper()
        if label == "UNSAFE" or float(turn.reward or 0.0) >= 1.0:
            return index
    return None


def _first_success_turns(episodes) -> list[int]:
    return [
        first_turn
        for state in episodes
        if (first_turn := _first_success_turn_from_turns(state.turns)) is not None
    ]


def _mean_first_success_turn(episodes) -> float | None:
    first_success_turns = _first_success_turns(episodes)
    if not first_success_turns:
        return None
    return sum(first_success_turns) / len(first_success_turns)


def _first_success_turn_histogram(episodes) -> dict[str, int]:
    first_success_turns = _first_success_turns(episodes)
    return {
        str(turn): first_success_turns.count(turn)
        for turn in sorted(set(first_success_turns))
    }


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _adapter_version_label(adapter_path: str | Path | None, prefix: str) -> str:
    if not adapter_path:
        return f"{prefix}_base"
    path = Path(adapter_path)
    cycle_match = re.search(r"cycle_(\d{3})", str(path))
    if cycle_match:
        return f"{prefix}_cycle_{cycle_match.group(1)}"
    round_match = re.search(r"round_(\d{3})", str(path))
    if round_match:
        return f"{prefix}_round_{round_match.group(1)}"
    return f"{prefix}_{path.name or 'adapter'}"


def _log_stage_wandb(
    *,
    wandb_project: str | None,
    run_name: str,
    metrics: dict[str, float | int],
    summary: dict,
) -> None:
    if not wandb_project:
        return
    try:
        import wandb

        run = wandb.init(project=wandb_project, name=run_name, resume="never")
        run.log(metrics, step=int(metrics.get("cycle", 0)))
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                run.summary[key] = value
        run.finish()
    except Exception as exc:
        print(f"wandb stage logging failed for {run_name}: {exc}", flush=True)


def _summarize_policy_batch(batch: list[dict]) -> dict:
    from redteam_rl.actions import ACTIONS

    episode_returns = [sum(float(reward) for reward in episode["rewards"]) for episode in batch]
    episode_successes = [
        first_success_turn is not None
        for episode in batch
        for first_success_turn in [_first_success_turn_from_episode_dict(episode)]
    ]
    first_success_turns = [
        first_success_turn
        for episode in batch
        if (first_success_turn := _first_success_turn_from_episode_dict(episode)) is not None
    ]
    all_rewards = [float(reward) for episode in batch for reward in episode["rewards"]]
    all_values = [float(value) for episode in batch for value in episode["values"]]
    all_actions = [int(action) for episode in batch for action in episode["actions"]]
    all_turn_metadata = [metadata for episode in batch for metadata in episode.get("turn_metadata", [])]
    action_counts = {ACTIONS[index].value: 0 for index in range(len(ACTIONS))}
    for action in all_actions:
        action_counts[ACTIONS[action].value] += 1
    total_actions = max(len(all_actions), 1)

    mutator_refusals = [bool(metadata.get("mutator_refused")) for metadata in all_turn_metadata]
    mutator_invalid_templates = [bool(metadata.get("mutator_invalid_template")) for metadata in all_turn_metadata]
    mutator_fallbacks = [bool(metadata.get("mutator_fallback_used")) for metadata in all_turn_metadata]
    raw_templates = [str(metadata.get("raw_attack_template", "")) for metadata in all_turn_metadata]
    final_templates = [str(metadata.get("attack_template", "")) for metadata in all_turn_metadata]
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


def _first_success_turn_from_episode_dict(episode: dict) -> int | None:
    rewards = episode.get("rewards", [])
    turn_metadata = episode.get("turn_metadata", [])
    for index, (reward, metadata) in enumerate(zip(rewards, turn_metadata), start=1):
        label = str(metadata.get("judge_label", "")).upper()
        if label == "UNSAFE" or float(reward or 0.0) >= 1.0:
            return index
    return None


def _compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [0.0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return returns, advantages


def _ppo_update(policy, batch, optimizer, clip_eps=0.2, value_coef=0.5, ent_coef=0.01) -> dict[str, float]:
    import torch

    device = policy.device
    all_states = []
    all_actions = []
    old_log_probs = []
    returns = []
    advantages = []

    for episode in batch:
        episode_returns, episode_advantages = _compute_gae(episode["rewards"], episode["values"])
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


def _save_policy_checkpoint(policy, optimizer, checkpoint_path: Path, epoch: int, trainer_state: dict) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.update(
        {
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "trainer_state": trainer_state,
        }
    )
    torch.save(checkpoint, checkpoint_path)


def _append_training_episode(path: Path, *, cycle_index: int, epoch: int, episode_index: int, episode_seed: int, state) -> None:
    record = {
        "cycle_index": cycle_index,
        "epoch": epoch,
        "episode_index": episode_index,
        "seed": episode_seed,
        "template_format": "rlbreaker_mutable_template",
        "seed_prompt": state.seed_prompt,
        "initial_template": state.initial_template,
        "num_turns": len(state.turns),
        "episode_return": sum(float(turn.reward or 0.0) for turn in state.turns),
        "success": any(str(turn.metadata.get("judge_label", "")).upper() == "UNSAFE" for turn in state.turns),
        "first_success_turn": _first_success_turn_from_turns(state.turns),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _checkpoint_epoch(checkpoint: dict, checkpoint_path: str) -> int | None:
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        return int(epoch)
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _judge_model_name(cfg, reward_backend: str) -> str:
    if reward_backend in {"qwen_judge", "qwen_safety_judge"}:
        return cfg.models.qwen_safety_judge
    if reward_backend == "prompt_guard":
        return cfg.models.prompt_guard
    if reward_backend == "llama_guard":
        return cfg.models.llama_guard
    if reward_backend == "wildguard":
        return cfg.models.wildguard
    return "fake"


if __name__ == "__main__":
    pass
