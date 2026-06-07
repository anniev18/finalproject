"""Robust Modal orchestration for PPO/victim co-evolution.

Unlike ``modal_coevolve.py``, this version uses a fresh Modal container for
each PPO cycle and each victim LoRA update. That costs some model reload time,
but avoids vLLM/PyTorch GPU memory leaks between stages.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-coevolution-robust")
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
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train_policy_cycle_remote(
    run_dir: str,
    cycle_index: int,
    policy_checkpoint: str,
    victim_lora_adapter: str | None = None,
    start_epoch: int | None = None,
    ppo_epochs: int = 5,
    episodes_per_batch: int = 16,
    max_turns: int = 3,
    lr: float = 3e-4,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    config_path: str = "/root/configs/default.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
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
    from redteam_rl.orchestration import annotate_episode_state
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.trajectory_bank import append_episodes
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
    run_id = new_run_id(f"robust_coevolve_cycle_{cycle_index:03d}")

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

    config_initial_attacker_adapter = cfg.attacker_evolution.get("initial_adapter_path")
    use_attacker_lora = config_initial_attacker_adapter is not None
    mutator = (
        TemplateMutator()
        if use_template_mutator
        else LLMMutator(
            cfg.mutator_config(enable_lora=use_attacker_lora or cfg.mutator.get("enable_lora", False)),
            capture_debug_prompt=True,
            lora_adapter_path=config_initial_attacker_adapter,
        )
    )
    attacker = (
        EvolvingAttacker(
            policy=policy,
            mutator=mutator,
            config=cfg.attacker_evolution_config(initial_adapter_path=config_initial_attacker_adapter),
        )
        if cfg.attacker_evolution.get("enabled", False) or config_initial_attacker_adapter
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
                    "victim_lora_adapter": victim_lora_adapter,
                    "ppo_epochs": ppo_epochs,
                    "episodes_per_batch": episodes_per_batch,
                    "max_turns": max_turns,
                    "reward_backend": selected_reward_backend,
                },
            )
        except Exception as exc:
            print(f"wandb init failed: {exc}", flush=True)

    all_cycle_episodes = []
    epoch_summaries = []
    rewards = []
    for epoch in range(first_epoch, first_epoch + ppo_epochs):
        batch = []
        epoch_episodes = []
        for episode_index in range(episodes_per_batch):
            episode_seed = seed + epoch * episodes_per_batch + episode_index
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
            metadata = build_model_metadata(
                run_id=run_id,
                victim_model=cfg.models.victim,
                victim_adapter_path=victim_lora_adapter,
                attacker_model="template" if use_template_mutator else cfg.models.mutator,
                judge_model=_judge_model_name(cfg, selected_reward_backend),
                policy_type="trailblazer_coevolution_robust",
                policy_checkpoint=policy_checkpoint,
                extra={
                    "cycle_index": cycle_index,
                    "epoch": epoch,
                    "reward_backend": selected_reward_backend,
                },
            )
            annotate_episode_state(state, metadata)
            episode["final_state"] = copy.deepcopy(state)
            batch.append(episode)
            epoch_episodes.append(copy.deepcopy(state))
            _append_training_episode(
                episode_log_path,
                cycle_index=cycle_index,
                epoch=epoch,
                episode_index=episode_index,
                episode_seed=episode_seed,
                state=state,
            )

        loss_value = _ppo_update(policy, batch, optimizer)
        checkpoint_path = policy_dir / f"checkpoint_epoch_{epoch}.pt"
        _save_policy_checkpoint(
            policy=policy,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            trainer_state={
                "cycle_index": cycle_index,
                "policy_checkpoint": policy_checkpoint,
                "victim_lora_adapter": victim_lora_adapter,
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
                "source": "modal_coevolve_robust.py",
                "run_id": run_id,
                "cycle_index": cycle_index,
                "epoch": epoch,
                "victim_adapter_path": victim_lora_adapter,
            },
        )
        all_cycle_episodes.extend(epoch_episodes)
        for state in epoch_episodes:
            rewards.extend(float(turn.reward or 0.0) for turn in state.turns)
        epoch_summary = {
            "epoch": epoch,
            "loss": loss_value,
            "checkpoint_path": str(checkpoint_path),
            "episodes": len(epoch_episodes),
            "average_reward_so_far": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward_so_far": max(rewards) if rewards else 0.0,
        }
        epoch_summaries.append(epoch_summary)
        _write_json(cycle_dir / "policy_summary.json", {"epochs": epoch_summaries})
        volume.commit()
        print(
            f"cycle={cycle_index} epoch={epoch} loss={loss_value:.6f} checkpoint={checkpoint_path}",
            flush=True,
        )
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "cycle": cycle_index,
                        "epoch": epoch,
                        "loss": loss_value,
                        "average_reward_so_far": epoch_summary["average_reward_so_far"],
                        "max_reward_so_far": epoch_summary["max_reward_so_far"],
                    },
                    step=epoch,
                )
            except Exception:
                pass

    final_checkpoint = str(policy_dir / f"checkpoint_epoch_{first_epoch + ppo_epochs - 1}.pt")
    result = {
        "cycle_index": cycle_index,
        "stage": "policy",
        "input_policy_checkpoint": policy_checkpoint,
        "input_victim_adapter": victim_lora_adapter,
        "policy_start_epoch": first_epoch,
        "policy_end_epoch": first_epoch + ppo_epochs - 1,
        "policy_dir": str(policy_dir),
        "final_policy_checkpoint": final_checkpoint,
        "training_episodes_path": str(episode_log_path),
        "trajectory_bank_path": str(trajectory_bank_path),
        "cycle_episodes": len(all_cycle_episodes),
        "cycle_turns": sum(len(state.turns) for state in all_cycle_episodes),
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "epochs": epoch_summaries,
    }
    _write_json(cycle_dir / "policy_stage_summary.json", result)
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
)
def train_victim_cycle_remote(
    run_dir: str,
    cycle_index: int,
    base_adapter_path: str | None = None,
    config_path: str = "/root/configs/default.json",
    min_reward_for_training: float | None = None,
    verify_adapter_update: bool = False,
    victim_train_steps: int | None = None,
    victim_train_batch_size: int = 1,
    victim_train_grad_accum_steps: int = 8,
    victim_train_max_length: int = 768,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import sys

    sys.path.insert(0, "/root")

    from redteam_rl.config import load_config
    from redteam_rl.orchestration import (
        VictimEvolutionConfig,
        build_safety_examples,
        verify_adapter_update as verify_adapter_artifact,
    )
    from redteam_rl.trajectory_bank import load_episodes
    from redteam_rl.victim_training import train_victim_from_episodes

    cfg = load_config(config_path)
    run_path = Path(run_dir)
    cycle_dir = run_path / f"cycle_{cycle_index:03d}"
    trajectory_bank_path = run_path / "trajectory_bank" / "episodes.jsonl"
    episodes = load_episodes(trajectory_bank_path)
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
    verification = None
    active_adapter_path = base_adapter_path
    if examples:
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
        active_adapter_path = adapter_path
        if verify_adapter_update:
            verification = verify_adapter_artifact(adapter_path)

    result = {
        "cycle_index": cycle_index,
        "stage": "victim",
        "input_base_adapter_path": base_adapter_path,
        "trajectory_bank_path": str(trajectory_bank_path),
        "cumulative_episodes": len(episodes),
        "cumulative_turns": sum(len(episode.turns) for episode in episodes),
        "training_examples": len(examples),
        "base_adapter_path": base_adapter_path,
        "adapter_path": adapter_path,
        "active_adapter_path": active_adapter_path,
        "adapter_verification": verification,
        "victim_training_overrides": {
            "train_steps": victim_train_steps,
            "batch_size": victim_train_batch_size,
            "grad_accum_steps": victim_train_grad_accum_steps,
            "max_length": victim_train_max_length,
        },
    }
    _write_json(cycle_dir / "victim_stage_summary.json", result)
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
    initial_victim_adapter: str | None = None,
    ppo_epochs_per_cycle: int = 5,
    episodes_per_batch: int = 16,
    max_turns: int = 3,
    lr: float = 3e-4,
    start_epoch: int | None = None,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    min_reward_for_victim_training: float | None = None,
    verify_adapter_update: bool = False,
    victim_train_steps: int | None = None,
    victim_train_batch_size: int = 1,
    victim_train_grad_accum_steps: int = 8,
    victim_train_max_length: int = 768,
    wandb_project: str | None = None,
) -> dict:
    return _orchestrate_stages(
        run_dir=run_dir,
        initial_policy_checkpoint=initial_policy_checkpoint,
        cycles=cycles,
        start_cycle=start_cycle,
        initial_victim_adapter=initial_victim_adapter,
        ppo_epochs_per_cycle=ppo_epochs_per_cycle,
        episodes_per_batch=episodes_per_batch,
        max_turns=max_turns,
        lr=lr,
        start_epoch=start_epoch,
        seed=seed,
        seed_prompt_file=seed_prompt_file,
        victim_history_turns=victim_history_turns,
        reward_backend=reward_backend,
        use_template_mutator=use_template_mutator,
        min_reward_for_victim_training=min_reward_for_victim_training,
        verify_adapter_update=verify_adapter_update,
        victim_train_steps=victim_train_steps,
        victim_train_batch_size=victim_train_batch_size,
        victim_train_grad_accum_steps=victim_train_grad_accum_steps,
        victim_train_max_length=victim_train_max_length,
        wandb_project=wandb_project,
    )


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    initial_policy_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_19.pt",
    remote_output_dir: str = "/root/outputs/coevolution",
    cycles: int = 3,
    start_cycle: int = 1,
    initial_victim_adapter: str | None = None,
    ppo_epochs_per_cycle: int = 5,
    episodes_per_batch: int = 16,
    max_turns: int = 3,
    lr: float = 3e-4,
    start_epoch: int | None = None,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    min_reward_for_victim_training: float | None = None,
    verify_adapter_update: bool = False,
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
    resolved_run_name = run_name or f"coevolve_robust_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = str(Path(remote_output_dir) / resolved_run_name)
    call_kwargs = {
        "run_dir": run_dir,
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "cycles": cycles,
        "start_cycle": start_cycle,
        "initial_victim_adapter": initial_victim_adapter,
        "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
        "episodes_per_batch": episodes_per_batch,
        "max_turns": max_turns,
        "lr": lr,
        "start_epoch": start_epoch,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "victim_history_turns": victim_history_turns,
        "reward_backend": reward_backend,
        "use_template_mutator": use_template_mutator,
        "min_reward_for_victim_training": min_reward_for_victim_training,
        "verify_adapter_update": verify_adapter_update,
        "victim_train_steps": victim_train_steps,
        "victim_train_batch_size": victim_train_batch_size,
        "victim_train_grad_accum_steps": victim_train_grad_accum_steps,
        "victim_train_max_length": victim_train_max_length,
        "wandb_project": wandb_project,
    }
    if victim_only_cycle is not None:
        function_call = train_victim_cycle_remote.spawn(
            run_dir=run_dir,
            cycle_index=victim_only_cycle,
            base_adapter_path=victim_base_adapter_path,
            min_reward_for_training=min_reward_for_victim_training,
            verify_adapter_update=verify_adapter_update,
            victim_train_steps=victim_train_steps,
            victim_train_batch_size=victim_train_batch_size,
            victim_train_grad_accum_steps=victim_train_grad_accum_steps,
            victim_train_max_length=victim_train_max_length,
        )
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted victim-only co-evolution stage without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("remote run dir:", run_dir)
        print("victim cycle:", victim_only_cycle)
        print("victim summary path:", f"{run_dir}/cycle_{victim_only_cycle:03d}/victim_stage_summary.json")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-coevolution-robust` to follow progress.")
        return

    if not local_orchestrator:
        if wait_for_result:
            result = orchestrate_coevolution_remote.remote(**call_kwargs)
            print(json.dumps(result, indent=2))
            return
        function_call = orchestrate_coevolution_remote.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted robust co-evolution orchestrator without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("remote run dir:", run_dir)
        print("remote summary path:", f"{run_dir}/summary.json")
        print("remote trajectory bank:", f"{run_dir}/trajectory_bank/episodes.jsonl")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-coevolution-robust` to follow progress.")
        return

    result = _orchestrate_stages(**call_kwargs)
    print(json.dumps(result, indent=2))


def _orchestrate_stages(
    *,
    run_dir: str,
    initial_policy_checkpoint: str,
    cycles: int,
    start_cycle: int,
    initial_victim_adapter: str | None,
    ppo_epochs_per_cycle: int,
    episodes_per_batch: int,
    max_turns: int,
    lr: float,
    start_epoch: int | None,
    seed: int,
    seed_prompt_file: str,
    victim_history_turns: int | None,
    reward_backend: str | None,
    use_template_mutator: bool,
    min_reward_for_victim_training: float | None,
    verify_adapter_update: bool,
    victim_train_steps: int | None,
    victim_train_batch_size: int,
    victim_train_grad_accum_steps: int,
    victim_train_max_length: int,
    wandb_project: str | None,
) -> dict:
    current_policy_checkpoint = initial_policy_checkpoint
    current_victim_adapter = initial_victim_adapter
    next_epoch = start_epoch
    cycle_results = _load_existing_cycles(run_dir) if start_cycle > 1 else []

    for cycle_index in range(start_cycle, cycles + 1):
        print(f"\nStarting PPO cycle {cycle_index}/{cycles}")
        policy_result = train_policy_cycle_remote.remote(
            run_dir=run_dir,
            cycle_index=cycle_index,
            policy_checkpoint=current_policy_checkpoint,
            victim_lora_adapter=current_victim_adapter,
            start_epoch=next_epoch,
            ppo_epochs=ppo_epochs_per_cycle,
            episodes_per_batch=episodes_per_batch,
            max_turns=max_turns,
            lr=lr,
            seed=seed,
            seed_prompt_file=seed_prompt_file,
            victim_history_turns=victim_history_turns,
            reward_backend=reward_backend,
            use_template_mutator=use_template_mutator,
            wandb_project=wandb_project,
        )
        current_policy_checkpoint = policy_result["final_policy_checkpoint"]
        next_epoch = int(policy_result["policy_end_epoch"]) + 1

        print(f"\nStarting victim update {cycle_index}/{cycles}")
        victim_result = train_victim_cycle_remote.remote(
            run_dir=run_dir,
            cycle_index=cycle_index,
            base_adapter_path=current_victim_adapter,
            min_reward_for_training=min_reward_for_victim_training,
            verify_adapter_update=verify_adapter_update,
            victim_train_steps=victim_train_steps,
            victim_train_batch_size=victim_train_batch_size,
            victim_train_grad_accum_steps=victim_train_grad_accum_steps,
            victim_train_max_length=victim_train_max_length,
        )
        current_victim_adapter = victim_result["active_adapter_path"]
        cycle_results.append(
            {
                "cycle_index": cycle_index,
                "policy": policy_result,
                "victim": victim_result,
            }
        )
        summary = _run_summary(
            run_dir=run_dir,
            initial_policy_checkpoint=initial_policy_checkpoint,
            final_policy_checkpoint=current_policy_checkpoint,
            final_victim_adapter=current_victim_adapter,
            cycles=cycle_results,
            config={
                "cycles": cycles,
                "start_cycle": start_cycle,
                "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
                "episodes_per_batch": episodes_per_batch,
                "max_turns": max_turns,
                "lr": lr,
                "start_epoch": start_epoch,
                "seed": seed,
                "seed_prompt_file": seed_prompt_file,
                "victim_history_turns": victim_history_turns,
                "reward_backend": reward_backend,
                "use_template_mutator": use_template_mutator,
                "min_reward_for_victim_training": min_reward_for_victim_training,
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
    print("final victim adapter:", current_victim_adapter)
    print("summary:", f"{run_dir}/summary.json")
    return summary


def _run_summary(
    *,
    run_dir: str,
    initial_policy_checkpoint: str,
    final_policy_checkpoint: str,
    final_victim_adapter: str | None,
    cycles: list[dict],
    config: dict,
) -> dict:
    return {
        "run_dir": run_dir,
        "trajectory_bank_path": f"{run_dir}/trajectory_bank/episodes.jsonl",
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "final_policy_checkpoint": final_policy_checkpoint,
        "final_victim_adapter": final_victim_adapter,
        "cycles": cycles,
        "config": config,
    }


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


def _ppo_update(policy, batch, optimizer, clip_eps=0.2, value_coef=0.5, ent_coef=0.01) -> float:
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
