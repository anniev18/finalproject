"""Modal entrypoint for PPO/victim co-evolution.

This runs an alternating loop:
  1. Continue TrailBlazer PPO against the current victim.
  2. Append that cycle's episodes to a run-local trajectory bank.
  3. Train the next victim LoRA adapter from cumulative successful attacks.

The run-local bank keeps the experiment boundary clean while still letting the
victim train cumulatively on successes from all previous cycles in this run.
"""

from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-coevolution")
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
    timeout=60 * 60 * 8,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def run_coevolution_remote(
    run_dir: str,
    initial_policy_checkpoint: str,
    cycles: int = 3,
    ppo_epochs_per_cycle: int = 5,
    episodes_per_batch: int = 16,
    max_turns: int = 3,
    lr: float = 3e-4,
    start_epoch: int | None = None,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    config_path: str = "/root/configs/default.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    min_reward_for_victim_training: float | None = None,
    verify_adapter_update: bool = False,
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
    from redteam_rl.orchestration import (
        VictimEvolutionConfig,
        annotate_episode_state,
        build_safety_examples,
        verify_adapter_update as verify_adapter_artifact,
    )
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.trajectory_bank import append_episodes
    from redteam_rl.types import EpisodeState
    from redteam_rl.versioning import adapter_version, build_model_metadata, new_run_id
    from redteam_rl.victim_training import train_victim_from_episodes
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    selected_reward_backend = reward_backend or cfg.reward_backend()
    output_root = Path(run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_bank_path = output_root / "trajectory_bank" / "episodes.jsonl"
    run_id = new_run_id("modal_coevolution")
    selected_min_reward = (
        min_reward_for_victim_training
        if min_reward_for_victim_training is not None
        else cfg.victim_evolution_config().min_reward_for_training
    )
    victim_evolution_config = VictimEvolutionConfig(
        min_reward_for_training=selected_min_reward,
        safety_target_file="/root/data/safety_target_templates.json",
        trajectory_bank_path=str(trajectory_bank_path),
        verify_adapter_update=verify_adapter_update,
    )

    policy = TrailBlazerPolicy.from_checkpoint(initial_policy_checkpoint, cfg=cfg.state_feature_config())
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    initial_checkpoint = torch.load(initial_policy_checkpoint, map_location=policy.device)
    if initial_checkpoint.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(initial_checkpoint["optimizer_state"])
        except ValueError:
            pass
    initial_epoch = _checkpoint_epoch(initial_checkpoint, initial_policy_checkpoint)
    next_epoch = start_epoch if start_epoch is not None else ((initial_epoch + 1) if initial_epoch is not None else 0)

    current_victim_adapter: str | None = None
    cumulative_episodes: list[EpisodeState] = []
    cycle_summaries: list[dict] = []

    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                name=Path(run_dir).name,
                resume="never",
                config={
                    "run_dir": run_dir,
                    "initial_policy_checkpoint": initial_policy_checkpoint,
                    "cycles": cycles,
                    "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
                    "episodes_per_batch": episodes_per_batch,
                    "max_turns": max_turns,
                    "lr": lr,
                    "seed": seed,
                    "seed_prompt_file": seed_prompt_file,
                    "reward_backend": selected_reward_backend,
                    "min_reward_for_victim_training": selected_min_reward,
                },
            )
        except Exception as exc:
            print(f"wandb init failed: {exc}", flush=True)

    for cycle_index in range(1, cycles + 1):
        input_policy_checkpoint = (
            initial_policy_checkpoint
            if cycle_index == 1
            else cycle_summaries[-1]["final_policy_checkpoint"]
        )
        input_victim_adapter = current_victim_adapter
        cycle_dir = output_root / f"cycle_{cycle_index:03d}"
        policy_dir = cycle_dir / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        cycle_episode_log_path = cycle_dir / "training_episodes.jsonl"
        cycle_summary_path = cycle_dir / "summary.json"

        env = _build_env(
            cfg=cfg,
            policy=policy,
            max_turns=max_turns,
            victim_history_turns=victim_history_turns,
            selected_reward_backend=selected_reward_backend,
            use_template_mutator=use_template_mutator,
            victim_lora_adapter=current_victim_adapter,
        )

        cycle_rewards: list[float] = []
        cycle_episodes: list[EpisodeState] = []
        epoch_summaries: list[dict] = []
        cycle_start_epoch = next_epoch
        for epoch in range(cycle_start_epoch, cycle_start_epoch + ppo_epochs_per_cycle):
            batch = _collect_ppo_batch(
                env=env,
                seed_prompt_file=seed_prompt_file,
                seed=seed + epoch * episodes_per_batch,
                episodes_per_batch=episodes_per_batch,
                epoch=epoch,
                cycle_index=cycle_index,
                episode_log_path=cycle_episode_log_path,
                metadata=build_model_metadata(
                    run_id=run_id,
                    victim_model=cfg.models.victim,
                    victim_adapter_path=current_victim_adapter,
                    attacker_model="template" if use_template_mutator else cfg.models.mutator,
                    judge_model=_judge_model_name(cfg, selected_reward_backend),
                    policy_type="trailblazer_coevolution",
                    policy_checkpoint=initial_policy_checkpoint,
                    extra={
                        "cycle_index": cycle_index,
                        "epoch": epoch,
                        "reward_backend": selected_reward_backend,
                    },
                ),
            )
            loss_value = _ppo_update(policy, batch, optimizer)
            checkpoint_path = policy_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_policy_checkpoint(
                policy=policy,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                trainer_state={
                    "run_id": run_id,
                    "cycle_index": cycle_index,
                    "initial_policy_checkpoint": initial_policy_checkpoint,
                    "victim_lora_adapter": current_victim_adapter,
                    "lr": lr,
                    "episodes_per_batch": episodes_per_batch,
                    "max_turns": max_turns,
                    "seed_prompt_file": seed_prompt_file,
                    "reward_backend": selected_reward_backend,
                },
            )
            batch_episodes = [episode["final_state"] for episode in batch]
            cycle_episodes.extend(batch_episodes)
            for episode in batch_episodes:
                for turn in episode.turns:
                    cycle_rewards.append(float(turn.reward or 0.0))
            epoch_summary = {
                "epoch": epoch,
                "loss": loss_value,
                "checkpoint_path": str(checkpoint_path),
                "episodes": len(batch_episodes),
                "average_reward": sum(cycle_rewards) / len(cycle_rewards) if cycle_rewards else 0.0,
                "max_reward": max(cycle_rewards) if cycle_rewards else 0.0,
            }
            epoch_summaries.append(epoch_summary)
            print(
                f"cycle={cycle_index} epoch={epoch} loss={loss_value:.6f} "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "cycle": cycle_index,
                            "epoch": epoch,
                            "loss": loss_value,
                            "cycle/average_reward_so_far": epoch_summary["average_reward"],
                            "cycle/max_reward_so_far": epoch_summary["max_reward"],
                        },
                        step=epoch,
                    )
                except Exception:
                    pass

        next_epoch = cycle_start_epoch + ppo_epochs_per_cycle
        cumulative_episodes.extend(cycle_episodes)
        append_episodes(
            cycle_episodes,
            path=trajectory_bank_path,
            metadata={
                "source": "modal_coevolve.py",
                "run_id": run_id,
                "cycle_index": cycle_index,
                "victim_adapter_path": current_victim_adapter,
            },
        )
        _release_env(env)
        env = None

        victim_adapter_dir = cycle_dir / "victim_adapter"
        victim_training_config = cfg.victim_finetune_config(
            output_dir=victim_adapter_dir,
            base_adapter_path=current_victim_adapter,
        )
        examples = build_safety_examples(cumulative_episodes, victim_evolution_config)
        adapter_path: str | None = None
        adapter_verification = None
        if examples:
            adapter_path = str(
                train_victim_from_episodes(
                    cumulative_episodes,
                    victim_training_config,
                    min_reward=selected_min_reward,
                )
            )
            _release_cuda_memory()
            current_victim_adapter = adapter_path
            if verify_adapter_update:
                adapter_verification = verify_adapter_artifact(adapter_path)
        else:
            current_victim_adapter = None

        _write_json(
            cycle_dir / "victim_training_examples.json",
            [asdict(example) for example in examples],
        )

        cycle_summary = {
            "cycle_index": cycle_index,
            "input_policy_checkpoint": input_policy_checkpoint,
            "input_victim_adapter": input_victim_adapter,
            "policy_start_epoch": cycle_start_epoch,
            "policy_end_epoch": next_epoch - 1,
            "policy_dir": str(policy_dir),
            "final_policy_checkpoint": str(policy_dir / f"checkpoint_epoch_{next_epoch - 1}.pt"),
            "cycle_episodes": len(cycle_episodes),
            "cumulative_episodes": len(cumulative_episodes),
            "cycle_turns": sum(len(episode.turns) for episode in cycle_episodes),
            "cumulative_turns": sum(len(episode.turns) for episode in cumulative_episodes),
            "training_examples": len(examples),
            "victim_adapter_path": adapter_path,
            "victim_version": adapter_version("victim", adapter_path),
            "adapter_verification": adapter_verification,
            "average_reward": sum(cycle_rewards) / len(cycle_rewards) if cycle_rewards else 0.0,
            "max_reward": max(cycle_rewards) if cycle_rewards else 0.0,
            "epochs": epoch_summaries,
        }
        _write_json(cycle_summary_path, cycle_summary)
        cycle_summaries.append(cycle_summary)
        volume.commit()

    final_summary = {
        "run_id": run_id,
        "run_dir": str(output_root),
        "trajectory_bank_path": str(trajectory_bank_path),
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "final_policy_checkpoint": cycle_summaries[-1]["final_policy_checkpoint"] if cycle_summaries else None,
        "final_victim_adapter": current_victim_adapter,
        "cycles": cycle_summaries,
        "config": {
            "cycles": cycles,
            "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
            "episodes_per_batch": episodes_per_batch,
            "max_turns": max_turns,
            "lr": lr,
            "start_epoch": start_epoch,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "config_path": config_path,
            "victim_history_turns": victim_history_turns,
            "reward_backend": selected_reward_backend,
            "use_template_mutator": use_template_mutator,
            "min_reward_for_victim_training": selected_min_reward,
        },
    }
    _write_json(output_root / "summary.json", final_summary)
    if wandb_run is not None:
        try:
            wandb_run.summary["final_policy_checkpoint"] = final_summary["final_policy_checkpoint"]
            wandb_run.summary["final_victim_adapter"] = final_summary["final_victim_adapter"]
            wandb_run.finish()
        except Exception:
            pass
    volume.commit()
    return final_summary


def _build_env(
    *,
    cfg,
    policy,
    max_turns: int,
    victim_history_turns: int | None,
    selected_reward_backend: str,
    use_template_mutator: bool,
    victim_lora_adapter: str | None,
):
    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

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
    return RedTeamEnv(
        config=cfg.env_config(
            max_turns=max_turns,
            **({"victim_history_turns": victim_history_turns} if victim_history_turns is not None else {}),
        ),
        attacker=attacker,
        victim=victim,
        reward_model=reward_model,
    )


def _collect_ppo_batch(
    *,
    env,
    seed_prompt_file: str,
    seed: int,
    episodes_per_batch: int,
    epoch: int,
    cycle_index: int,
    episode_log_path: Path,
    metadata: dict[str, object],
) -> list[dict]:
    import copy

    from redteam_rl.actions import ACTIONS
    from redteam_rl.orchestration import annotate_episode_state
    from redteam_rl.seed_prompts import sample_seed_prompt

    batch = []
    for episode_index in range(episodes_per_batch):
        episode_seed = seed + episode_index
        selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
        state = env.reset(selected_seed_prompt)
        done = False
        episode = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "final_state": None,
        }
        while not done:
            episode["states"].append(copy.deepcopy(state))
            state, reward_value, done, info = env.step()
            latest_turn = state.turns[-1]
            episode["actions"].append(ACTIONS.index(latest_turn.action))
            episode["log_probs"].append(float(info.get("policy_log_prob", 0.0)))
            episode["values"].append(float(info.get("policy_value", 0.0)))
            episode["rewards"].append(float(reward_value))
        annotate_episode_state(state, metadata)
        episode["final_state"] = copy.deepcopy(state)
        _append_training_episode(
            path=episode_log_path,
            epoch=epoch,
            cycle_index=cycle_index,
            episode_index=episode_index,
            episode_seed=episode_seed,
            state=state,
        )
        batch.append(episode)
    return batch


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


def _append_training_episode(
    *,
    path: Path,
    epoch: int,
    cycle_index: int,
    episode_index: int,
    episode_seed: int,
    state,
) -> None:
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


def _release_env(env) -> None:
    _shutdown_vllm_engines(env)
    del env
    _release_cuda_memory()


def _shutdown_vllm_engines(obj, seen: set[int] | None = None) -> None:
    if obj is None:
        return
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    llm = getattr(obj, "llm", None)
    if llm is not None:
        _shutdown_single_llm(llm)

    for attr in ("attacker", "mutator", "victim", "reward_model"):
        child = getattr(obj, attr, None)
        if child is not None:
            _shutdown_vllm_engines(child, seen)


def _shutdown_single_llm(llm) -> None:
    for candidate in (
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core", None),
    ):
        if candidate is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    initial_policy_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_19.pt",
    remote_output_dir: str = "/root/outputs/coevolution",
    wait_for_result: bool = False,
    cycles: int = 3,
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
    wandb_project: str | None = None,
) -> None:
    resolved_run_name = run_name or f"coevolve_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = str(Path(remote_output_dir) / resolved_run_name)
    call_kwargs = {
        "run_dir": run_dir,
        "initial_policy_checkpoint": initial_policy_checkpoint,
        "cycles": cycles,
        "ppo_epochs_per_cycle": ppo_epochs_per_cycle,
        "episodes_per_batch": episodes_per_batch,
        "max_turns": max_turns,
        "lr": lr,
        "start_epoch": start_epoch,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "config_path": "/root/configs/default.json",
        "victim_history_turns": victim_history_turns,
        "reward_backend": reward_backend,
        "use_template_mutator": use_template_mutator,
        "min_reward_for_victim_training": min_reward_for_victim_training,
        "verify_adapter_update": verify_adapter_update,
        "wandb_project": wandb_project,
    }
    if not wait_for_result:
        function_call = run_coevolution_remote.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted co-evolution run without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        print("run name:", resolved_run_name)
        print("remote run dir:", run_dir)
        print("remote summary path:", f"{run_dir}/summary.json")
        print("remote trajectory bank:", f"{run_dir}/trajectory_bank/episodes.jsonl")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-coevolution` to follow progress.")
        return

    result = run_coevolution_remote.remote(**call_kwargs)
    print(json.dumps(result, indent=2))
    print("\nsaved co-evolution artifacts to Modal volume cs224r-redteam-rl-data")
    print("run name:", resolved_run_name)
    print("remote run dir:", run_dir)
    print("remote summary path:", f"{run_dir}/summary.json")
