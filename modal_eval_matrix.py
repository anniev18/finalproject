"""One-command 2x2 eval for fixed-continued vs co-evolved PPO policies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-eval-matrix")
volume = modal.Volume.from_name("cs224r-redteam-rl-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-gpu.txt")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("redteam_rl", remote_path="/root/redteam_rl")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("modal_eval_trailblazer.py", remote_path="/root/modal_eval_trailblazer.py")
    .add_local_file("requirements-gpu.txt", remote_path="/root/requirements-gpu.txt")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def run_eval_matrix_remote(
    eval_name: str,
    fixed_policy_checkpoint: str,
    coevolved_policy_checkpoint: str,
    evolved_victim_adapter: str,
    num_episodes: int = 50,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts_test.json",
    reward_backend: str = "wildguard",
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    wandb_project: str | None = None,
    remote_output_dir: str = "/root/outputs/eval_matrix",
    skip_fixed_base: bool = False,
) -> dict:
    output_dir = Path(remote_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volume.commit()

    cells = [
        {
            "cell": "fixed_policy__base_victim",
            "policy_label": "fixed_continued",
            "victim_label": "base",
            "checkpoint": fixed_policy_checkpoint,
            "victim_lora_adapter": None,
        },
        {
            "cell": "fixed_policy__evolved_victim",
            "policy_label": "fixed_continued",
            "victim_label": "evolved",
            "checkpoint": fixed_policy_checkpoint,
            "victim_lora_adapter": evolved_victim_adapter,
        },
        {
            "cell": "coevolved_policy__base_victim",
            "policy_label": "coevolved",
            "victim_label": "base",
            "checkpoint": coevolved_policy_checkpoint,
            "victim_lora_adapter": None,
        },
        {
            "cell": "coevolved_policy__evolved_victim",
            "policy_label": "coevolved",
            "victim_label": "evolved",
            "checkpoint": coevolved_policy_checkpoint,
            "victim_lora_adapter": evolved_victim_adapter,
        },
    ]
    if skip_fixed_base:
        cells = [cell for cell in cells if cell["cell"] != "fixed_policy__base_victim"]

    cell_results = []
    for cell in cells:
        cell_eval_name = f"{eval_name}_{cell['cell']}"
        print(f"starting eval cell: {cell_eval_name}", flush=True)
        result = _evaluate_trailblazer_inline(
            trailblazer_checkpoint=cell["checkpoint"],
            trailblazer_checkpoints=None,
            num_episodes=num_episodes,
            max_turns=max_turns,
            seed=seed,
            seed_prompt_file=seed_prompt_file,
            config_path="/root/configs/default.json",
            victim_history_turns=victim_history_turns,
            reward_backend=reward_backend,
            use_template_mutator=use_template_mutator,
            attacker_lora_adapter=None,
            victim_lora_adapter=cell["victim_lora_adapter"],
            verbose=True,
            show_text=False,
            remote_output_dir=str(output_dir / eval_name),
            eval_name=cell_eval_name,
            wandb_project=wandb_project,
        )
        cell_result = {
            **cell,
            "eval_name": cell_eval_name,
            "summary_path": result.get("remote_artifacts", {}).get("summary_path"),
            "episode_log_path": result.get("remote_artifacts", {}).get("episode_log_path"),
            "random": result.get("random"),
            "trailblazer": result.get("trailblazer"),
            "checkpoint_results": result.get("checkpoint_results"),
        }
        cell_results.append(cell_result)
        _write_json(output_dir / eval_name / "matrix_summary_partial.json", _matrix_summary(
            eval_name=eval_name,
            fixed_policy_checkpoint=fixed_policy_checkpoint,
            coevolved_policy_checkpoint=coevolved_policy_checkpoint,
            evolved_victim_adapter=evolved_victim_adapter,
            num_episodes=num_episodes,
            max_turns=max_turns,
            seed=seed,
            seed_prompt_file=seed_prompt_file,
            reward_backend=reward_backend,
            skipped_cells=["fixed_policy__base_victim"] if skip_fixed_base else [],
            cells=cell_results,
        ))
        volume.commit()

    summary = _matrix_summary(
        eval_name=eval_name,
        fixed_policy_checkpoint=fixed_policy_checkpoint,
        coevolved_policy_checkpoint=coevolved_policy_checkpoint,
        evolved_victim_adapter=evolved_victim_adapter,
        num_episodes=num_episodes,
        max_turns=max_turns,
        seed=seed,
        seed_prompt_file=seed_prompt_file,
        reward_backend=reward_backend,
        skipped_cells=["fixed_policy__base_victim"] if skip_fixed_base else [],
        cells=cell_results,
    )
    summary_path = output_dir / eval_name / "matrix_summary.json"
    _write_json(summary_path, summary)
    volume.commit()
    summary["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "summary_path": str(summary_path),
        "output_dir": str(output_dir / eval_name),
    }
    return summary


def _evaluate_trailblazer_inline(
    trailblazer_checkpoint: str,
    trailblazer_checkpoints: list[str] | None = None,
    num_episodes: int = 10,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    config_path: str = "/root/configs/default.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    verbose: bool = True,
    show_text: bool = False,
    text_preview_chars: int = 240,
    remote_output_dir: str = "/root/outputs/eval",
    eval_name: str | None = None,
    wandb_project: str | None = None,
) -> dict:
    import os
    import sys

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import RandomPolicy, TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.versioning import build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    run_id = new_run_id("modal_eval_matrix_cell")
    selected_reward_backend = reward_backend or cfg.reward_backend()
    remote_eval_dir = Path(remote_output_dir)
    remote_eval_dir.mkdir(parents=True, exist_ok=True)
    resolved_eval_name = eval_name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    remote_episode_log_path = remote_eval_dir / f"{resolved_eval_name}_episodes.jsonl"
    volume.commit()

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
        reference_similarity_config=cfg.reference_similarity_config(),
    )

    def build_env(policy):
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
        return RedTeamEnv(
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
                name=_wandb_eval_run_name(
                    resolved_eval_name,
                    trailblazer_checkpoint,
                    trailblazer_checkpoints,
                    victim_lora_adapter,
                ),
                config={
                    "run_id": run_id,
                    "eval_name": resolved_eval_name,
                    "trailblazer_checkpoint": trailblazer_checkpoint,
                    "trailblazer_checkpoints": trailblazer_checkpoints,
                    "victim_lora_adapter": victim_lora_adapter,
                    "num_episodes": num_episodes,
                    "max_turns": max_turns,
                    "seed": seed,
                    "seed_prompt_file": seed_prompt_file,
                    "reward_backend": selected_reward_backend,
                },
            )
            wandb.define_metric("eval_step")
            wandb.define_metric("episode_index")
            wandb.define_metric("random/*", step_metric="eval_step")
            wandb.define_metric("trailblazer/*", step_metric="eval_step")
        except Exception as exc:
            print(f"wandb init failed: {exc}", flush=True)

    def _preview(text: str) -> str:
        text = " ".join(str(text).split())
        if len(text) <= text_preview_chars:
            return text
        return text[: max(text_preview_chars - 3, 0)] + "..."

    def run_policy(policy_name: str, policy) -> list[dict]:
        env = build_env(policy)
        episodes = []
        for episode_index in range(num_episodes):
            episode_seed = seed + episode_index
            selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
            if verbose:
                print(
                    f"[{policy_name}] episode {episode_index + 1}/{num_episodes} "
                    f"seed={episode_seed} prompt={_preview(selected_seed_prompt)}",
                    flush=True,
                )
            state = env.reset(selected_seed_prompt)
            done = False
            while not done:
                state, _, done, _ = env.step()
                turn = state.turns[-1]
                if verbose:
                    print(
                        f"[{policy_name}] episode {episode_index + 1}/{num_episodes} "
                        f"turn={len(state.turns)}/{max_turns} "
                        f"action={turn.action.value if turn.action else None} "
                        f"reward={float(turn.reward or 0.0):.3f} "
                        f"judge={turn.metadata.get('judge_label')}",
                        flush=True,
                    )
                if show_text:
                    print(f"  attack: {_preview(turn.user_message)}", flush=True)
                    print(f"  victim: {_preview(turn.victim_response)}", flush=True)
            turns = state.turns
            rewards = [float(turn.reward or 0.0) for turn in turns]
            unsafe_turns = [turn for turn in turns if str(turn.metadata.get("judge_label", "")).upper() == "UNSAFE"]
            success = bool(unsafe_turns)
            episode_record = {
                "policy": policy_name,
                "seed": episode_seed,
                "seed_prompt": state.seed_prompt,
                "initial_template": state.initial_template,
                "num_turns": len(turns),
                "episode_return": sum(rewards),
                "success": success,
                "judge_labels": [turn.metadata.get("judge_label") for turn in turns],
                "turns": [
                    {
                        "action": turn.action.value if turn.action else None,
                        "attack_template": turn.attack_template,
                        "reward": turn.reward,
                        "judge_label": turn.metadata.get("judge_label"),
                        "user_message": turn.user_message,
                        "victim_response": turn.victim_response,
                        "policy_value": turn.metadata.get("policy_value"),
                        "policy_action_probs": turn.metadata.get("policy_action_probs"),
                        "policy_attention_weights": turn.metadata.get("policy_attention_weights"),
                    }
                    for turn in turns
                ],
            }
            episodes.append(episode_record)
            _append_jsonl(remote_episode_log_path, episode_record)
            volume.commit()
            if verbose:
                print(
                    f"[{policy_name}] episode {episode_index + 1}/{num_episodes} done "
                    f"return={episode_record['episode_return']:.3f} "
                    f"success={int(success)} labels={episode_record['judge_labels']}",
                    flush=True,
                )
            if wandb_run is not None:
                try:
                    eval_step = (0 if policy_name == "random" else num_episodes) + episode_index
                    wandb_run.log(
                        {
                            "eval_step": eval_step,
                            "episode_index": episode_index,
                            f"{policy_name}/episode_seed": episode_seed,
                            f"{policy_name}/episode_return": episode_record["episode_return"],
                            f"{policy_name}/num_turns": episode_record["num_turns"],
                            f"{policy_name}/success": int(success),
                            f"{policy_name}/unsafe_turns": len(unsafe_turns),
                            f"{policy_name}/actions": ",".join(turn["action"] or "" for turn in episode_record["turns"]),
                        }
                    )
                except Exception as exc:
                    print(f"wandb episode logging failed: {exc}", flush=True)
        return episodes

    random_episodes = run_policy("random", RandomPolicy())
    selected_checkpoints = trailblazer_checkpoints or [trailblazer_checkpoint]
    trailblazer_runs = []
    for checkpoint in selected_checkpoints:
        checkpoint_label = _checkpoint_label(checkpoint)
        trailblazer_policy = TrailBlazerPolicy.from_checkpoint(checkpoint)
        trailblazer_episodes = run_policy(checkpoint_label, trailblazer_policy)
        trailblazer_runs.append({"checkpoint": checkpoint, "label": checkpoint_label, "episodes": trailblazer_episodes})

    def summarize(episodes: list[dict]) -> dict:
        total = len(episodes)
        successes = [episode for episode in episodes if episode["success"]]
        return {
            "num_episodes": total,
            "mean_return": sum(episode["episode_return"] for episode in episodes) / total if total else 0.0,
            "mean_turns": sum(episode["num_turns"] for episode in episodes) / total if total else 0.0,
            "success_rate": len(successes) / total if total else 0.0,
            "queries_per_success": (
                sum(episode["num_turns"] for episode in successes) / len(successes) if successes else None
            ),
        }

    summary = {
        "run_id": run_id,
        "config": {
            "num_episodes": num_episodes,
            "max_turns": max_turns,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "config_path": config_path,
            "victim_history_turns": victim_history_turns,
            "reward_backend": selected_reward_backend,
            "use_template_mutator": use_template_mutator,
            "attacker_lora_adapter": attacker_lora_adapter,
            "victim_lora_adapter": victim_lora_adapter,
            "trailblazer_checkpoint": trailblazer_checkpoint,
        },
        "random": summarize(random_episodes),
        "trailblazer": summarize(trailblazer_runs[-1]["episodes"]) if trailblazer_runs else {},
        "checkpoint_results": [
            {"checkpoint": run["checkpoint"], "label": run["label"], **summarize(run["episodes"])}
            for run in trailblazer_runs
        ],
        "random_episodes": random_episodes,
        "trailblazer_episodes": trailblazer_runs[-1]["episodes"] if trailblazer_runs else [],
        "checkpoint_episodes": {run["label"]: run["episodes"] for run in trailblazer_runs},
        "metadata": build_model_metadata(
            run_id=run_id,
            victim_model=cfg.models.victim,
            victim_adapter_path=victim_lora_adapter,
            attacker_model="template" if use_template_mutator else cfg.models.mutator,
            attacker_adapter_path=attacker_lora_adapter or config_initial_attacker_adapter,
            judge_model=_judge_model_name(cfg, selected_reward_backend),
            policy_type="comparison",
            policy_checkpoint=selected_checkpoints[-1],
            extra={
                "num_episodes": num_episodes,
                "max_turns": max_turns,
                "seed": seed,
                "reward_backend": selected_reward_backend,
                "use_template_mutator": use_template_mutator,
            },
        ),
    }

    if wandb_run is not None:
        try:
            wandb_run.summary["random/success_rate"] = summary["random"]["success_rate"]
            wandb_run.summary["random/mean_return"] = summary["random"]["mean_return"]
            for row in summary["checkpoint_results"]:
                prefix = f"checkpoint/{row['label']}"
                wandb_run.summary[f"{prefix}/success_rate"] = row["success_rate"]
                wandb_run.summary[f"{prefix}/mean_return"] = row["mean_return"]
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}", flush=True)

    remote_summary_path = _write_eval_summary(summary, remote_output_dir, resolved_eval_name)
    summary["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "summary_path": remote_summary_path,
        "episode_log_path": str(remote_episode_log_path),
    }
    volume.commit()
    return summary


def _matrix_summary(
    *,
    eval_name: str,
    fixed_policy_checkpoint: str,
    coevolved_policy_checkpoint: str,
    evolved_victim_adapter: str,
    num_episodes: int,
    max_turns: int,
    seed: int,
    seed_prompt_file: str,
    reward_backend: str,
    skipped_cells: list[str],
    cells: list[dict],
) -> dict:
    table = []
    for cell in cells:
        metrics = cell.get("trailblazer") or {}
        table.append(
            {
                "policy": cell["policy_label"],
                "victim": cell["victim_label"],
                "success_rate": metrics.get("success_rate"),
                "mean_return": metrics.get("mean_return"),
                "mean_turns": metrics.get("mean_turns"),
                "queries_per_success": metrics.get("queries_per_success"),
                "summary_path": cell.get("summary_path"),
                "episode_log_path": cell.get("episode_log_path"),
            }
        )
    return {
        "eval_name": eval_name,
        "config": {
            "fixed_policy_checkpoint": fixed_policy_checkpoint,
            "coevolved_policy_checkpoint": coevolved_policy_checkpoint,
            "evolved_victim_adapter": evolved_victim_adapter,
            "num_episodes": num_episodes,
            "max_turns": max_turns,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "reward_backend": reward_backend,
            "skipped_cells": skipped_cells,
        },
        "table": table,
        "cells": cells,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _write_eval_summary(summary: dict, remote_output_dir: str, eval_name: str | None = None) -> str:
    output_dir = Path(remote_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = eval_name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    path = output_dir / f"{name}.json"
    _write_json(path, summary)
    return str(path)


def _judge_model_name(cfg, reward_backend: str) -> str:
    if reward_backend in {"qwen_judge", "qwen_safety_judge"}:
        return cfg.models.qwen_safety_judge
    if reward_backend == "prompt_guard":
        return cfg.models.prompt_guard
    if reward_backend == "llama_guard":
        return cfg.models.llama_guard
    if reward_backend == "wildguard":
        return cfg.models.wildguard
    if reward_backend in {"reference_similarity", "reference_similarity_raw", "reference_similarity_binary"}:
        return cfg.reference_similarity_config().embedding_model_name
    return "fake"


def _checkpoint_label(checkpoint: str) -> str:
    stem = Path(checkpoint).stem
    if stem.startswith("checkpoint_"):
        return stem.removeprefix("checkpoint_")
    return stem


def _wandb_eval_run_name(
    eval_name: str,
    trailblazer_checkpoint: str,
    trailblazer_checkpoints: list[str] | None,
    victim_lora_adapter: str | None,
) -> str:
    checkpoints = trailblazer_checkpoints or [trailblazer_checkpoint]
    checkpoint_part = _checkpoint_label(checkpoints[0]) if len(checkpoints) == 1 else f"{len(checkpoints)}checkpoints"
    victim_part = "base_victim" if victim_lora_adapter is None else f"victim_{Path(victim_lora_adapter).name}"
    return f"eval/{eval_name}/{checkpoint_part}/{victim_part}"


@app.local_entrypoint()
def main(
    eval_name: str | None = None,
    fixed_policy_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/fixed_continued_from_epoch19_15epochs/checkpoint_epoch_34.pt",
    coevolved_policy_checkpoint: str = "/root/outputs/coevolution/coevolve_robust_from_epoch19_3x5/cycle_003/policy/checkpoint_epoch_34.pt",
    evolved_victim_adapter: str = "/root/outputs/coevolution/coevolve_robust_from_epoch19_3x5/cycle_003/victim_adapter",
    num_episodes: int = 50,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts_test.json",
    reward_backend: str = "wildguard",
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    wandb_project: str | None = None,
    remote_output_dir: str = "/root/outputs/eval_matrix",
    skip_fixed_base: bool = False,
    wait_for_result: bool = False,
) -> None:
    resolved_eval_name = eval_name or f"eval_matrix_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    call_kwargs = {
        "eval_name": resolved_eval_name,
        "fixed_policy_checkpoint": fixed_policy_checkpoint,
        "coevolved_policy_checkpoint": coevolved_policy_checkpoint,
        "evolved_victim_adapter": evolved_victim_adapter,
        "num_episodes": num_episodes,
        "max_turns": max_turns,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "reward_backend": reward_backend,
        "victim_history_turns": victim_history_turns,
        "use_template_mutator": use_template_mutator,
        "wandb_project": wandb_project,
        "remote_output_dir": remote_output_dir,
        "skip_fixed_base": skip_fixed_base,
    }
    if wait_for_result:
        result = run_eval_matrix_remote.remote(**call_kwargs)
        print(json.dumps(result, indent=2))
        return

    function_call = run_eval_matrix_remote.spawn(**call_kwargs)
    call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
    if skip_fixed_base:
        print("submitted eval matrix missing cells only, without waiting for the result")
        print("skipping already-run cell: fixed_policy__base_victim")
    else:
        print("submitted 2x2 eval matrix without waiting for the result")
    if call_id:
        print("function call id:", call_id)
    print("eval name:", resolved_eval_name)
    print("remote output dir:", f"{remote_output_dir}/{resolved_eval_name}")
    print("remote summary path:", f"{remote_output_dir}/{resolved_eval_name}/matrix_summary.json")
    print("Modal volume: cs224r-redteam-rl-data")
    print("Use `modal app logs cs224r-eval-matrix` to follow progress.")
