"""Modal entrypoint for comparing RandomPolicy vs TrailBlazerPolicy.

This wrapper runs both policies on the same set of seeds and returns a compact
summary with return, turn-count, and success metrics. By default it mirrors the
real episode/training stack with the LLM mutator; pass ``--use-template-mutator``
for cheaper checks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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
    timeout=60 * 60 * 5,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def evaluate_trailblazer_remote(
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
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import sys

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
    run_id = new_run_id("modal_trailblazer_eval")
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

    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name=run_id)
            wandb.define_metric("eval_step")
            wandb.define_metric("episode_index")
            wandb.define_metric("random/*", step_metric="eval_step")
            wandb.define_metric("trailblazer/*", step_metric="eval_step")
        except Exception as exc:
            print(f"wandb init failed: {exc}")

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
            _append_remote_eval_episode(remote_episode_log_path, episode_record)
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
                            f"{policy_name}/actions": ",".join(
                                turn["action"] or "" for turn in episode_record["turns"]
                            ),
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
        trailblazer_runs.append(
            {
                "checkpoint": checkpoint,
                "label": checkpoint_label,
                "episodes": trailblazer_episodes,
            }
        )

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
            {
                "checkpoint": run["checkpoint"],
                "label": run["label"],
                **summarize(run["episodes"]),
            }
            for run in trailblazer_runs
        ],
        "checkpoint_ranking": sorted(
            [
                {
                    "checkpoint": run["checkpoint"],
                    "label": run["label"],
                    **summarize(run["episodes"]),
                }
                for run in trailblazer_runs
            ],
            key=lambda row: (row["success_rate"], row["mean_return"]),
            reverse=True,
        ),
        "random_episodes": random_episodes,
        "trailblazer_episodes": trailblazer_runs[-1]["episodes"] if trailblazer_runs else [],
        "checkpoint_episodes": {
            run["label"]: run["episodes"]
            for run in trailblazer_runs
        },
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
            wandb_run.summary["random/mean_turns"] = summary["random"]["mean_turns"]
            for row in summary["checkpoint_results"]:
                prefix = f"checkpoint/{row['label']}"
                wandb_run.summary[f"{prefix}/success_rate"] = row["success_rate"]
                wandb_run.summary[f"{prefix}/mean_return"] = row["mean_return"]
                wandb_run.summary[f"{prefix}/mean_turns"] = row["mean_turns"]
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    remote_summary_path = _write_remote_eval_summary(summary, remote_output_dir, resolved_eval_name)
    summary["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "summary_path": remote_summary_path,
        "episode_log_path": str(remote_episode_log_path),
    }
    volume.commit()
    return summary


def _append_remote_eval_episode(path: Path, episode_record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(episode_record) + "\n")


def _write_remote_eval_summary(summary: dict, remote_output_dir: str, eval_name: str | None = None) -> str:
    output_dir = Path(remote_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = eval_name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    path = output_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
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
    return "fake"


def _checkpoint_label(checkpoint: str) -> str:
    stem = Path(checkpoint).stem
    if stem.startswith("checkpoint_"):
        return stem.removeprefix("checkpoint_")
    return stem


def _parse_checkpoint_epochs(checkpoint_dir: str, checkpoint_epochs: str | None) -> list[str]:
    if not checkpoint_epochs:
        return []
    checkpoints = []
    for raw_epoch in checkpoint_epochs.split(","):
        epoch = raw_epoch.strip()
        if not epoch:
            continue
        checkpoints.append(str(Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"))
    return checkpoints


def _print_checkpoint_ranking(result: dict) -> None:
    ranking = result.get("checkpoint_ranking", [])
    if not ranking:
        return
    print("\nCheckpoint ranking")
    print("=" * 96)
    print(f"{'rank':<5} {'label':<16} {'success':<10} {'mean_return':<12} {'mean_turns':<11} checkpoint")
    for index, row in enumerate(ranking, start=1):
        print(
            f"{index:<5} "
            f"{str(row.get('label', '')):<16} "
            f"{float(row.get('success_rate', 0.0)):<10.3f} "
            f"{float(row.get('mean_return', 0.0)):<12.3f} "
            f"{float(row.get('mean_turns', 0.0)):<11.3f} "
            f"{row.get('checkpoint')}"
        )


@app.local_entrypoint()
def main(
    trailblazer_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_9.pt",
    checkpoint_dir: str | None = None,
    checkpoint_epochs: str | None = None,
    eval_name: str | None = None,
    remote_output_dir: str = "/root/outputs/eval",
    wait_for_result: bool = False,
    num_episodes: int = 10,
    max_turns: int = 3,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    victim_history_turns: int | None = None,
    reward_backend: str | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    verbose: bool = True,
    show_text: bool = False,
    text_preview_chars: int = 240,
    wandb_project: str | None = None,
    output_dir: str = "outputs/eval",
    save_local: bool = False,
) -> None:
    sweep_checkpoints = (
        _parse_checkpoint_epochs(checkpoint_dir, checkpoint_epochs)
        if checkpoint_dir and checkpoint_epochs
        else None
    )
    selected_checkpoint = sweep_checkpoints[0] if sweep_checkpoints else trailblazer_checkpoint
    resolved_eval_name = eval_name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    call_kwargs = {
        "trailblazer_checkpoint": selected_checkpoint,
        "trailblazer_checkpoints": sweep_checkpoints,
        "num_episodes": num_episodes,
        "max_turns": max_turns,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "config_path": "/root/configs/default.json",
        "victim_history_turns": victim_history_turns,
        "reward_backend": reward_backend,
        "use_template_mutator": use_template_mutator,
        "attacker_lora_adapter": attacker_lora_adapter,
        "victim_lora_adapter": victim_lora_adapter,
        "verbose": verbose,
        "show_text": show_text,
        "text_preview_chars": text_preview_chars,
        "remote_output_dir": remote_output_dir,
        "eval_name": resolved_eval_name,
        "wandb_project": wandb_project,
    }
    if not wait_for_result:
        function_call = evaluate_trailblazer_remote.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted TrailBlazer eval without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        if checkpoint_dir and checkpoint_epochs:
            print("checkpoint dir:", checkpoint_dir)
            print("checkpoint epochs:", checkpoint_epochs)
        else:
            print("checkpoint:", selected_checkpoint)
        print("remote output dir:", remote_output_dir)
        print("remote summary path:", f"{remote_output_dir}/{resolved_eval_name}.json")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-trailblazer-eval` to follow progress.")
        return

    result = evaluate_trailblazer_remote.remote(**call_kwargs)
    _print_checkpoint_ranking(result)
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
