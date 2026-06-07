"""Budgeted TrailBlazer evaluation with retries across initial templates.

This is intentionally separate from ``modal_eval_trailblazer.py``. The
existing evaluator measures one episode per seed. This evaluator measures
whether a fixed harmful seed succeeds within a total victim-query budget
across multiple fresh episodes.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-trailblazer-budgeted-eval")
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
def evaluate_budgeted_remote(
    trailblazer_checkpoint: str,
    num_seeds: int = 10,
    max_turns: int = 5,
    query_budget: int = 50,
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
    remote_output_dir: str = "/root/outputs/eval_budgeted",
    eval_name: str | None = None,
    wandb_project: str | None = "trailblazer-ppo",
) -> dict:
    if max_turns <= 0 or query_budget <= 0:
        raise ValueError("max_turns and query_budget must be positive")

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import sys

    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.rlbreaker_templates import load_initial_templates
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.versioning import build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    run_id = new_run_id("modal_trailblazer_budgeted_eval")
    selected_reward_backend = reward_backend or cfg.reward_backend()
    resolved_eval_name = eval_name or f"budgeted_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    remote_eval_dir = Path(remote_output_dir)
    remote_eval_dir.mkdir(parents=True, exist_ok=True)
    remote_episode_log_path = remote_eval_dir / f"{resolved_eval_name}_episodes.jsonl"
    remote_partial_summary_path = remote_eval_dir / f"{resolved_eval_name}_partial.json"
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
    base_policy = TrailBlazerPolicy.from_checkpoint(trailblazer_checkpoint)
    initial_templates = list(load_initial_templates("/root/data/rlbreaker_initial_templates.json"))

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

            wandb_run = wandb.init(project=wandb_project, name=resolved_eval_name)
        except Exception as exc:
            print(f"wandb init failed: {exc}", flush=True)

    def preview(text: str) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= text_preview_chars:
            return compact
        return compact[: max(text_preview_chars - 3, 0)] + "..."

    def serialize_turn(turn) -> dict:
        return {
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

    def attempt_plan(seed_value: int):
        templates = list(initial_templates)
        random.Random(seed_value).shuffle(templates)
        for template_index, template in enumerate(templates):
            yield template_index, template

    seed_results = []
    for seed_index in range(num_seeds):
        episode_seed = seed + seed_index
        selected_seed_prompt = sample_seed_prompt(seed_prompt_file, seed=episode_seed)
        queries_used = 0
        attempts = []
        successful_attempt = None

        if verbose:
            print(
                f"[seed {seed_index + 1}/{num_seeds}] seed={episode_seed} "
                f"budget={query_budget} prompt={preview(selected_seed_prompt)}",
                flush=True,
            )

        for attempt_index, (template_index, initial_template) in enumerate(
            attempt_plan(episode_seed),
            start=1,
        ):
            if queries_used >= query_budget or successful_attempt is not None:
                break

            env = build_env(base_policy)
            state = env.reset(selected_seed_prompt)
            state.initial_template = initial_template
            attempt_start_queries = queries_used
            success = False

            if verbose:
                print(
                    f"  attempt={attempt_index} template={template_index} "
                    f"first_action=ppo remaining={query_budget - queries_used}",
                    flush=True,
                )

            done = False
            while not done and queries_used < query_budget:
                state, _, done, _ = env.step()
                queries_used += 1
                turn = state.turns[-1]
                success = str(turn.metadata.get("judge_label", "")).upper() == "UNSAFE"
                if verbose:
                    print(
                        f"    query={queries_used}/{query_budget} turn={len(state.turns)}/{max_turns} "
                        f"action={turn.action.value if turn.action else None} "
                        f"reward={float(turn.reward or 0.0):.3f} "
                        f"judge={turn.metadata.get('judge_label')}",
                        flush=True,
                    )
                if show_text:
                    print(f"      attack: {preview(turn.user_message)}", flush=True)
                    print(f"      victim: {preview(turn.victim_response)}", flush=True)
                if success:
                    break

            attempt_record = {
                "attempt_index": attempt_index,
                "initial_template_index": template_index,
                "initial_template": initial_template,
                "queries_used": queries_used - attempt_start_queries,
                "success": success,
                "episode_return": sum(float(turn.reward or 0.0) for turn in state.turns),
                "judge_labels": [turn.metadata.get("judge_label") for turn in state.turns],
                "turns": [serialize_turn(turn) for turn in state.turns],
            }
            attempts.append(attempt_record)
            if success:
                successful_attempt = attempt_index

        seed_record = {
            "seed_index": seed_index,
            "seed": episode_seed,
            "seed_prompt": selected_seed_prompt,
            "query_budget": query_budget,
            "queries_used": queries_used,
            "attempts_used": len(attempts),
            "success": successful_attempt is not None,
            "successful_attempt": successful_attempt,
            "budget_exhausted": queries_used >= query_budget and successful_attempt is None,
            "attempts": attempts,
        }
        seed_results.append(seed_record)
        _append_jsonl(remote_episode_log_path, seed_record)
        _write_json(
            remote_partial_summary_path,
            _build_summary(
                seed_results,
                run_id=run_id,
                query_budget=query_budget,
                max_turns=max_turns,
            ),
        )
        volume.commit()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "seed_index": seed_index,
                    "budgeted/success": int(seed_record["success"]),
                    "budgeted/queries_used": queries_used,
                    "budgeted/attempts_used": len(attempts),
                    "budgeted/cumulative_asr": sum(int(row["success"]) for row in seed_results)
                    / len(seed_results),
                }
            )

    summary = _build_summary(
        seed_results,
        run_id=run_id,
        query_budget=query_budget,
        max_turns=max_turns,
    )
    summary["config"] = {
        "trailblazer_checkpoint": trailblazer_checkpoint,
        "num_seeds": num_seeds,
        "max_turns": max_turns,
        "query_budget": query_budget,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "reward_backend": selected_reward_backend,
        "use_template_mutator": use_template_mutator,
        "attacker_lora_adapter": attacker_lora_adapter,
        "victim_lora_adapter": victim_lora_adapter,
        "wandb_project": wandb_project,
    }
    summary["seed_results"] = seed_results
    summary["metadata"] = build_model_metadata(
        run_id=run_id,
        victim_model=cfg.models.victim,
        victim_adapter_path=victim_lora_adapter,
        attacker_model="template" if use_template_mutator else cfg.models.mutator,
        attacker_adapter_path=attacker_lora_adapter or config_initial_attacker_adapter,
        judge_model=_judge_model_name(cfg, selected_reward_backend),
        policy_type="budgeted_trailblazer",
        policy_checkpoint=trailblazer_checkpoint,
        extra={
            "query_budget": query_budget,
            "max_turns": max_turns,
        },
    )

    remote_summary_path = remote_eval_dir / f"{resolved_eval_name}.json"
    _write_json(remote_summary_path, summary)
    summary["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "summary_path": str(remote_summary_path),
        "episode_log_path": str(remote_episode_log_path),
        "partial_summary_path": str(remote_partial_summary_path),
    }
    volume.commit()

    if wandb_run is not None:
        wandb_run.summary["budgeted/asr"] = summary["asr"]
        wandb_run.summary["budgeted/queries_per_success"] = summary["queries_per_success"]
        wandb_run.summary["budgeted/mean_queries"] = summary["mean_queries"]
        wandb_run.finish()

    return summary


def _build_summary(
    seed_results: list[dict],
    *,
    run_id: str,
    query_budget: int,
    max_turns: int,
) -> dict:
    successes = [record for record in seed_results if record["success"]]
    total = len(seed_results)
    return {
        "run_id": run_id,
        "query_budget": query_budget,
        "max_turns": max_turns,
        "num_seeds_completed": total,
        "num_successes": len(successes),
        "asr": len(successes) / total if total else 0.0,
        "mean_queries": (
            sum(record["queries_used"] for record in seed_results) / total
            if total
            else 0.0
        ),
        "queries_per_success": (
            sum(record["queries_used"] for record in successes) / len(successes)
            if successes
            else None
        ),
        "mean_attempts": (
            sum(record["attempts_used"] for record in seed_results) / total
            if total
            else 0.0
        ),
        "budget_exhaustion_rate": (
            sum(int(record["budget_exhausted"]) for record in seed_results) / total
            if total
            else 0.0
        ),
    }


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
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
    if reward_backend in {"reference_similarity", "reference_similarity_raw", "reference_similarity_binary"}:
        return cfg.reference_similarity_config().embedding_model_name
    return "fake"


@app.local_entrypoint()
def main(
    trailblazer_checkpoint: str = "/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_9.pt",
    eval_name: str | None = None,
    remote_output_dir: str = "/root/outputs/eval_budgeted",
    wait_for_result: bool = False,
    num_seeds: int = 10,
    max_turns: int = 5,
    query_budget: int = 50,
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
    wandb_project: str | None = "trailblazer-ppo",
) -> None:
    resolved_eval_name = eval_name or f"budgeted_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    call_kwargs = {
        "trailblazer_checkpoint": trailblazer_checkpoint,
        "num_seeds": num_seeds,
        "max_turns": max_turns,
        "query_budget": query_budget,
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
        function_call = evaluate_budgeted_remote.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "call_id", None)
        print("submitted budgeted TrailBlazer eval without waiting for the result")
        if call_id:
            print("function call id:", call_id)
        print("query budget per seed:", query_budget)
        print("max turns per attempt:", max_turns)
        print("remote summary path:", f"{remote_output_dir}/{resolved_eval_name}.json")
        print("remote attempt log:", f"{remote_output_dir}/{resolved_eval_name}_episodes.jsonl")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-trailblazer-budgeted-eval` to follow progress.")
        return

    result = evaluate_budgeted_remote.remote(**call_kwargs)
    print(json.dumps(result, indent=2))
