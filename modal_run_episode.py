"""Modal entrypoint for running a multi-turn red-team episode on GPU.

Run from the repo root:

    modal run modal_run_episode.py

By default this loads the Qwen LLM mutator, Qwen victim, and Qwen safety judge.
Use `--use-template-mutator` for a cheaper deterministic mutator smoke test.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import modal


app = modal.App("cs224r-redteam-rl")
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
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def run_episode_remote(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    victim_history_turns: int | None = None,
    config_path: str = "/root/configs/default.json",
    use_template_mutator: bool = False,
    reward_backend: str | None = None,
    aux_reward_backends: list[str] | None = None,
    show_mutator_input: bool = False,
    show_victim_input: bool = False,
    attacker_lora_adapter: str | None = None,
    policy_checkpoint: str | None = None,
    victim_lora_adapter: str | None = None,
    remote_output_dir: str = "/root/outputs/modal_episodes",
    remote_trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
    wandb_project: str | None = None,
) -> dict:
    return _run_episode_impl(
        seed_prompt=seed_prompt,
        seed_prompt_file=seed_prompt_file,
        seed=seed,
        max_turns=max_turns,
        victim_history_turns=victim_history_turns,
        config_path=config_path,
        use_template_mutator=use_template_mutator,
        reward_backend=reward_backend,
        aux_reward_backends=aux_reward_backends,
        show_mutator_input=show_mutator_input,
        show_victim_input=show_victim_input,
        attacker_lora_adapter=attacker_lora_adapter,
        policy_checkpoint=policy_checkpoint,
        victim_lora_adapter=victim_lora_adapter,
        remote_output_dir=remote_output_dir,
        remote_trajectory_bank=remote_trajectory_bank,
        wandb_project=wandb_project,
    )


def _run_episode_impl(
    seed_prompt: str | None,
    seed_prompt_file: str,
    seed: int | None,
    max_turns: int,
    victim_history_turns: int | None,
    config_path: str,
    use_template_mutator: bool,
    reward_backend: str | None,
    aux_reward_backends: list[str] | None,
    show_mutator_input: bool,
    show_victim_input: bool,
    attacker_lora_adapter: str | None,
    policy_checkpoint: str | None,
    victim_lora_adapter: str | None,
    remote_output_dir: str,
    remote_trajectory_bank: str,
    wandb_project: str | None,
) -> dict:
    import sys
    from datetime import datetime
    from pathlib import Path

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _ensure_hf_token_env()
    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import RandomPolicy, TrailBlazerPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.versioning import annotate_turn_metadata, build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    run_id = new_run_id("modal_episode")
    # Optional Weights & Biases integration (best-effort: may not be installed in image)
    wandb_run = None
    if wandb_project or os.environ.get("WANDB_PROJECT"):
        try:
            import wandb

            _proj = wandb_project or os.environ.get("WANDB_PROJECT")
            wandb_run = wandb.init(project=_proj, name=run_id)
            wandb.define_metric("turn")
            wandb.define_metric("reward", step_metric="turn")
            wandb.define_metric("judge_label", step_metric="turn")
        except Exception as e:
            print(f"wandb init failed: {e}")
    if attacker_lora_adapter and use_template_mutator:
        raise ValueError("attacker_lora_adapter requires the default LLM mutator; remove --use-template-mutator.")
    selected_seed_prompt = seed_prompt or sample_seed_prompt(seed_prompt_file, seed=seed)
    policy = TrailBlazerPolicy.from_checkpoint(policy_checkpoint) if policy_checkpoint else RandomPolicy()
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
    selected_reward_backend = reward_backend or cfg.reward_backend()
    reward_model = build_reward_model(
        selected_reward_backend,
        prompt_guard_config=cfg.prompt_guard_config(),
        qwen_judge_config=cfg.qwen_judge_config(),
        llama_guard_config=cfg.llama_guard_config(),
        wildguard_config=cfg.wildguard_config(),
        reference_similarity_config=cfg.reference_similarity_config(),
    )
    auxiliary_reward_models = {
        backend: build_reward_model(
            backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
            wildguard_config=cfg.wildguard_config(),
            reference_similarity_config=cfg.reference_similarity_config(),
        )
        for backend in (aux_reward_backends or [])
    }

    env = RedTeamEnv(
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
        auxiliary_reward_models=auxiliary_reward_models,
    )

    state = env.reset(selected_seed_prompt)
    done = False
    while not done:
        state, _, done, _ = env.step()
        # Log per-turn metrics to W&B if enabled
        try:
            if wandb_run:
                turn = state.turns[-1]
                metrics = {"turn": len(state.turns), "reward": float(turn.reward) if turn.reward is not None else None}
                jl = None
                if isinstance(turn.metadata, dict):
                    jl = turn.metadata.get("judge_label")
                if jl is not None:
                    metrics["judge_label"] = jl
                wandb_run.log(metrics, step=len(state.turns))
        except Exception:
            pass

    result = {
        "seed_prompt": state.seed_prompt,
        "initial_template": state.initial_template,
        "turns": [
            {
                "action": turn.action.value if turn.action else None,
                "attack_template": turn.attack_template,
                "user_message": turn.user_message,
                "victim_response": turn.victim_response,
                "reward": turn.reward,
                "metadata": turn.metadata,
            }
            for turn in state.turns
        ],
    }
    run_metadata = build_model_metadata(
        run_id=run_id,
        victim_model=cfg.models.victim,
        victim_adapter_path=victim_lora_adapter,
        attacker_model="template" if use_template_mutator else cfg.models.mutator,
        attacker_adapter_path=attacker_lora_adapter or config_initial_attacker_adapter,
        policy_type="trailblazer" if policy_checkpoint else "random",
        policy_checkpoint=policy_checkpoint,
        judge_model=_judge_model_name(cfg, selected_reward_backend),
        extra={
            "config": config_path,
            "max_turns": max_turns,
            "victim_history_turns": env.config.victim_history_turns,
            "seed_prompt_file": seed_prompt_file,
            "reward_backend": selected_reward_backend,
        },
    )
    annotate_turn_metadata(result["turns"], run_metadata)
    result["metadata"] = run_metadata
    remote_episode_path, remote_full_inputs_path = _write_remote_episode_results(
        result,
        remote_output_dir,
    )
    from redteam_rl.trajectory_bank import append_episode_result

    clean_result = _strip_full_inputs(result)
    append_episode_result(
        clean_result,
        path=remote_trajectory_bank,
        metadata={
            "source": "modal_run_episode.py",
            "episode_path": remote_episode_path,
            "full_inputs_path": remote_full_inputs_path,
        },
    )
    volume.commit()
    if wandb_run:
        try:
            wandb_run.finish()
        except Exception:
            pass
    result["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "episode_path": remote_episode_path,
        "full_inputs_path": remote_full_inputs_path,
        "trajectory_bank": remote_trajectory_bank,
        "victim_lora_adapter": victim_lora_adapter,
    }
    return result


def _write_remote_episode_results(result: dict, remote_output_dir: str) -> tuple[str, str]:
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(remote_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"episode_{timestamp}.json"
    full_inputs_path = output_dir / f"episode_{timestamp}_full_inputs.json"
    stripped_result = _strip_full_inputs(result)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stripped_result, f, indent=2)
        f.write("\n")
    with full_inputs_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return str(path), str(full_inputs_path)


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


def _strip_full_inputs(result: dict) -> dict:
    stripped = {
        "seed_prompt": result["seed_prompt"],
        "initial_template": result.get("initial_template"),
        "turns": [],
    }
    if "metadata" in result:
        stripped["metadata"] = dict(result["metadata"])
    for turn in result.get("turns", []):
        metadata = dict(turn.get("metadata", {}))
        metadata.pop("mutator_input", None)
        metadata.pop("victim_input", None)
        stripped["turns"].append(
            {
                "action": turn.get("action"),
                "attack_template": turn.get("attack_template"),
                "user_message": turn.get("user_message"),
                "victim_response": turn.get("victim_response"),
                "reward": turn.get("reward"),
                "metadata": metadata,
            }
        )
    return stripped


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


def _print_episode_trace(
    result: dict,
    *,
    show_mutator_input: bool = False,
    show_victim_input: bool = False,
) -> None:
    """Print the episode in generation order instead of dumping nested JSON."""
    print("\nEpisode trace")
    print("=" * 80)
    metadata = result.get("metadata", {}) or {}
    artifacts = result.get("remote_artifacts", {}) or {}
    print(_kv_section(
        "Run metadata",
        {
            "run_id": metadata.get("run_id"),
            "policy_type": metadata.get("policy_type"),
            "policy_checkpoint": metadata.get("policy_checkpoint"),
            "victim_model": metadata.get("victim_model"),
            "victim_version": metadata.get("victim_version"),
            "victim_adapter_path": metadata.get("victim_adapter_path"),
            "attacker_model": metadata.get("attacker_model"),
            "attacker_version": metadata.get("attacker_version"),
            "attacker_adapter_path": metadata.get("attacker_adapter_path"),
            "judge_model": metadata.get("judge_model"),
            "reward_backend": metadata.get("reward_backend"),
            "max_turns": metadata.get("max_turns"),
            "victim_history_turns": metadata.get("victim_history_turns"),
            "seed_prompt_file": metadata.get("seed_prompt_file"),
        },
    ))
    print(_kv_section(
        "Remote artifacts",
        {
            "volume": artifacts.get("volume"),
            "episode_path": artifacts.get("episode_path"),
            "full_inputs_path": artifacts.get("full_inputs_path"),
            "trajectory_bank": artifacts.get("trajectory_bank"),
        },
    ))
    print(_section("Initial template", result.get("initial_template")))

    for index, turn in enumerate(result.get("turns", []), start=1):
        turn_metadata = turn.get("metadata", {}) or {}
        print(f"\nTurn {index}")
        print("-" * 80)
        print(_section("Action", turn.get("action")))
        if show_mutator_input:
            print(_section("Mutator input", turn_metadata.get("mutator_input")))
        print(_section("Mutator response", turn_metadata.get("raw_attack_template")))
        print(_section("Final template used", turn.get("attack_template")))
        print(_section("Bad query", result.get("seed_prompt")))
        print(_section("Attack message sent to victim", turn.get("user_message")))
        if show_victim_input:
            print(_section("Victim input", turn_metadata.get("victim_input")))
        print(_section("Victim response", turn.get("victim_response")))
        print(_kv_section(
            "Judge and reward",
            {
                "judge_label": turn_metadata.get("judge_label"),
                "reward": turn.get("reward"),
                "auxiliary_scores": turn_metadata.get("auxiliary_scores"),
            },
        ))
        print(_kv_section(
            "Mutator diagnostics",
            {
                "mutator_refused": turn_metadata.get("mutator_refused"),
                "mutator_invalid_template": turn_metadata.get("mutator_invalid_template"),
                "mutator_fallback_used": turn_metadata.get("mutator_fallback_used"),
                "mutator_fallback_template": turn_metadata.get("mutator_fallback_template"),
            },
        ))
        print(_kv_section(
            "Policy diagnostics",
            {
                "policy_log_prob": turn_metadata.get("policy_log_prob"),
                "policy_value": turn_metadata.get("policy_value"),
                "policy_action_probs": turn_metadata.get("policy_action_probs"),
                "policy_attention_weights": turn_metadata.get("policy_attention_weights"),
            },
        ))
        print(_kv_section(
            "Model versions",
            {
                "victim_model": turn_metadata.get("victim_model"),
                "victim_version": turn_metadata.get("victim_version"),
                "victim_adapter_path": turn_metadata.get("victim_adapter_path"),
                "attacker_model": turn_metadata.get("attacker_model"),
                "attacker_version": turn_metadata.get("attacker_version"),
                "attacker_adapter_path": turn_metadata.get("attacker_adapter_path"),
                "judge_model": turn_metadata.get("judge_model"),
            },
        ))

    print("\nEpisode summary")
    print("=" * 80)
    print(_kv_section("Summary metrics", _episode_summary_metrics(result)))


def _section(label: str, value: object) -> str:
    text = "" if value is None else str(value)
    return f"{label}:\n{text}\n"


def _kv_section(label: str, values: dict[str, object]) -> str:
    lines = [f"{label}:"]
    for key, value in values.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    if len(lines) == 1:
        lines.append("  n/a")
    return "\n".join(lines) + "\n"


def _episode_summary_metrics(result: dict) -> dict[str, object]:
    turns = result.get("turns", [])
    rewards = [
        float(turn.get("reward"))
        for turn in turns
        if turn.get("reward") is not None
    ]
    unsafe_turns = 0
    safe_turns = 0
    unknown_judge_turns = 0
    mutator_refusals = 0
    invalid_templates = 0
    fallback_uses = 0
    successful_attacks = 0

    for turn in turns:
        metadata = turn.get("metadata", {}) or {}
        label = str(metadata.get("judge_label", "")).upper()
        reward = turn.get("reward")
        if label == "UNSAFE":
            unsafe_turns += 1
        elif label == "SAFE":
            safe_turns += 1
        else:
            unknown_judge_turns += 1
        if metadata.get("mutator_refused"):
            mutator_refusals += 1
        if metadata.get("mutator_invalid_template"):
            invalid_templates += 1
        if metadata.get("mutator_fallback_used"):
            fallback_uses += 1
        if reward is not None and float(reward) >= 1.0:
            successful_attacks += 1

    return {
        "turns": len(turns),
        "successful_attacks_reward_ge_1": successful_attacks,
        "judge_unsafe": unsafe_turns,
        "judge_safe": safe_turns,
        "judge_unknown": unknown_judge_turns,
        "mutator_declines": mutator_refusals,
        "invalid_templates": invalid_templates,
        "fallbacks_used": fallback_uses,
        "average_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def run_episode_llama_guard_remote(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    victim_history_turns: int | None = None,
    config_path: str = "/root/configs/default.json",
    use_template_mutator: bool = False,
    aux_reward_backends: list[str] | None = None,
    show_mutator_input: bool = False,
    show_victim_input: bool = False,
    attacker_lora_adapter: str | None = None,
    policy_checkpoint: str | None = None,
    victim_lora_adapter: str | None = None,
    remote_output_dir: str = "/root/outputs/modal_episodes",
    remote_trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
    wandb_project: str | None = None,
) -> dict:
    return _run_episode_impl(
        seed_prompt=seed_prompt,
        seed_prompt_file=seed_prompt_file,
        seed=seed,
        max_turns=max_turns,
        victim_history_turns=victim_history_turns,
        config_path=config_path,
        use_template_mutator=use_template_mutator,
        reward_backend="llama_guard",
        aux_reward_backends=aux_reward_backends,
        show_mutator_input=show_mutator_input,
        show_victim_input=show_victim_input,
        attacker_lora_adapter=attacker_lora_adapter,
        policy_checkpoint=policy_checkpoint,
        victim_lora_adapter=victim_lora_adapter,
        remote_output_dir=remote_output_dir,
        remote_trajectory_bank=remote_trajectory_bank,
        wandb_project=wandb_project,
    )


@app.local_entrypoint()
def main(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    seeds: str | None = None,
    max_turns: int = 2,
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    reward_backend: str | None = None,
    aux_reward_backend: str | None = None,
    big_gpu: bool = False,
    show_mutator_input: bool = False,
    show_victim_input: bool = False,
    attacker_lora_adapter: str | None = None,
    policy_checkpoint: str | None = None,
    victim_lora_adapter: str | None = None,
    remote_output_dir: str = "/root/outputs/modal_episodes",
    remote_trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
    wandb_project: str | None = None,
    output_dir: str = "outputs/modal_episodes",
    output_file: str | None = None,
    save_local: bool = False,
    trajectory_bank: str = "outputs/trajectory_bank/episodes.jsonl",
) -> None:
    aux_reward_backends = [aux_reward_backend] if aux_reward_backend else None
    selected_use_template_mutator = use_template_mutator
    selected_seeds = _parse_seeds(seeds)
    if selected_seeds and seed_prompt:
        raise ValueError("--seeds cannot be combined with --seed-prompt because every seed would use the same prompt.")

    results = []
    seeds_to_run = selected_seeds or [seed]
    for seed_index, selected_seed in enumerate(seeds_to_run, start=1):
        result = _run_remote_entrypoint_episode(
            big_gpu=big_gpu,
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=selected_seed,
            max_turns=max_turns,
            victim_history_turns=victim_history_turns,
            use_template_mutator=selected_use_template_mutator,
            reward_backend=reward_backend,
            aux_reward_backends=aux_reward_backends,
            show_mutator_input=show_mutator_input,
            show_victim_input=show_victim_input,
            attacker_lora_adapter=attacker_lora_adapter,
            policy_checkpoint=policy_checkpoint,
            victim_lora_adapter=victim_lora_adapter,
            remote_output_dir=remote_output_dir,
            remote_trajectory_bank=remote_trajectory_bank,
            wandb_project=wandb_project,
        )
        results.append(result)
        if save_local:
            output_path = _write_episode_result(result, output_dir=output_dir, output_file=output_file)
            _append_episode_result_to_bank(result, trajectory_bank)
        if selected_seeds:
            print(f"\n\nSeed {selected_seed} ({seed_index}/{len(seeds_to_run)})")
        _print_episode_trace(
            result,
            show_mutator_input=show_mutator_input,
            show_victim_input=show_victim_input,
        )
        print("\nsaved remote episode JSON to Modal volume cs224r-redteam-rl-data")
        print("remote episode path:", result["remote_artifacts"]["episode_path"])
        print("remote full-inputs path:", result["remote_artifacts"]["full_inputs_path"])
        print("remote trajectory bank:", result["remote_artifacts"]["trajectory_bank"])
        if save_local:
            print(f"saved local episode JSON to {output_path}")
            print(f"appended local episode to {trajectory_bank}")

    if selected_seeds:
        print("\nMulti-seed episode summary")
        print("=" * 80)
        print(_kv_section("Summary", _multi_episode_summary_metrics(results)))


def _run_remote_entrypoint_episode(
    *,
    big_gpu: bool,
    seed_prompt: str | None,
    seed_prompt_file: str,
    seed: int | None,
    max_turns: int,
    victim_history_turns: int | None,
    use_template_mutator: bool,
    reward_backend: str | None,
    aux_reward_backends: list[str] | None,
    show_mutator_input: bool,
    show_victim_input: bool,
    attacker_lora_adapter: str | None,
    policy_checkpoint: str | None,
    victim_lora_adapter: str | None,
    remote_output_dir: str,
    remote_trajectory_bank: str,
    wandb_project: str | None,
) -> dict:
    if big_gpu:
        return run_episode_llama_guard_remote.remote(
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=seed,
            max_turns=max_turns,
            victim_history_turns=victim_history_turns,
            use_template_mutator=use_template_mutator,
            aux_reward_backends=aux_reward_backends,
            show_mutator_input=show_mutator_input,
            show_victim_input=show_victim_input,
            attacker_lora_adapter=attacker_lora_adapter,
            policy_checkpoint=policy_checkpoint,
            victim_lora_adapter=victim_lora_adapter,
            remote_output_dir=remote_output_dir,
            remote_trajectory_bank=remote_trajectory_bank,
            wandb_project=wandb_project,
        )
    return run_episode_remote.remote(
        seed_prompt=seed_prompt,
        seed_prompt_file=seed_prompt_file,
        seed=seed,
        max_turns=max_turns,
        victim_history_turns=victim_history_turns,
        use_template_mutator=use_template_mutator,
        reward_backend=reward_backend,
        aux_reward_backends=aux_reward_backends,
        show_mutator_input=show_mutator_input,
        show_victim_input=show_victim_input,
        attacker_lora_adapter=attacker_lora_adapter,
        policy_checkpoint=policy_checkpoint,
        victim_lora_adapter=victim_lora_adapter,
        remote_output_dir=remote_output_dir,
        remote_trajectory_bank=remote_trajectory_bank,
        wandb_project=wandb_project,
    )


def _parse_seeds(seeds: str | None) -> list[int]:
    if not seeds:
        return []
    parsed = []
    for raw_seed in seeds.split(","):
        stripped = raw_seed.strip()
        if stripped:
            parsed.append(int(stripped))
    return parsed


def _multi_episode_summary_metrics(results: list[dict]) -> dict[str, object]:
    total_turns = sum(len(result.get("turns", [])) for result in results)
    successful_episodes = 0
    successful_turns = 0
    for result in results:
        episode_success = False
        for turn in result.get("turns", []):
            if turn.get("reward") is not None and float(turn.get("reward")) >= 1.0:
                successful_turns += 1
                episode_success = True
        if episode_success:
            successful_episodes += 1
    return {
        "episodes": len(results),
        "turns": total_turns,
        "successful_episodes_reward_ge_1": successful_episodes,
        "successful_turns_reward_ge_1": successful_turns,
    }


def _write_episode_result(result: dict, output_dir: str, output_file: str | None = None) -> Path:
    path = Path(output_file) if output_file else _default_output_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return path


def _default_output_path(output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"episode_{timestamp}.json"


def _append_episode_result_to_bank(result: dict, trajectory_bank: str) -> None:
    from redteam_rl.trajectory_bank import append_episode_result

    append_episode_result(
        result,
        path=trajectory_bank,
        metadata={"source": "modal_run_episode.py"},
    )
