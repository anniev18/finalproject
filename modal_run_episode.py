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
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/outputs": volume},
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
    victim_lora_adapter: str | None = None,
    remote_output_dir: str = "/root/outputs/modal_episodes",
    remote_trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
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
        victim_lora_adapter=victim_lora_adapter,
        remote_output_dir=remote_output_dir,
        remote_trajectory_bank=remote_trajectory_bank,
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
    victim_lora_adapter: str | None,
    remote_output_dir: str,
    remote_trajectory_bank: str,
) -> dict:
    import sys
    from datetime import datetime
    from pathlib import Path

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker, EvolvingAttacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import RandomPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.versioning import annotate_turn_metadata, build_model_metadata, new_run_id
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    run_id = new_run_id("modal_episode")
    if attacker_lora_adapter and use_template_mutator:
        raise ValueError("attacker_lora_adapter requires the default LLM mutator; remove --use-template-mutator.")
    selected_seed_prompt = seed_prompt or sample_seed_prompt(seed_prompt_file, seed=seed)
    policy = RandomPolicy()
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
    )
    auxiliary_reward_models = {
        backend: build_reward_model(
            backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
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

    result = {
        "seed_prompt": state.seed_prompt,
        "turns": [
            {
                "action": turn.action.value if turn.action else None,
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


def _strip_full_inputs(result: dict) -> dict:
    stripped = {
        "seed_prompt": result["seed_prompt"],
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
    return "fake"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/outputs": volume},
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
    victim_lora_adapter: str | None = None,
    remote_output_dir: str = "/root/outputs/modal_episodes",
    remote_trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
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
        victim_lora_adapter=victim_lora_adapter,
        remote_output_dir=remote_output_dir,
        remote_trajectory_bank=remote_trajectory_bank,
    )


@app.local_entrypoint()
def main(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    reward_backend: str | None = None,
    aux_reward_backend: str | None = None,
    big_gpu: bool = False,
    show_mutator_input: bool = False,
    show_victim_input: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    output_dir: str = "outputs/modal_episodes",
    output_file: str | None = None,
    save_local: bool = False,
    trajectory_bank: str = "outputs/trajectory_bank/episodes.jsonl",
) -> None:
    aux_reward_backends = [aux_reward_backend] if aux_reward_backend else None
    selected_use_template_mutator = use_template_mutator
    if big_gpu:
        result = run_episode_llama_guard_remote.remote(
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=seed,
            max_turns=max_turns,
            victim_history_turns=victim_history_turns,
            use_template_mutator=selected_use_template_mutator,
            aux_reward_backends=aux_reward_backends,
            show_mutator_input=show_mutator_input,
            show_victim_input=show_victim_input,
            attacker_lora_adapter=attacker_lora_adapter,
            victim_lora_adapter=victim_lora_adapter,
        )
    else:
        result = run_episode_remote.remote(
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=seed,
            max_turns=max_turns,
            victim_history_turns=victim_history_turns,
            use_template_mutator=selected_use_template_mutator,
            reward_backend=reward_backend,
            aux_reward_backends=aux_reward_backends,
            show_mutator_input=show_mutator_input,
            show_victim_input=show_victim_input,
            attacker_lora_adapter=attacker_lora_adapter,
            victim_lora_adapter=victim_lora_adapter,
        )
    if save_local:
        output_path = _write_episode_result(result, output_dir=output_dir, output_file=output_file)
        _append_episode_result_to_bank(result, trajectory_bank)
    print(json.dumps(result, indent=2))
    print("\nsaved remote episode JSON to Modal volume cs224r-redteam-rl-data")
    print("remote episode path:", result["remote_artifacts"]["episode_path"])
    print("remote full-inputs path:", result["remote_artifacts"]["full_inputs_path"])
    print("remote trajectory bank:", result["remote_artifacts"]["trajectory_bank"])
    if save_local:
        print(f"saved local episode JSON to {output_path}")
        print(f"appended local episode to {trajectory_bank}")


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
