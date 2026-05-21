"""Modal entrypoint for real victim LoRA evolution.

Run from the repo root only when you want GPU fine-tuning:

    modal run modal_victim_evolution.py --num-rounds=1 --episodes-per-round=2
"""

import json
import os

import modal


app = modal.App("cs224r-victim-evolution")
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
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/outputs": volume},
)
def run_victim_evolution_remote(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = 0,
    config_path: str = "/root/configs/default.json",
    num_rounds: int | None = None,
    episodes_per_round: int | None = None,
    max_turns: int | None = None,
    output_dir: str = "/root/outputs/victim_rounds",
    use_template_mutator: bool = True,
    reward_backend: str | None = None,
    verify_adapter_update: bool = False,
    trajectory_source: str | None = None,
) -> list[dict]:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from dataclasses import asdict
    from pathlib import Path
    import sys

    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker
    from redteam_rl.config import load_config
    from redteam_rl.env import RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.orchestration import run_victim_evolution_rounds
    from redteam_rl.policy import RandomPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.versioning import build_model_metadata, new_run_id
    from redteam_rl.victim_training import train_victim_from_episodes
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    run_id = new_run_id("modal_victim_evolution")
    evolution_config = cfg.victim_evolution_config(
        output_dir=output_dir,
        safety_target_file="/root/data/safety_target_templates.json",
        trajectory_bank_path="/root/outputs/trajectory_bank/episodes.jsonl",
        verify_adapter_update=verify_adapter_update,
        **({"trajectory_source": trajectory_source} if trajectory_source else {}),
        **({"num_rounds": num_rounds} if num_rounds is not None else {}),
        **(
            {"episodes_per_round": episodes_per_round}
            if episodes_per_round is not None
            else {}
        ),
    )

    mutator = TemplateMutator() if use_template_mutator else LLMMutator(cfg.mutator_config())
    attacker = Attacker(policy=RandomPolicy(), mutator=mutator)
    victim = EvolvingVictim(VLLMVictim(cfg.victim_config()))
    selected_reward_backend = reward_backend or cfg.reward_backend()
    reward_model = build_reward_model(
        selected_reward_backend,
        prompt_guard_config=cfg.prompt_guard_config(),
        qwen_judge_config=cfg.qwen_judge_config(),
        llama_guard_config=cfg.llama_guard_config(),
    )
    env = RedTeamEnv(
        config=cfg.env_config(**({"max_turns": max_turns} if max_turns else {})),
        attacker=attacker,
        victim=victim,
        reward_model=reward_model,
    )

    def train_adapter_fn(episodes, round_index, _examples):
        round_output_dir = Path(evolution_config.output_dir) / f"round_{round_index:03d}"
        training_config = cfg.victim_finetune_config(
            output_dir=round_output_dir,
            base_adapter_path=victim.adapter_history[-1] if victim.adapter_history else None,
        )
        return train_victim_from_episodes(
            episodes,
            training_config,
            min_reward=evolution_config.min_reward_for_training,
        )

    summaries = run_victim_evolution_rounds(
        env=env,
        victim=victim,
        config=evolution_config,
        seed_prompt_file=seed_prompt_file,
        train_adapter_fn=train_adapter_fn,
        seed=seed,
        seed_prompt=seed_prompt,
        log_dir=Path(evolution_config.output_dir) / "logs",
        model_metadata=build_model_metadata(
            run_id=run_id,
            victim_model=cfg.models.victim,
            attacker_model="template" if use_template_mutator else cfg.models.mutator,
            judge_model=_judge_model_name(cfg, selected_reward_backend),
            trajectory_source=evolution_config.trajectory_source,
            extra={
                "config": config_path,
                "max_turns": max_turns,
                "seed_prompt_file": seed_prompt_file,
                "reward_backend": selected_reward_backend,
            },
        ),
    )
    volume.commit()
    return [asdict(summary) for summary in summaries]


@app.local_entrypoint()
def main(
    seed_prompt: str | None = None,
    seed: int | None = 0,
    num_rounds: int | None = None,
    episodes_per_round: int | None = None,
    max_turns: int | None = None,
    output_dir: str = "outputs/victim_rounds",
    save_local: bool = False,
    use_template_mutator: bool = True,
    use_llm_mutator: bool = False,
    reward_backend: str | None = None,
    verify_adapter_update: bool = False,
    trajectory_source: str | None = None,
) -> None:
    result = run_victim_evolution_remote.remote(
        seed=seed,
        seed_prompt=seed_prompt,
        num_rounds=num_rounds,
        episodes_per_round=episodes_per_round,
        max_turns=max_turns,
        output_dir="/root/outputs/victim_rounds",
        use_template_mutator=use_template_mutator and not use_llm_mutator,
        reward_backend=reward_backend,
        verify_adapter_update=verify_adapter_update,
        trajectory_source=trajectory_source,
    )
    print(json.dumps(result, indent=2))
    print("\nsaved remote victim evolution artifacts to Modal volume cs224r-redteam-rl-data")
    print("remote victim rounds path: /root/outputs/victim_rounds")
    print("remote trajectory bank: /root/outputs/trajectory_bank/episodes.jsonl")
    if save_local:
        os.makedirs(output_dir, exist_ok=True)
        local_summary_path = os.path.join(output_dir, "modal_victim_evolution_summary.json")
        with open(local_summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")


def _judge_model_name(cfg, reward_backend: str) -> str:
    if reward_backend in {"qwen_judge", "qwen_safety_judge"}:
        return cfg.models.qwen_safety_judge
    if reward_backend == "prompt_guard":
        return cfg.models.prompt_guard
    if reward_backend == "llama_guard":
        return cfg.models.llama_guard
    return "fake"
