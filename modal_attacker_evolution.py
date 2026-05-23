"""Modal entrypoint for attacker mutator LoRA evolution from trajectory bank."""

import json
import os

import modal


app = modal.App("cs224r-attacker-evolution")
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
    volumes={"/root/outputs": volume},
)
def run_attacker_evolution_remote(
    config_path: str = "/root/configs/default.json",
    trajectory_bank: str = "/root/outputs/trajectory_bank/episodes.jsonl",
    output_dir: str = "/root/outputs/attacker_rounds",
    round_index: int = 1,
    min_reward: float = 1.0,
    victim_version: str | None = None,
    attacker_version: str | None = None,
    max_bank_episodes: int = 200,
    verify_adapter_update: bool = False,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from dataclasses import asdict
    from pathlib import Path
    import sys

    sys.path.insert(0, "/root")

    from redteam_rl.attacker_training import (
        attacker_examples_from_trajectory_bank,
        fine_tune_attacker_lora,
    )
    from redteam_rl.config import load_config
    from redteam_rl.orchestration import verify_adapter_update as verify_adapter
    from redteam_rl.versioning import new_run_id

    cfg = load_config(config_path)
    run_id = new_run_id("modal_attacker_evolution")
    examples = attacker_examples_from_trajectory_bank(
        trajectory_bank,
        min_reward=min_reward,
        victim_version=victim_version,
        attacker_version=attacker_version,
        limit=max_bank_episodes,
    )

    adapter_path = None
    if examples:
        adapter_path = fine_tune_attacker_lora(
            cfg.attacker_finetune_config(
                output_dir=Path(output_dir) / f"round_{round_index:03d}"
            ),
            examples,
        )

    verification = (
        verify_adapter(adapter_path)
        if verify_adapter_update and adapter_path is not None
        else None
    )
    summary = {
        "run_id": run_id,
        "round_index": round_index,
        "trajectory_bank": trajectory_bank,
        "training_examples": len(examples),
        "min_reward": min_reward,
        "victim_version_filter": victim_version,
        "attacker_version_filter": attacker_version,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_verification": verification,
        "examples": [asdict(example) for example in examples[:10]],
    }

    output_path = Path(output_dir) / f"attacker_round_{round_index:03d}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    round_index: int = 1,
    min_reward: float = 1.0,
    victim_version: str | None = None,
    attacker_version: str | None = None,
    max_bank_episodes: int = 200,
    verify_adapter_update: bool = False,
    save_local: bool = False,
    output_dir: str = "outputs/attacker_rounds",
) -> None:
    result = run_attacker_evolution_remote.remote(
        round_index=round_index,
        min_reward=min_reward,
        victim_version=victim_version,
        attacker_version=attacker_version,
        max_bank_episodes=max_bank_episodes,
        verify_adapter_update=verify_adapter_update,
    )
    print(json.dumps(result, indent=2))
    print("\nsaved remote attacker evolution artifacts to Modal volume cs224r-redteam-rl-data")
    print("remote attacker rounds path: /root/outputs/attacker_rounds")
    print("remote trajectory bank: /root/outputs/trajectory_bank/episodes.jsonl")
    if save_local:
        os.makedirs(output_dir, exist_ok=True)
        local_summary_path = os.path.join(output_dir, "modal_attacker_evolution_summary.json")
        with open(local_summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")

