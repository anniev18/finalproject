"""Bounded parallel checkpoint sweep for TrailBlazer evaluation.

The existing ``modal_eval_trailblazer.py`` sweep remains unchanged. This
entrypoint partitions checkpoints across a limited number of A100 workers.
Each worker evaluates its assigned checkpoints sequentially, while workers run
in parallel. Individual checkpoint summaries and one combined ranking are
saved to the shared Modal volume.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cs224r-trailblazer-parallel-sweep")
volume = modal.Volume.from_name("cs224r-redteam-rl-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-gpu.txt")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("redteam_rl", remote_path="/root/redteam_rl")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("modal_eval_matrix.py", remote_path="/root/modal_eval_matrix.py")
    .add_local_file("modal_eval_trailblazer.py", remote_path="/root/modal_eval_trailblazer.py")
    .add_local_file("requirements-gpu.txt", remote_path="/root/requirements-gpu.txt")
)


def _evaluate_worker_chunk(payload: dict) -> dict:
    import modal_eval_matrix

    # The imported evaluator uses its module-level volume for periodic commits.
    # Point it at this app's hydrated shared volume.
    modal_eval_matrix.volume = volume
    evaluate_inline = modal_eval_matrix._evaluate_trailblazer_inline

    worker_index = int(payload["worker_index"])
    checkpoints = list(payload["checkpoints"])
    output_dir = Path(payload["remote_output_dir"]) / payload["eval_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    volume.commit()

    checkpoint_results = []
    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        label = _checkpoint_label(checkpoint)
        checkpoint_eval_name = f"{payload['eval_name']}_worker{worker_index:02d}_{label}"
        print(
            f"[worker {worker_index}] checkpoint {checkpoint_index}/{len(checkpoints)} "
            f"label={label} path={checkpoint}",
            flush=True,
        )
        result = evaluate_inline(
            trailblazer_checkpoint=checkpoint,
            trailblazer_checkpoints=None,
            num_episodes=payload["num_episodes"],
            max_turns=payload["max_turns"],
            seed=payload["seed"],
            seed_prompt_file=payload["seed_prompt_file"],
            config_path="/root/configs/default.json",
            victim_history_turns=payload.get("victim_history_turns"),
            reward_backend=payload.get("reward_backend"),
            use_template_mutator=payload.get("use_template_mutator", False),
            attacker_lora_adapter=payload.get("attacker_lora_adapter"),
            victim_lora_adapter=payload.get("victim_lora_adapter"),
            verbose=payload.get("verbose", True),
            show_text=False,
            remote_output_dir=str(output_dir / "checkpoints"),
            eval_name=checkpoint_eval_name,
            wandb_project=payload.get("wandb_project"),
        )
        metrics = dict(result.get("trailblazer", {}))
        checkpoint_result = {
            "checkpoint": checkpoint,
            "label": label,
            **metrics,
            "summary_path": result.get("remote_artifacts", {}).get("summary_path"),
            "episode_log_path": result.get("remote_artifacts", {}).get("episode_log_path"),
        }
        checkpoint_results.append(checkpoint_result)
        _write_json(
            output_dir / f"worker_{worker_index:02d}_partial.json",
            {
                "worker_index": worker_index,
                "assigned_checkpoints": checkpoints,
                "completed": checkpoint_results,
            },
        )
        volume.commit()

    worker_summary = {
        "worker_index": worker_index,
        "assigned_checkpoints": checkpoints,
        "completed": checkpoint_results,
    }
    _write_json(output_dir / f"worker_{worker_index:02d}.json", worker_summary)
    volume.commit()
    return worker_summary


@app.function(
    image=image,
    gpu="L40S",
    max_containers=4,
    timeout=60 * 60 * 12,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def sweep_worker_l40s(payload: dict) -> dict:
    return _evaluate_worker_chunk(payload)


@app.function(
    image=image,
    gpu="A100-80GB",
    max_containers=4,
    timeout=60 * 60 * 12,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def sweep_worker_a100_80gb(payload: dict) -> dict:
    return _evaluate_worker_chunk(payload)


@app.function(
    image=image,
    timeout=60 * 60 * 14,
    volumes={"/root/outputs": volume},
)
def coordinate_parallel_sweep(
    eval_name: str,
    checkpoints: list[str],
    num_workers: int = 4,
    gpu_type: str = "A100-80GB",
    num_episodes: int = 10,
    max_turns: int = 5,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts_val.json",
    reward_backend: str = "wildguard",
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    wandb_project: str | None = "trailblazer-ppo",
    verbose: bool = True,
    remote_output_dir: str = "/root/outputs/eval_parallel_sweeps",
) -> dict:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    if num_workers < 1 or num_workers > 4:
        raise ValueError("num_workers must be between 1 and 4.")

    selected_gpu = _normalize_gpu_type(gpu_type)
    chunks = _partition(checkpoints, min(num_workers, len(checkpoints)))
    output_dir = Path(remote_output_dir) / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    volume.commit()

    payloads = [
        {
            "worker_index": worker_index,
            "checkpoints": chunk,
            "eval_name": eval_name,
            "remote_output_dir": remote_output_dir,
            "num_episodes": num_episodes,
            "max_turns": max_turns,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "reward_backend": reward_backend,
            "victim_history_turns": victim_history_turns,
            "use_template_mutator": use_template_mutator,
            "attacker_lora_adapter": attacker_lora_adapter,
            "victim_lora_adapter": victim_lora_adapter,
            "wandb_project": wandb_project,
            "verbose": verbose,
        }
        for worker_index, chunk in enumerate(chunks, start=1)
    ]

    worker_function = {
        "L40S": sweep_worker_l40s,
        "A100-80GB": sweep_worker_a100_80gb,
    }[selected_gpu]
    worker_results = list(
        worker_function.map(
            payloads,
            order_outputs=True,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )

    initial_worker_failures = []
    completed = []
    for payload, result in zip(payloads, worker_results):
        if isinstance(result, BaseException):
            initial_worker_failures.append(_worker_failure(payload, result, selected_gpu))
            continue
        completed.extend(result.get("completed", []))

    unresolved_failures = list(initial_worker_failures)
    ranking = sorted(
        completed,
        key=lambda row: (
            float(row.get("success_rate", 0.0)),
            float(row.get("mean_return", 0.0)),
        ),
        reverse=True,
    )
    summary = {
        "eval_name": eval_name,
        "gpu_type": selected_gpu,
        "requested_workers": num_workers,
        "active_workers": len(chunks),
        "checkpoints_requested": checkpoints,
        "checkpoints_completed": len(completed),
        "initial_worker_failures": initial_worker_failures,
        "failures": unresolved_failures,
        "config": {
            "num_episodes": num_episodes,
            "max_turns": max_turns,
            "seed": seed,
            "seed_prompt_file": seed_prompt_file,
            "reward_backend": reward_backend,
            "victim_lora_adapter": victim_lora_adapter,
        },
        "checkpoint_results": completed,
        "checkpoint_ranking": ranking,
    }
    summary_path = output_dir / "sweep_summary.json"
    _write_json(summary_path, summary)
    volume.commit()
    summary["remote_artifacts"] = {
        "volume": "cs224r-redteam-rl-data",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
    }
    return summary


def _partition(items: list[str], num_chunks: int) -> list[list[str]]:
    chunks = [[] for _ in range(num_chunks)]
    for index, item in enumerate(items):
        chunks[index % num_chunks].append(item)
    return [chunk for chunk in chunks if chunk]


def _worker_failure(payload: dict, error: BaseException, gpu_type: str) -> dict:
    return {
        "worker_index": payload["worker_index"],
        "assigned_checkpoints": payload["checkpoints"],
        "gpu_type": gpu_type,
        "status": "failed",
        "error": repr(error),
    }


def _checkpoint_label(checkpoint: str) -> str:
    stem = Path(checkpoint).stem
    return stem.removeprefix("checkpoint_") if stem.startswith("checkpoint_") else stem


def _parse_checkpoints(checkpoint_dir: str, checkpoint_epochs: str) -> list[str]:
    checkpoints = []
    for raw_epoch in checkpoint_epochs.split(","):
        epoch = raw_epoch.strip()
        if epoch:
            checkpoints.append(str(Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"))
    return checkpoints


def _normalize_gpu_type(gpu_type: str) -> str:
    normalized = gpu_type.strip().upper().replace("_", "-")
    if normalized in {"L40S", "L40"}:
        return "L40S"
    if normalized in {"A100", "A100-80GB", "A100-80G"}:
        return "A100-80GB"
    raise ValueError("gpu_type must be L40S or A100-80GB")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


@app.local_entrypoint()
def main(
    checkpoint_dir: str,
    checkpoint_epochs: str,
    eval_name: str | None = None,
    num_workers: int = 4,
    gpu_type: str = "A100-80GB",
    num_episodes: int = 10,
    max_turns: int = 5,
    seed: int = 0,
    seed_prompt_file: str = "/root/data/seed_prompts_val.json",
    reward_backend: str = "wildguard",
    victim_history_turns: int | None = None,
    use_template_mutator: bool = False,
    attacker_lora_adapter: str | None = None,
    victim_lora_adapter: str | None = None,
    wandb_project: str | None = "trailblazer-ppo",
    verbose: bool = True,
    remote_output_dir: str = "/root/outputs/eval_parallel_sweeps",
    wait_for_result: bool = False,
) -> None:
    resolved_eval_name = eval_name or f"parallel_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    checkpoints = _parse_checkpoints(checkpoint_dir, checkpoint_epochs)
    call_kwargs = {
        "eval_name": resolved_eval_name,
        "checkpoints": checkpoints,
        "num_workers": num_workers,
        "gpu_type": gpu_type,
        "num_episodes": num_episodes,
        "max_turns": max_turns,
        "seed": seed,
        "seed_prompt_file": seed_prompt_file,
        "reward_backend": reward_backend,
        "victim_history_turns": victim_history_turns,
        "use_template_mutator": use_template_mutator,
        "attacker_lora_adapter": attacker_lora_adapter,
        "victim_lora_adapter": victim_lora_adapter,
        "wandb_project": wandb_project,
        "verbose": verbose,
        "remote_output_dir": remote_output_dir,
    }
    if not wait_for_result:
        call = coordinate_parallel_sweep.spawn(**call_kwargs)
        call_id = getattr(call, "object_id", None) or getattr(call, "call_id", None)
        print("submitted bounded parallel checkpoint sweep")
        if call_id:
            print("function call id:", call_id)
        print("eval name:", resolved_eval_name)
        print("gpu type:", _normalize_gpu_type(gpu_type))
        print("maximum parallel workers:", num_workers)
        print("checkpoints:", checkpoints)
        print("remote summary path:", f"{remote_output_dir}/{resolved_eval_name}/sweep_summary.json")
        print("Modal volume: cs224r-redteam-rl-data")
        print("Use `modal app logs cs224r-trailblazer-parallel-sweep` to follow progress.")
        return

    result = coordinate_parallel_sweep.remote(**call_kwargs)
    print(json.dumps(result, indent=2))
