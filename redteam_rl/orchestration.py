"""Round-based orchestration helpers for victim evolution."""

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from redteam_rl.env import RedTeamEnv
from redteam_rl.safety_targets import load_safety_target_templates
from redteam_rl.seed_prompts import sample_seed_prompt
from redteam_rl.trajectory_bank import append_episodes, load_episodes
from redteam_rl.types import EpisodeState
from redteam_rl.versioning import adapter_version
from redteam_rl.victim_training import SafetyExample, safety_examples_from_episodes

TrajectorySource = Literal["fresh", "bank", "fresh_plus_bank"]


class EvolvableVictim(Protocol):
    def evolve_to_adapter(self, adapter_path: str | Path) -> None:
        """Switch the active victim adapter."""


@dataclass(frozen=True)
class VictimEvolutionConfig:
    enabled: bool = False
    num_rounds: int = 2
    episodes_per_round: int = 4
    min_reward_for_training: float = 1.0
    output_dir: str = "outputs/victim_rounds"
    safety_target_file: str = "data/safety_target_templates.json"
    skip_empty_updates: bool = True
    verify_adapter_update: bool = False
    trajectory_bank_path: str = "outputs/trajectory_bank/episodes.jsonl"
    trajectory_source: TrajectorySource = "fresh_plus_bank"
    max_bank_episodes: int = 200


@dataclass(frozen=True)
class RoundSummary:
    round_index: int
    run_id: str | None
    victim_version: str | None
    victim_adapter_path: str | None
    attacker_version: str | None
    attacker_adapter_path: str | None
    judge_model: str | None
    episodes_collected: int
    bank_episodes_used: int
    turns_collected: int
    training_examples: int
    adapter_path: str | None
    adapter_verification: dict[str, object] | None
    average_reward: float
    max_reward: float


def collect_episodes(
    env: RedTeamEnv,
    num_episodes: int,
    seed_prompt_file: str | Path,
    seed: int | None = None,
    seed_prompt: str | None = None,
) -> list[EpisodeState]:
    episodes = []
    for episode_index in range(num_episodes):
        episode_seed = None if seed is None else seed + episode_index
        selected_seed_prompt = seed_prompt or sample_seed_prompt(seed_prompt_file, seed=episode_seed)
        state = env.reset(selected_seed_prompt)
        done = False
        while not done:
            state, _, done, _ = env.step()
        episodes.append(state)
    return episodes


def attach_safety_targets(
    episodes: list[EpisodeState],
    safety_target_file: str | Path,
    min_reward: float,
) -> None:
    """Attach safe target responses to successful attack turns in-place."""

    templates = load_safety_target_templates(safety_target_file)
    if not templates:
        raise ValueError(f"No safety target templates found in {safety_target_file}.")

    for episode in episodes:
        for turn_index, turn in enumerate(episode.turns):
            reward = turn.reward if turn.reward is not None else 0.0
            if reward < min_reward:
                continue
            template = templates[turn_index % len(templates)]
            turn.metadata["safe_response"] = template.text
            turn.metadata["safe_response_template_id"] = template.id


def build_safety_examples(
    episodes: list[EpisodeState],
    config: VictimEvolutionConfig,
) -> list[SafetyExample]:
    attach_safety_targets(
        episodes,
        safety_target_file=config.safety_target_file,
        min_reward=config.min_reward_for_training,
    )
    return safety_examples_from_episodes(
        episodes,
        min_reward=config.min_reward_for_training,
    )


def run_victim_evolution_rounds(
    env: RedTeamEnv,
    victim: EvolvableVictim,
    config: VictimEvolutionConfig,
    seed_prompt_file: str | Path,
    train_adapter_fn: Callable[[list[EpisodeState], int, list[SafetyExample]], Path],
    seed: int | None = None,
    seed_prompt: str | None = None,
    log_dir: str | Path | None = None,
    model_metadata: dict[str, object] | None = None,
) -> list[RoundSummary]:
    """Collect episodes, train/update victim adapter, and continue for rounds."""

    summaries = []
    round_log_dir = Path(log_dir or Path(config.output_dir) / "logs")
    round_log_dir.mkdir(parents=True, exist_ok=True)

    for round_index in range(1, config.num_rounds + 1):
        round_metadata = build_round_metadata(
            base_metadata=model_metadata,
            victim=victim,
            round_index=round_index,
            trajectory_source=config.trajectory_source,
        )
        episodes, training_episodes, bank_episodes_used = prepare_round_episodes(
            env=env,
            config=config,
            seed_prompt_file=seed_prompt_file,
            round_index=round_index,
            seed=seed,
            seed_prompt=seed_prompt,
            metadata=round_metadata,
        )

        examples = build_safety_examples(training_episodes, config)
        adapter_path: Path | None = None

        if examples or not config.skip_empty_updates:
            adapter_path = train_adapter_fn(training_episodes, round_index, examples)
            victim.evolve_to_adapter(adapter_path)

        adapter_verification = (
            verify_adapter_update(adapter_path) if config.verify_adapter_update and adapter_path else None
        )
        summary = summarize_round(
            round_index=round_index,
            episodes=episodes,
            bank_episodes_used=bank_episodes_used,
            training_examples=len(examples),
            adapter_path=adapter_path,
            adapter_verification=adapter_verification,
            metadata=round_metadata,
        )
        summaries.append(summary)
        write_round_artifacts(round_log_dir, summary, episodes, examples)

    return summaries


def prepare_round_episodes(
    env: RedTeamEnv,
    config: VictimEvolutionConfig,
    seed_prompt_file: str | Path,
    round_index: int,
    seed: int | None = None,
    seed_prompt: str | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[list[EpisodeState], list[EpisodeState], int]:
    """Return fresh episodes, training episodes, and number loaded from bank."""

    if config.trajectory_source not in {"fresh", "bank", "fresh_plus_bank"}:
        raise ValueError(f"Unsupported trajectory_source: {config.trajectory_source}")

    fresh_episodes: list[EpisodeState] = []
    if config.trajectory_source in {"fresh", "fresh_plus_bank"}:
        round_seed = None if seed is None else seed + (round_index - 1) * config.episodes_per_round
        fresh_episodes = collect_episodes(
            env,
            num_episodes=config.episodes_per_round,
            seed_prompt_file=seed_prompt_file,
            seed=round_seed,
            seed_prompt=seed_prompt,
        )
        for episode in fresh_episodes:
            annotate_episode_state(episode, metadata or {})

    bank_episodes: list[EpisodeState] = []
    if config.trajectory_source in {"bank", "fresh_plus_bank"}:
        if fresh_episodes:
            append_episodes(
                fresh_episodes,
                path=config.trajectory_bank_path,
                metadata={
                    **(metadata or {}),
                    "source": "victim_evolution",
                    "round_index": round_index,
                },
            )
        bank_episodes = load_episodes(
            config.trajectory_bank_path,
            limit=config.max_bank_episodes,
        )

    if config.trajectory_source == "fresh":
        return fresh_episodes, fresh_episodes, 0
    if config.trajectory_source == "bank":
        return fresh_episodes, bank_episodes, len(bank_episodes)
    return fresh_episodes, bank_episodes, max(len(bank_episodes) - len(fresh_episodes), 0)


def summarize_round(
    round_index: int,
    episodes: list[EpisodeState],
    bank_episodes_used: int,
    training_examples: int,
    adapter_path: Path | None,
    adapter_verification: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> RoundSummary:
    rewards = [
        float(turn.reward or 0.0)
        for episode in episodes
        for turn in episode.turns
    ]
    return RoundSummary(
        round_index=round_index,
        run_id=str(metadata.get("run_id")) if metadata and metadata.get("run_id") else None,
        victim_version=str(metadata.get("victim_version")) if metadata and metadata.get("victim_version") else None,
        victim_adapter_path=str(metadata.get("victim_adapter_path")) if metadata and metadata.get("victim_adapter_path") else None,
        attacker_version=str(metadata.get("attacker_version")) if metadata and metadata.get("attacker_version") else None,
        attacker_adapter_path=str(metadata.get("attacker_adapter_path")) if metadata and metadata.get("attacker_adapter_path") else None,
        judge_model=str(metadata.get("judge_model")) if metadata and metadata.get("judge_model") else None,
        episodes_collected=len(episodes),
        bank_episodes_used=max(bank_episodes_used, 0),
        turns_collected=sum(len(episode.turns) for episode in episodes),
        training_examples=training_examples,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        adapter_verification=adapter_verification,
        average_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        max_reward=max(rewards) if rewards else 0.0,
    )


def build_round_metadata(
    base_metadata: dict[str, object] | None,
    victim: EvolvableVictim,
    round_index: int,
    trajectory_source: str,
) -> dict[str, object]:
    metadata = dict(base_metadata or {})
    current_adapter = latest_victim_adapter_path(victim)
    metadata.update(
        {
            "round_index": round_index,
            "trajectory_source": trajectory_source,
            "victim_version": adapter_version("victim", current_adapter),
            "victim_adapter_path": str(current_adapter) if current_adapter else None,
        }
    )
    return metadata


def latest_victim_adapter_path(victim: EvolvableVictim) -> Path | None:
    adapter_history = getattr(victim, "adapter_history", None)
    if adapter_history:
        return Path(adapter_history[-1])
    adapter_path = getattr(victim, "adapter_path", None)
    return Path(adapter_path) if adapter_path else None


def annotate_episode_state(episode: EpisodeState, metadata: dict[str, object]) -> None:
    per_turn_keys = (
        "run_id",
        "round_index",
        "trajectory_source",
        "victim_model",
        "victim_version",
        "victim_adapter_path",
        "attacker_model",
        "attacker_version",
        "attacker_adapter_path",
        "judge_model",
    )
    for turn in episode.turns:
        for key in per_turn_keys:
            if key in metadata:
                turn.metadata.setdefault(key, metadata[key])


def write_round_artifacts(
    log_dir: Path,
    summary: RoundSummary,
    episodes: list[EpisodeState],
    examples: list[SafetyExample],
) -> None:
    round_dir = log_dir / f"round_{summary.round_index:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    with (round_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
        f.write("\n")

    with (round_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps({"episode": asdict(episode)}) + "\n")

    with (round_dir / "safety_examples.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(example) for example in examples], f, indent=2)
        f.write("\n")


def create_dry_run_adapter(output_dir: str | Path, round_index: int, examples: list[SafetyExample]) -> Path:
    """Create a marker directory that stands in for a trained adapter."""

    adapter_path = Path(output_dir) / f"dry_round_{round_index:03d}"
    adapter_path.mkdir(parents=True, exist_ok=True)
    with (adapter_path / "DRY_RUN_ADAPTER.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dry-run adapter placeholder for round {round_index}\n")
        f.write(f"training_examples={len(examples)}\n")
    return adapter_path


def verify_adapter_update(adapter_path: str | Path) -> dict[str, object]:
    """Return lightweight evidence that an adapter artifact was written."""

    path = Path(adapter_path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    verification: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "files": [
            {
                "name": str(file.relative_to(path)),
                "size_bytes": file.stat().st_size,
                "sha256": _sha256(file),
            }
            for file in files
        ],
    }

    adapter_weights = path / "adapter_model.safetensors"
    if adapter_weights.exists():
        verification["adapter_model_safetensors"] = _summarize_safetensors(adapter_weights)

    dry_marker = path / "DRY_RUN_ADAPTER.txt"
    if dry_marker.exists():
        verification["dry_run_marker"] = dry_marker.read_text(encoding="utf-8").strip()

    attacker_dry_marker = path / "DRY_RUN_ATTACKER_ADAPTER.txt"
    if attacker_dry_marker.exists():
        verification["dry_run_marker"] = attacker_dry_marker.read_text(encoding="utf-8").strip()

    return verification


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _summarize_safetensors(path: Path) -> dict[str, object]:
    try:
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
    except Exception as exc:
        return {"loaded": False, "error": str(exc)}

    tensor_summaries = {}
    total_params = 0
    for name, tensor in tensors.items():
        total_params += tensor.numel()
        tensor_summaries[name] = {
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "l2_norm": float(tensor.float().norm().item()),
        }

    return {
        "loaded": True,
        "tensor_count": len(tensors),
        "total_params": total_params,
        "tensors": tensor_summaries,
    }
