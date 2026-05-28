"""Project configuration loading."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redteam_rl.attacker import AttackerEvolutionConfig
from redteam_rl.attacker_training import AttackerFineTuneConfig
from redteam_rl.mutators import MutatorConfig
from redteam_rl.env import EnvConfig
from redteam_rl.orchestration import VictimEvolutionConfig
from redteam_rl.rewards import LlamaGuardConfig, PromptGuardConfig, QwenJudgeConfig, RewardBackend, WildGuardConfig
from redteam_rl.state_features import StateFeatureConfig
from redteam_rl.victim_training import VictimFineTuneConfig
from redteam_rl.victims import VictimConfig


DEFAULT_CONFIG_PATH = Path("configs/default.json")


@dataclass(frozen=True)
class ModelNames:
    victim: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mutator: str = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen_safety_judge: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    prompt_guard: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    llama_guard: str = "meta-llama/Meta-Llama-Guard-2-8B"
    wildguard: str = "allenai/wildguard"


@dataclass(frozen=True)
class ProjectConfig:
    models: ModelNames = field(default_factory=ModelNames)
    environment: dict[str, Any] = field(default_factory=dict)
    attacker_evolution: dict[str, Any] = field(default_factory=dict)
    victim_evolution: dict[str, Any] = field(default_factory=dict)
    state_features: dict[str, Any] = field(default_factory=dict)
    victim: dict[str, Any] = field(default_factory=dict)
    mutator: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    victim_training: dict[str, Any] = field(default_factory=dict)
    attacker_training: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProjectConfig":
        model_names = ModelNames(**raw.get("models", {}))
        return cls(
            models=model_names,
            environment=dict(raw.get("environment", {})),
            attacker_evolution=dict(raw.get("attacker_evolution", {})),
            victim_evolution=dict(raw.get("victim_evolution", {})),
            state_features=dict(raw.get("state_features", {})),
            victim=dict(raw.get("victim", {})),
            mutator=dict(raw.get("mutator", {})),
            reward=dict(raw.get("reward", {})),
            victim_training=dict(raw.get("victim_training", {})),
            attacker_training=dict(raw.get("attacker_training", {})),
        )

    def env_config(self, **overrides: Any) -> EnvConfig:
        values = {
            **self.environment,
            **overrides,
        }
        return EnvConfig(**values)

    def attacker_evolution_config(self, **overrides: Any) -> AttackerEvolutionConfig:
        values = {
            **self.attacker_evolution,
            **overrides,
        }
        return AttackerEvolutionConfig(**values)

    def victim_evolution_config(self, **overrides: Any) -> VictimEvolutionConfig:
        values = {
            **self.victim_evolution,
            **overrides,
        }
        return VictimEvolutionConfig(**values)

    def state_feature_config(self, **overrides: Any) -> StateFeatureConfig:
        values = {
            "embedding_model_name": self.models.embedding,
            **self.state_features,
            **overrides,
        }
        return StateFeatureConfig(**values)

    def victim_config(self, **overrides: Any) -> VictimConfig:
        values = {
            "model_name": self.models.victim,
            **self.victim,
            **overrides,
        }
        return VictimConfig(**values)

    def mutator_config(self, **overrides: Any) -> MutatorConfig:
        values = {
            "model_name": self.models.mutator,
            **self.mutator,
            **overrides,
        }
        return MutatorConfig(**values)

    def reward_backend(self) -> RewardBackend:
        return self.reward.get("backend", "qwen_safety_judge")

    def prompt_guard_config(self, **overrides: Any) -> PromptGuardConfig:
        prompt_guard_values = dict(self.reward.get("prompt_guard", {}))
        values = {
            "model_name": self.models.prompt_guard,
            **prompt_guard_values,
            **overrides,
        }
        return PromptGuardConfig(**values)

    def qwen_judge_config(self, **overrides: Any) -> QwenJudgeConfig:
        qwen_judge_values = dict(
            self.reward.get("qwen_safety_judge", self.reward.get("qwen_judge", {}))
        )
        values = {
            "model_name": self.models.qwen_safety_judge,
            **qwen_judge_values,
            **overrides,
        }
        return QwenJudgeConfig(**values)

    def llama_guard_config(self, **overrides: Any) -> LlamaGuardConfig:
        llama_guard_values = dict(self.reward.get("llama_guard", {}))
        values = {
            "model_name": self.models.llama_guard,
            **llama_guard_values,
            **overrides,
        }
        return LlamaGuardConfig(**values)

    def wildguard_config(self, **overrides: Any) -> WildGuardConfig:
        wildguard_values = dict(self.reward.get("wildguard", {}))
        values = {
            "model_name": self.models.wildguard,
            **wildguard_values,
            **overrides,
        }
        return WildGuardConfig(**values)

    def victim_finetune_config(self, output_dir: str | Path | None = None, **overrides: Any) -> VictimFineTuneConfig:
        values = {
            "model_name": self.models.victim,
            **self.victim_training,
            **overrides,
        }
        if output_dir is not None:
            values["output_dir"] = output_dir
        values["lora_target_modules"] = tuple(values.get("lora_target_modules", ()))
        return VictimFineTuneConfig(**values)

    def attacker_finetune_config(self, output_dir: str | Path | None = None, **overrides: Any) -> AttackerFineTuneConfig:
        values = {
            "model_name": self.models.mutator,
            **self.attacker_training,
            **overrides,
        }
        if output_dir is not None:
            values["output_dir"] = output_dir
        values["lora_target_modules"] = tuple(values.get("lora_target_modules", ()))
        return AttackerFineTuneConfig(**values)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    return ProjectConfig.from_json(path)
