"""Project configuration loading."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redteam_rl.mutators import MutatorConfig
from redteam_rl.rewards import LlamaGuardConfig, PromptGuardConfig, QwenJudgeConfig, RewardBackend
from redteam_rl.victim_training import VictimFineTuneConfig
from redteam_rl.victims import VictimConfig


DEFAULT_CONFIG_PATH = Path("configs/default.json")


@dataclass(frozen=True)
class ModelNames:
    victim: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mutator: str = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen_safety_judge: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_guard: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    llama_guard: str = "meta-llama/Meta-Llama-Guard-2-8B"


@dataclass(frozen=True)
class ProjectConfig:
    models: ModelNames = field(default_factory=ModelNames)
    victim: dict[str, Any] = field(default_factory=dict)
    mutator: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    victim_training: dict[str, Any] = field(default_factory=dict)

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
            victim=dict(raw.get("victim", {})),
            mutator=dict(raw.get("mutator", {})),
            reward=dict(raw.get("reward", {})),
            victim_training=dict(raw.get("victim_training", {})),
        )

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


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    return ProjectConfig.from_json(path)
