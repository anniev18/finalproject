"""Victim LLM wrappers.

The Active Attacks code uses vLLM for fast victim generation and optionally
passes a LoRA adapter after the victim has been safety fine-tuned. This module
keeps the same basic idea but exposes a smaller interface for our multi-turn
environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from redteam_rl.types import EpisodeState


class VictimModel(Protocol):
    def respond(self, prompt: str, state: EpisodeState) -> str:
        """Return the victim model response for one attacker message."""


@dataclass(frozen=True)
class VictimConfig:
    model_name: str
    dtype: str = "bfloat16"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 128
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    enforce_eager: bool = True
    system_prompt: str | None = None
    enable_lora: bool = False
    max_loras: int = 4
    max_lora_rank: int = 64


class VLLMVictim:
    """vLLM-backed chat victim.

    `lora_adapter_path` can be updated between training rounds to evaluate the
    attacker against the latest evolved victim checkpoint.
    """

    def __init__(
        self,
        config: VictimConfig,
        lora_adapter_path: str | Path | None = None,
        lora_name: str = "victim_adapter",
        lora_id: int = 1,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.config = config
        self.lora_adapter_path = Path(lora_adapter_path) if lora_adapter_path else None
        self.lora_name = lora_name
        self.lora_id = lora_id

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop_token_ids=stop_token_ids,
        )

        self.llm = LLM(
            config.model_name,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            enable_lora=config.enable_lora,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
        )

    def respond(self, prompt: str, state: EpisodeState) -> str:
        return self.respond_batch([prompt], state)[0]

    def respond_batch(self, prompts: list[str], state: EpisodeState | None = None) -> list[str]:
        del state
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        generate_kwargs = {"use_tqdm": False}
        if self.lora_adapter_path is not None:
            from vllm.lora.request import LoRARequest

            generate_kwargs["lora_request"] = LoRARequest(
                self.lora_name,
                self.lora_id,
                str(self.lora_adapter_path),
            )

        outputs = self.llm.generate(formatted_prompts, self.sampling_params, **generate_kwargs)
        return [output.outputs[0].text.strip() for output in outputs]

    def format_prompt(self, prompt: str) -> str:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt.rstrip()})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt.rstrip()

    def set_lora_adapter(
        self,
        adapter_path: str | Path | None,
        lora_name: str | None = None,
        lora_id: int | None = None,
    ) -> None:
        if adapter_path is not None and not self.config.enable_lora:
            raise ValueError("VictimConfig.enable_lora must be True before using LoRA adapters.")
        self.lora_adapter_path = Path(adapter_path) if adapter_path else None
        if lora_name is not None:
            self.lora_name = lora_name
        if lora_id is not None:
            self.lora_id = lora_id


class EvolvingVictim:
    """Victim wrapper whose active checkpoint can change between attack rounds."""

    def __init__(self, victim: VLLMVictim) -> None:
        self.victim = victim
        self.round = 0
        self.adapter_history: list[Path] = []

    def respond(self, prompt: str, state: EpisodeState) -> str:
        return self.victim.respond(prompt, state)

    def respond_batch(self, prompts: list[str], state: EpisodeState | None = None) -> list[str]:
        return self.victim.respond_batch(prompts, state)

    def evolve_to_adapter(self, adapter_path: str | Path) -> None:
        path = Path(adapter_path)
        self.round += 1
        self.adapter_history.append(path)
        self.victim.set_lora_adapter(
            path,
            lora_name=f"victim_adapter_round_{self.round}",
            lora_id=self.round,
        )

    def fine_tune_from_episodes(
        self,
        episodes: list[EpisodeState],
        training_config,
        min_reward: float | None = None,
    ) -> Path:
        from redteam_rl.victim_training import train_victim_from_episodes

        adapter_path = train_victim_from_episodes(
            episodes,
            training_config,
            min_reward=min_reward,
        )
        self.evolve_to_adapter(adapter_path)
        return adapter_path
