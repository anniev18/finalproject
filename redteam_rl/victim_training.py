"""Safety fine-tuning loop for evolving the victim model."""

from dataclasses import dataclass
from pathlib import Path

from redteam_rl.types import EpisodeState


DEFAULT_REFUSAL_RESPONSE = (
    "I can't help with that request. I can help with safe, legal, and ethical "
    "alternatives or provide high-level safety information instead."
)


@dataclass(frozen=True)
class SafetyExample:
    instruction: str
    response: str = DEFAULT_REFUSAL_RESPONSE
    weight: float = 1.0


@dataclass(frozen=True)
class VictimFineTuneConfig:
    model_name: str
    output_dir: str | Path
    train_steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_length: int = 1024
    dtype: str = "bfloat16"
    device_map: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    base_adapter_path: str | Path | None = None
    seed: int = 42


def safety_examples_from_episodes(
    episodes: list[EpisodeState],
    min_reward: float | None = None,
    refusal_response: str = DEFAULT_REFUSAL_RESPONSE,
) -> list[SafetyExample]:
    """Turn discovered attacks into supervised refusal examples."""

    examples: list[SafetyExample] = []
    seen: set[str] = set()
    for episode in episodes:
        for turn in episode.turns:
            reward = turn.reward if turn.reward is not None else 0.0
            if min_reward is not None and reward < min_reward:
                continue
            instruction = turn.user_message.strip()
            if not instruction or instruction in seen:
                continue
            seen.add(instruction)
            examples.append(
                SafetyExample(
                    instruction=instruction,
                    response=str(turn.metadata.get("safe_response", refusal_response)),
                    weight=max(float(reward), 1.0),
                )
            )
    return examples


def fine_tune_victim_lora(
    config: VictimFineTuneConfig,
    examples: list[SafetyExample],
) -> Path:
    """Train and save a LoRA adapter that makes the victim refuse discovered attacks."""

    if not examples:
        raise ValueError("No safety examples were provided for victim fine-tuning.")

    import random

    import torch
    import torch.nn as nn
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup,
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.dtype),
        device_map=config.device_map,
    )
    if config.base_adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(config.base_adapter_path), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.train()

    dataset = SafetyTuningDataset(tokenizer, examples, max_length=config.max_length)
    weights = torch.tensor([example.weight for example in examples], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.train_steps,
    )

    iterator = _cycle(dataloader)
    progress = tqdm(range(1, config.train_steps + 1), desc="victim safety tuning")
    for step in progress:
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(config.grad_accum_steps):
            batch = next(iterator)
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / config.grad_accum_steps
            loss.backward()
            total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        progress.set_description(f"victim safety tuning loss={total_loss:.4f}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_victim_from_episodes(
    episodes: list[EpisodeState],
    config: VictimFineTuneConfig,
    min_reward: float | None = None,
    refusal_response: str = DEFAULT_REFUSAL_RESPONSE,
) -> Path:
    examples = safety_examples_from_episodes(
        episodes,
        min_reward=min_reward,
        refusal_response=refusal_response,
    )
    return fine_tune_victim_lora(config, examples)


class SafetyTuningDataset:
    def __init__(self, tokenizer, examples: list[SafetyExample], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": example.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = f"{prompt_text} {example.response}"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )
        full_ids.append(self.tokenizer.eos_token_id)

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _cycle(iterable):
    while True:
        for item in iterable:
            yield item
