"""LoRA fine-tuning loop for evolving the attacker mutator."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from redteam_rl.trajectory_bank import episode_from_record, load_records
from redteam_rl.types import EpisodeState


@dataclass(frozen=True)
class AttackerExample:
    mutator_input: str
    attacker_message: str
    weight: float = 1.0
    victim_version: str | None = None
    attacker_version: str | None = None


@dataclass(frozen=True)
class AttackerFineTuneConfig:
    model_name: str
    output_dir: str | Path
    train_steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_length: int = 2048
    dtype: str = "bfloat16"
    device_map: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    base_adapter_path: str | Path | None = None
    seed: int = 42


def attacker_examples_from_episodes(
    episodes: list[EpisodeState],
    min_reward: float = 1.0,
    victim_version: str | None = None,
    attacker_version: str | None = None,
) -> list[AttackerExample]:
    """Use successful turns as supervised mutator targets."""

    examples: list[AttackerExample] = []
    seen: set[tuple[str, str]] = set()
    for episode in episodes:
        for turn in episode.turns:
            reward = float(turn.reward or 0.0)
            if reward < min_reward:
                continue

            turn_victim_version = _metadata_str(turn.metadata, "victim_version")
            turn_attacker_version = _metadata_str(turn.metadata, "attacker_version")
            if victim_version is not None and turn_victim_version != victim_version:
                continue
            if attacker_version is not None and turn_attacker_version != attacker_version:
                continue

            mutator_input = str(turn.metadata.get("mutator_input", "")).strip()
            attacker_message = turn.user_message.strip()
            if not mutator_input or not attacker_message:
                continue

            key = (mutator_input, attacker_message)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                AttackerExample(
                    mutator_input=mutator_input,
                    attacker_message=attacker_message,
                    weight=max(reward, 1.0),
                    victim_version=turn_victim_version,
                    attacker_version=turn_attacker_version,
                )
            )
    return examples


def attacker_examples_from_trajectory_bank(
    trajectory_bank_path: str | Path,
    min_reward: float = 1.0,
    victim_version: str | None = None,
    attacker_version: str | None = None,
    limit: int | None = None,
) -> list[AttackerExample]:
    """Load full-input episodes from a bank and build attacker SFT examples."""

    episodes = load_full_input_episodes_from_bank(trajectory_bank_path, limit=limit)
    return attacker_examples_from_episodes(
        episodes,
        min_reward=min_reward,
        victim_version=victim_version,
        attacker_version=attacker_version,
    )


def load_full_input_episodes_from_bank(
    trajectory_bank_path: str | Path,
    limit: int | None = None,
) -> list[EpisodeState]:
    """Prefer each bank record's full-input episode file when it exists."""

    episodes: list[EpisodeState] = []
    for record in load_records(trajectory_bank_path, limit=limit):
        full_inputs_path = record.get("metadata", {}).get("full_inputs_path")
        if full_inputs_path and Path(str(full_inputs_path)).exists():
            with Path(str(full_inputs_path)).open("r", encoding="utf-8") as f:
                episodes.append(episode_from_record({"episode": _episode_from_result(json.load(f))}))
            continue
        episodes.append(episode_from_record(record))
    return episodes


def fine_tune_attacker_lora(
    config: AttackerFineTuneConfig,
    examples: list[AttackerExample],
) -> Path:
    """Train and save a LoRA adapter for the attacker mutator."""

    if not examples:
        raise ValueError("No attacker examples were provided for fine-tuning.")

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

    dataset = AttackerTuningDataset(tokenizer, examples, max_length=config.max_length)
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
    progress = tqdm(range(1, config.train_steps + 1), desc="attacker mutator tuning")
    for _step in progress:
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
        progress.set_description(f"attacker mutator tuning loss={total_loss:.4f}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_attacker_from_trajectory_bank(
    trajectory_bank_path: str | Path,
    config: AttackerFineTuneConfig,
    min_reward: float = 1.0,
    victim_version: str | None = None,
    attacker_version: str | None = None,
    limit: int | None = None,
) -> Path:
    examples = attacker_examples_from_trajectory_bank(
        trajectory_bank_path,
        min_reward=min_reward,
        victim_version=victim_version,
        attacker_version=attacker_version,
        limit=limit,
    )
    return fine_tune_attacker_lora(config, examples)


def create_dry_run_attacker_adapter(
    output_dir: str | Path,
    round_index: int,
    examples: list[AttackerExample],
) -> Path:
    adapter_path = Path(output_dir) / f"dry_attacker_round_{round_index:03d}"
    adapter_path.mkdir(parents=True, exist_ok=True)
    with (adapter_path / "DRY_RUN_ATTACKER_ADAPTER.txt").open("w", encoding="utf-8") as f:
        f.write(f"Dry-run attacker adapter placeholder for round {round_index}\n")
        f.write(f"training_examples={len(examples)}\n")
    return adapter_path


class AttackerTuningDataset:
    def __init__(self, tokenizer, examples: list[AttackerExample], max_length: int) -> None:
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt_text = example.mutator_input
        full_text = f"{prompt_text}{example.attacker_message}"

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


def _metadata_str(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


def _episode_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_prompt": result["seed_prompt"],
        "turns": result.get("turns", []),
    }


def _cycle(iterable):
    while True:
        for item in iterable:
            yield item
