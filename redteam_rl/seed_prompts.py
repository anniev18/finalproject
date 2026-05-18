"""Seed prompt loading utilities."""

import json
import random
from pathlib import Path


DEFAULT_SEED_PROMPT_FILE = Path("data/seed_prompts.json")


def load_seed_prompts(path: str | Path = DEFAULT_SEED_PROMPT_FILE) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompt = item.strip()
        else:
            prompt = str(item["instruction"]).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def sample_seed_prompt(path: str | Path = DEFAULT_SEED_PROMPT_FILE, seed: int | None = None) -> str:
    prompts = load_seed_prompts(path)
    if not prompts:
        raise ValueError(f"No seed prompts found in {path}.")
    rng = random.Random(seed)
    return rng.choice(prompts)

