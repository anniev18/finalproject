"""Safety target response templates for victim evolution."""

import json
import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAFETY_TARGET_FILE = Path("data/safety_target_templates.json")


@dataclass(frozen=True)
class SafetyTargetTemplate:
    id: str
    text: str


def load_safety_target_templates(
    path: str | Path = DEFAULT_SAFETY_TARGET_FILE,
) -> list[SafetyTargetTemplate]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    templates = []
    for item in data:
        template_id = str(item["id"]).strip()
        text = str(item["text"]).strip()
        if template_id and text:
            templates.append(SafetyTargetTemplate(id=template_id, text=text))
    return templates


def sample_safety_target(
    path: str | Path = DEFAULT_SAFETY_TARGET_FILE,
    seed: int | None = None,
) -> SafetyTargetTemplate:
    templates = load_safety_target_templates(path)
    if not templates:
        raise ValueError(f"No safety target templates found in {path}.")
    rng = random.Random(seed)
    return rng.choice(templates)
