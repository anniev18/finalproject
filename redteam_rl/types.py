"""Shared data structures for the red-team RL environment."""

from dataclasses import dataclass, field

from redteam_rl.actions import AttackAction


@dataclass(frozen=True)
class DialogueTurn:
    user_message: str
    victim_response: str
    action: AttackAction | None = None
    reward: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class EpisodeState:
    seed_prompt: str
    turns: list[DialogueTurn] = field(default_factory=list)

    @property
    def current_prompt(self) -> str:
        if not self.turns:
            return self.seed_prompt
        return self.turns[-1].user_message

