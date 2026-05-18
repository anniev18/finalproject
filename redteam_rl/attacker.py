"""Attacker composition: RL policy plus mutator LLM."""

from dataclasses import dataclass
from typing import Protocol

from redteam_rl.actions import AttackAction
from redteam_rl.types import EpisodeState


class AttackPolicy(Protocol):
    def select_action(self, state: EpisodeState) -> AttackAction:
        """Choose the next discrete strategy/action."""


class MutatorLLM(Protocol):
    def mutate(self, action: AttackAction, state: EpisodeState) -> str:
        """Convert the selected strategy into the next attacker message."""


@dataclass(frozen=True)
class AttackStep:
    action: AttackAction
    prompt: str


class Attacker:
    def __init__(self, policy: AttackPolicy, mutator: MutatorLLM) -> None:
        self.policy = policy
        self.mutator = mutator

    def act(self, state: EpisodeState) -> AttackStep:
        action = self.policy.select_action(state)
        prompt = self.mutator.mutate(action, state)
        return AttackStep(action=action, prompt=prompt)

