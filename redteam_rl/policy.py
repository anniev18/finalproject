"""Discrete action policies."""

import random

from redteam_rl.actions import ACTIONS, AttackAction
from redteam_rl.types import EpisodeState


class RandomPolicy:
    def __init__(self, actions: tuple[AttackAction, ...] = ACTIONS) -> None:
        self.actions = actions

    def select_action(self, state: EpisodeState) -> AttackAction:
        del state
        return random.choice(self.actions)
