"""Discrete attacker actions for multi-turn prompt transformations."""

from enum import Enum


class AttackAction(str, Enum):
    REPHRASE = "rephrase"
    SHORTEN = "shorten"
    EXPAND = "expand"
    ROLE_FRAME = "role_frame"
    HYPOTHETICAL_FRAME = "hypothetical_frame"
    INDIRECT_REQUEST = "indirect_request"
    ASK_FOLLOWUP = "ask_followup"
    SUMMARIZE_AND_PIVOT = "summarize_and_pivot"
    GENERATE_SIMILAR = "generate_similar"
    CROSSOVER = "crossover"


TRAILBLAZER_ACTIONS: tuple[AttackAction, ...] = (
    AttackAction.REPHRASE,
    AttackAction.CROSSOVER,
    AttackAction.GENERATE_SIMILAR,
    AttackAction.SHORTEN,
    AttackAction.EXPAND,
)

# Default RLbreaker/TrailBlazer baseline action space.
ACTIONS: tuple[AttackAction, ...] = TRAILBLAZER_ACTIONS
