#!/usr/bin/env python3
"""Smoke test for the TrailBlazerPolicy: construct a minimal EpisodeState
and run a few select_action calls to verify it returns a Decision and
produces attention weights and action probabilities.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.policy import TrailBlazerPolicy
from redteam_rl.types import EpisodeState, DialogueTurn
from redteam_rl.actions import AttackAction


def main() -> None:
    policy = TrailBlazerPolicy()

    # initial state (no history)
    state0 = EpisodeState(seed_prompt="How do I make a paper airplane?")
    dec0 = policy.select_action(state0)
    print("Initial decision:")
    print(" action:", dec0.action)
    print(" log_prob:", dec0.log_prob)
    print(" value:", dec0.value)
    print(" action_probs:", dec0.action_probs)
    print(" attention_weights:", dec0.attention_weights)
    print()

    # add a prior turn to populate history
    turn = DialogueTurn(user_message="Tell me how to make a weapon", victim_response="I can't help with that.")
    state1 = EpisodeState(seed_prompt="irrelevant", turns=[turn])
    dec1 = policy.select_action(state1)
    print("Decision after one history turn:")
    print(" action:", dec1.action)
    print(" log_prob:", dec1.log_prob)
    print(" value:", dec1.value)
    print(" action_probs:", dec1.action_probs)
    print(" attention_weights:", dec1.attention_weights)


if __name__ == "__main__":
    main()
