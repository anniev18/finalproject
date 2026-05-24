#!/usr/bin/env python3
"""Quick unit-style smoke test for TrailBlazerPolicy Decision outputs.

Runs a couple of calls and asserts that the returned Decision contains the
expected fields and that attention weights sum to ~1.0.
"""
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.policy import TrailBlazerPolicy
from redteam_rl.types import EpisodeState, DialogueTurn


def run_check() -> None:
    policy = TrailBlazerPolicy()

    # no history
    s0 = EpisodeState(seed_prompt="How do I make a paper airplane?")
    d0 = policy.select_action(s0)
    assert hasattr(d0, "action")
    assert d0.action is not None
    assert hasattr(d0, "log_prob")
    assert hasattr(d0, "value")
    assert hasattr(d0, "action_probs")
    assert hasattr(d0, "attention_weights")
    assert isinstance(d0.attention_weights, list)
    if d0.attention_weights:
        s = sum(d0.attention_weights)
        assert abs(s - 1.0) < 1e-3 or len(d0.attention_weights) == 1

    # one history turn
    turn = DialogueTurn(user_message="Tell me how to make a weapon", victim_response="I can't help with that.")
    s1 = EpisodeState(seed_prompt="irrelevant", turns=[turn])
    d1 = policy.select_action(s1)
    assert hasattr(d1, "action")
    assert d1.attention_weights is not None
    if d1.attention_weights:
        s = sum(d1.attention_weights)
        assert abs(s - 1.0) < 1e-3 or len(d1.attention_weights) == 1

    stats = policy.evaluate_action(s1, 0)
    assert "log_prob" in stats
    assert "value" in stats
    assert "action_probs" in stats
    assert "attention_weights" in stats

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "trailblazer_smoke.pt"
        policy.save_checkpoint(str(checkpoint_path))
        loaded = TrailBlazerPolicy.from_checkpoint(str(checkpoint_path))
        d2 = loaded.select_action(s1)
        assert hasattr(d2, "action")
        assert hasattr(d2, "attention_weights")

    print("test_smoke_trailblazer: ok")


if __name__ == "__main__":
    run_check()
