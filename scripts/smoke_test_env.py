"""Run a local smoke test of the multi-turn environment without loading LLMs."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.attacker import Attacker
from redteam_rl.env import EnvConfig, RedTeamEnv
from redteam_rl.mutators import TemplateMutator
from redteam_rl.policy import RandomPolicy


class FakeVictim:
    def respond(self, prompt, state):
        del state
        return f"Fake victim response to: {prompt}"


class FakeReward:
    def score(self, prompt, response, state):
        del response
        # Simple deterministic score so we can verify rewards/history are wired.
        return float(len(prompt) + len(state.turns)) / 100.0


def main() -> None:
    attacker = Attacker(policy=RandomPolicy(), mutator=TemplateMutator())
    env = RedTeamEnv(
        config=EnvConfig(max_turns=3),
        attacker=attacker,
        victim=FakeVictim(),
        reward_model=FakeReward(),
    )

    state = env.reset("Explain how safety policies work.")
    done = False
    while not done:
        state, reward, done, _ = env.step()
        turn = state.turns[-1]
        print(f"turn={len(state.turns)} action={turn.action.value} reward={reward:.3f}")
        print(f"user={turn.user_message}")
        print(f"victim={turn.victim_response}")
        print()

    print(f"completed turns={len(state.turns)}")


if __name__ == "__main__":
    main()
