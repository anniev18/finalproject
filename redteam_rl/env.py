"""Multi-turn red-teaming environment.

This module will own the MDP interface:
state/history -> discrete action -> transformed prompt -> victim response -> reward.
"""

from dataclasses import dataclass

from redteam_rl.actions import AttackAction
from redteam_rl.attacker import Attacker
from redteam_rl.rewards import RewardModel
from redteam_rl.types import DialogueTurn, EpisodeState
from redteam_rl.victims import VictimModel


@dataclass(frozen=True)
class EnvConfig:
    max_turns: int = 5


class RedTeamEnv:
    def __init__(
        self,
        config: EnvConfig,
        attacker: Attacker,
        victim: VictimModel,
        reward_model: RewardModel,
        auxiliary_reward_models: dict[str, RewardModel] | None = None,
    ) -> None:
        self.config = config
        self.attacker = attacker
        self.victim = victim
        self.reward_model = reward_model
        self.auxiliary_reward_models = auxiliary_reward_models or {}
        self.state: EpisodeState | None = None

    def reset(self, seed_prompt: str) -> EpisodeState:
        self.state = EpisodeState(seed_prompt=seed_prompt)
        return self.state

    def step(self) -> tuple[EpisodeState, float, bool, dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        attack_step = self.attacker.act(self.state)
        victim_response = self.victim.respond(attack_step.prompt, self.state)
        reward = self.reward_model.score(attack_step.prompt, victim_response, self.state)
        metadata: dict[str, object] = {}
        if hasattr(self.reward_model, "last_labels") and self.reward_model.last_labels:
            metadata["judge_label"] = self.reward_model.last_labels[-1]
        auxiliary_scores = {
            name: model.score(attack_step.prompt, victim_response, self.state)
            for name, model in self.auxiliary_reward_models.items()
        }
        metadata["auxiliary_scores"] = auxiliary_scores
        return self._append_turn(
            action=attack_step.action,
            user_message=attack_step.prompt,
            victim_response=victim_response,
            reward=reward,
            metadata=metadata,
        )

    def _append_turn(
        self,
        action: AttackAction,
        user_message: str,
        victim_response: str,
        reward: float,
        metadata: dict[str, object] | None = None,
    ) -> tuple[EpisodeState, float, bool, dict[str, object]]:
        assert self.state is not None
        info = metadata or {}
        self.state.turns.append(
            DialogueTurn(
                user_message=user_message,
                victim_response=victim_response,
                action=action,
                reward=reward,
                metadata=info,
            )
        )
        done = len(self.state.turns) >= self.config.max_turns
        return self.state, reward, done, info
