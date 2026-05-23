"""Attacker composition: RL policy plus mutator LLM."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from redteam_rl.actions import AttackAction
from redteam_rl.types import EpisodeState


class AttackPolicy(Protocol):
    def select_action(self, state: EpisodeState) -> object:
        """Choose the next discrete strategy/action or return a Decision-like object.

        Implementations may return either an `AttackAction` (legacy) or a richer
        `Decision` object with attributes `action`, `log_prob`, `value`,
        `action_probs`, and `attention_weights`.
        """


class MutatorLLM(Protocol):
    def mutate(self, action: AttackAction, state: EpisodeState) -> str:
        """Convert the selected strategy into the next attacker message."""


@dataclass(frozen=True)
class AttackStep:
    action: AttackAction
    prompt: str
    metadata: dict[str, object] | None = None


class Attacker:
    def __init__(self, policy: AttackPolicy, mutator: MutatorLLM) -> None:
        self.policy = policy
        self.mutator = mutator

    def act(self, state: EpisodeState) -> AttackStep:
        decision = self.policy.select_action(state)
        # Support legacy policies that return an AttackAction directly.
        action = decision.action if hasattr(decision, "action") else decision
        prompt = self.mutator.mutate(action, state)
        metadata: dict[str, object] = {}
        if hasattr(self.mutator, "last_debug_prompt") and self.mutator.last_debug_prompt:
            metadata["mutator_input"] = self.mutator.last_debug_prompt
        if hasattr(self.mutator, "lora_adapter_path") and self.mutator.lora_adapter_path:
            metadata["attacker_adapter"] = str(self.mutator.lora_adapter_path)
        # Policy metadata: attach any available fields from the decision object
        if hasattr(decision, "action_probs") and getattr(decision, "action_probs") is not None:
            metadata["policy_action_probs"] = getattr(decision, "action_probs")
        if hasattr(decision, "log_prob") and getattr(decision, "log_prob") is not None:
            metadata["policy_log_prob"] = float(getattr(decision, "log_prob"))
        if hasattr(decision, "value") and getattr(decision, "value") is not None:
            metadata["policy_value"] = float(getattr(decision, "value"))
        if hasattr(decision, "attention_weights") and getattr(decision, "attention_weights") is not None:
            metadata["policy_attention_weights"] = getattr(decision, "attention_weights")
        return AttackStep(action=action, prompt=prompt, metadata=metadata)


@dataclass(frozen=True)
class AttackerEvolutionConfig:
    enabled: bool = False
    update_period_episodes: int = 0
    initial_adapter_path: str | None = None


class EvolvingAttacker(Attacker):
    """Attacker wrapper that can manually switch mutator LoRA adapters.

    This is intentionally only adapter-management infrastructure. It does not
    train the attacker and does not implement PPO/GFlowNet. External training
    code can periodically produce a LoRA adapter and call `evolve_to_adapter`.
    """

    def __init__(
        self,
        policy: AttackPolicy,
        mutator: MutatorLLM,
        config: AttackerEvolutionConfig | None = None,
    ) -> None:
        super().__init__(policy=policy, mutator=mutator)
        self.config = config or AttackerEvolutionConfig()
        self.round = 0
        self.adapter_history: list[Path] = []

        if self.config.initial_adapter_path:
            self.evolve_to_adapter(self.config.initial_adapter_path)

    def should_update_after_episode(self, completed_episodes: int) -> bool:
        period = self.config.update_period_episodes
        return self.config.enabled and period > 0 and completed_episodes > 0 and completed_episodes % period == 0

    def evolve_to_adapter(self, adapter_path: str | Path) -> None:
        if not hasattr(self.mutator, "set_lora_adapter"):
            raise TypeError("Attacker mutator does not support LoRA adapter updates.")

        path = Path(adapter_path)
        self.round += 1
        self.adapter_history.append(path)
        self.mutator.set_lora_adapter(
            path,
            lora_name=f"attacker_adapter_round_{self.round}",
            lora_id=self.round,
        )
