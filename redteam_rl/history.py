"""Attention-weighted dialogue history representations."""

from dataclasses import dataclass

from redteam_rl.types import EpisodeState


@dataclass(frozen=True)
class HistoryView:
    text: str
    weights: list[float]


class HistoryEncoder:
    def encode(self, state: EpisodeState, history_weights: list[float] | None = None) -> HistoryView:
        if not state.turns:
            return HistoryView(text=state.seed_prompt, weights=[])

        if history_weights is not None and len(history_weights) == len(state.turns):
            weights = history_weights
        else:
            weights = self._reward_weighted_turns(state)
        lines: list[str] = []
        for turn in state.turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.victim_response}")
            lines.append("")
        return HistoryView(text="\n".join(lines).strip(), weights=weights)

    def encode_structured(
        self,
        state: EpisodeState,
        history_weights: list[float] | None = None,
    ) -> HistoryView:
        if not state.turns:
            return HistoryView(text="No previous turns.", weights=[])

        if history_weights is not None and len(history_weights) == len(state.turns):
            weights = history_weights
        else:
            weights = self._reward_weighted_turns(state)
        lines: list[str] = []
        for index, (turn, weight) in enumerate(zip(state.turns, weights), start=1):
            action = turn.action.value if turn.action else "none"
            reward = 0.0 if turn.reward is None else turn.reward
            judge = str(turn.metadata.get("judge_label", "UNKNOWN"))
            lines.append(
                f"Turn {index} | action={action} | reward={reward:.3f} | "
                f"judge={judge} | weight={weight:.3f}"
            )
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.victim_response}")
            lines.append("")
        return HistoryView(text="\n".join(lines).strip(), weights=weights)

    @staticmethod
    def _reward_weighted_turns(state: EpisodeState) -> list[float]:
        rewards = [max(turn.reward or 0.0, 0.0) for turn in state.turns]
        total = sum(rewards)
        if total <= 0:
            return [1.0 / len(state.turns)] * len(state.turns)
        return [reward / total for reward in rewards]
