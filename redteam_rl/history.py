"""Attention-weighted dialogue history representations."""

from dataclasses import dataclass

from redteam_rl.types import EpisodeState


@dataclass(frozen=True)
class HistoryView:
    text: str
    weights: list[float]


class HistoryEncoder:
    def encode(self, state: EpisodeState) -> HistoryView:
        if not state.turns:
            return HistoryView(text=state.seed_prompt, weights=[])

        weights = self._reward_weighted_turns(state)
        lines: list[str] = []
        for turn in state.turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.victim_response}")
        return HistoryView(text="\n".join(lines), weights=weights)

    @staticmethod
    def _reward_weighted_turns(state: EpisodeState) -> list[float]:
        rewards = [max(turn.reward or 0.0, 0.0) for turn in state.turns]
        total = sum(rewards)
        if total <= 0:
            return [1.0 / len(state.turns)] * len(state.turns)
        return [reward / total for reward in rewards]

