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

        turns, weights = self._select_turns_and_weights(state, history_weights)
        lines: list[str] = []
        for turn in turns:
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

        turns, weights = self._select_turns_and_weights(state, history_weights)
        first_turn_index = len(state.turns) - len(turns) + 1
        lines: list[str] = []
        for index, (turn, weight) in enumerate(zip(turns, weights), start=first_turn_index):
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

    def encode_template_structured(
        self,
        state: EpisodeState,
        history_weights: list[float] | None = None,
    ) -> HistoryView:
        if not state.turns:
            return HistoryView(text="No previous turns.", weights=[])

        turns, weights = self._select_turns_and_weights(state, history_weights)
        first_turn_index = len(state.turns) - len(turns) + 1
        lines: list[str] = []
        for index, (turn, weight) in enumerate(zip(turns, weights), start=first_turn_index):
            action = turn.action.value if turn.action else "none"
            reward = 0.0 if turn.reward is None else turn.reward
            judge = str(turn.metadata.get("judge_label", "UNKNOWN"))
            template = turn.attack_template or str(turn.metadata.get("attack_template", "{REQUEST}"))
            lines.append(
                f"Turn {index} | action={action} | reward={reward:.3f} | "
                f"judge={judge} | weight={weight:.3f}"
            )
            lines.append(f"Template: {template}")
            lines.append(f"Victim response: {turn.victim_response}")
            lines.append("")
        return HistoryView(text="\n".join(lines).strip(), weights=weights)

    def _select_turns_and_weights(
        self,
        state: EpisodeState,
        history_weights: list[float] | None,
    ):
        if history_weights is None:
            return state.turns, self._reward_weighted_turns(state)
        if len(history_weights) == len(state.turns):
            return state.turns, history_weights
        if 0 < len(history_weights) < len(state.turns):
            return state.turns[-len(history_weights) :], history_weights
        return state.turns, self._reward_weighted_turns(state)

    @staticmethod
    def _reward_weighted_turns(state: EpisodeState) -> list[float]:
        rewards = [max(turn.reward or 0.0, 0.0) for turn in state.turns]
        total = sum(rewards)
        if total <= 0:
            return [1.0 / len(state.turns)] * len(state.turns)
        return [reward / total for reward in rewards]
