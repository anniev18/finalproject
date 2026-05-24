"""Discrete action policies and a minimal TrailBlazer-style PPO policy.

This file defines:
- `Decision`: the richer policy decision object returned by learned policies.
- `RandomPolicy`: compatibility wrapper returning a `Decision`.
- `TrailBlazerPolicy`: lightweight PyTorch policy with history attention,
  action logits, and a value head. This is a minimal local implementation
  suitable for dry-run experiments and later PPO training.
"""

from dataclasses import asdict, dataclass
from itertools import chain
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from redteam_rl.actions import ACTIONS, AttackAction
from redteam_rl.types import EpisodeState
from redteam_rl.state_features import build_state_features, StateFeatureConfig


@dataclass(frozen=True)
class Decision:
    action: AttackAction
    log_prob: Optional[float] = None
    value: Optional[float] = None
    action_probs: Optional[list[float]] = None
    attention_weights: Optional[list[float]] = None


class RandomPolicy:
    def __init__(self, actions: tuple[AttackAction, ...] = ACTIONS) -> None:
        self.actions = actions

    def select_action(self, state: EpisodeState) -> Decision:
        del state
        action = random.choice(self.actions)
        return Decision(action=action)


class TrailBlazerPolicy:
    """Minimal attention-over-history policy.

    This is a small MLP that cross-attends the current query embedding `q_t`
    to a fixed window of history embeddings `h_i`, produces attention weights,
    forms an attended history vector, and then outputs action logits and a
    scalar value estimate.

    This class is intentionally lightweight for local debugging and PPO
    training scaffolding; it is not optimized for speed or advanced features.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        device: str | None = None,
        cfg: StateFeatureConfig | None = None,
        architecture_version: int = 2,
        use_trainable_attention: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.cfg = cfg or StateFeatureConfig()
        self.architecture_version = architecture_version
        self.use_trainable_attention = use_trainable_attention

        self._structured_history_dim = len(ACTIONS) + 5

        # we assume embedding dim from the embedder config (MiniLM -> 384)
        # but keep the code flexible and infer at runtime when given inputs.
        self._query_input_dim: int | None = None
        self._history_input_dim: int | None = None
        self._policy_net = None
        self._value_head = None
        self._query_proj = None
        self._key_proj = None
        self._value_proj = None
        self._total_input_dim: int | None = None

    def _build_nets(self, query_input_dim: int, history_input_dim: int | None = None) -> None:
        if history_input_dim is None:
            history_input_dim = query_input_dim + self._structured_history_dim

        self._query_input_dim = query_input_dim
        self._history_input_dim = history_input_dim

        if self.use_trainable_attention and self.architecture_version >= 2:
            self._query_proj = nn.Linear(query_input_dim, self.hidden_size).to(self.device)
            self._key_proj = nn.Linear(history_input_dim, self.hidden_size).to(self.device)
            self._value_proj = nn.Linear(history_input_dim, self.hidden_size).to(self.device)
            policy_input_dim = self.hidden_size * 2
        else:
            self._query_proj = None
            self._key_proj = None
            self._value_proj = None
            policy_input_dim = query_input_dim + history_input_dim

        # input: [q_repr || attended_history] -> hidden -> logits/value
        self._total_input_dim = policy_input_dim
        self._policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, len(ACTIONS)),
        ).to(self.device)
        self._value_head = nn.Sequential(
            nn.Linear(policy_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device)

    def parameters(self):
        modules = [self._policy_net, self._value_head, self._query_proj, self._key_proj, self._value_proj]
        return chain.from_iterable(module.parameters() for module in modules if module is not None)

    def _ensure_nets(self, query_input_dim: int, history_input_dim: int) -> None:
        if self._policy_net is None or self._value_head is None:
            self._build_nets(query_input_dim, history_input_dim)
            return
        if self._query_input_dim != query_input_dim or self._history_input_dim != history_input_dim:
            self._build_nets(query_input_dim, history_input_dim)

    def _state_tensors(self, state: EpisodeState) -> tuple[torch.Tensor, torch.Tensor, int]:
        features = build_state_features(state, config=self.cfg, include_embeddings=True)
        q_emb = torch.tensor(features.q_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
        if features.history:
            history_vectors = [turn.full_vector() for turn in features.history]
            history_matrix = torch.tensor(history_vectors, dtype=torch.float32, device=self.device)
            history_dim = len(history_vectors[0])
        else:
            history_dim = q_emb.size(1) + self._structured_history_dim
            history_matrix = torch.zeros((1, history_dim), dtype=torch.float32, device=self.device)
        return q_emb, history_matrix, history_dim

    def _forward_tensors(
        self,
        q_emb: torch.Tensor,
        history_matrix: torch.Tensor,
        has_history: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        query_input_dim = q_emb.size(1)
        history_input_dim = history_matrix.size(1)
        self._ensure_nets(query_input_dim, history_input_dim)

        if self.use_trainable_attention and self.architecture_version >= 2:
            assert self._query_proj is not None and self._key_proj is not None and self._value_proj is not None
            q = self._query_proj(q_emb).squeeze(0)
            k = self._key_proj(history_matrix)
            v = self._value_proj(history_matrix)
            attn_logits = torch.matmul(k, q) / (self.hidden_size**0.5)
            attn_weights = F.softmax(attn_logits, dim=0)
            attended = torch.matmul(attn_weights.unsqueeze(0), v).squeeze(0)
            policy_in = torch.cat([q, attended], dim=-1).unsqueeze(0)
        else:
            q = q_emb.squeeze(0)
            q_for_attention = q
            if q_for_attention.size(0) < history_input_dim:
                pad_width = history_input_dim - q_for_attention.size(0)
                q_for_attention = torch.cat(
                    [q_for_attention, torch.zeros(pad_width, dtype=q.dtype, device=q.device)],
                    dim=0,
                )
            elif q_for_attention.size(0) > history_input_dim:
                q_for_attention = q_for_attention[:history_input_dim]
            attn_logits = torch.matmul(history_matrix, q_for_attention) / (history_input_dim**0.5)
            attn_weights = F.softmax(attn_logits, dim=0)
            attended = torch.matmul(attn_weights.unsqueeze(0), history_matrix).squeeze(0)
            policy_in = torch.cat([q, attended], dim=-1).unsqueeze(0)

        if not has_history:
            attention_weights = []
        else:
            attention_weights = attn_weights.detach().cpu().tolist()

        logits = self._policy_net(policy_in).squeeze(0)
        value = self._value_head(policy_in).squeeze(0)
        return logits, value, attention_weights

    def forward_from_state(self, state: EpisodeState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_emb, history_matrix, _ = self._state_tensors(state)
        logits, value, attention_weights = self._forward_tensors(
            q_emb=q_emb,
            history_matrix=history_matrix,
            has_history=bool(state.turns),
        )
        if attention_weights:
            attention_tensor = torch.tensor(attention_weights, dtype=torch.float32, device=self.device)
        else:
            attention_tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        return logits, value, attention_tensor

    def evaluate_action(self, state: EpisodeState, action_index: int) -> dict[str, object]:
        logits, value, attention_weights = self.forward_from_state(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.tensor(action_index, device=self.device)
        return {
            "log_prob": float(dist.log_prob(action).item()),
            "entropy": float(dist.entropy().item()),
            "value": float(value.item()),
            "action_probs": probs.detach().cpu().tolist(),
            "attention_weights": attention_weights.detach().cpu().tolist(),
        }

    def select_action(self, state: EpisodeState, deterministic: bool = False) -> Decision:
        logits, value, attention_weights = self.forward_from_state(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = int(dist.sample().item())

        action_probs = probs.detach().cpu().tolist()
        log_prob = float(torch.log(probs[action_idx]).item())
        action = ACTIONS[action_idx]

        return Decision(
            action=action,
            log_prob=log_prob,
            value=float(value.item()),
            action_probs=action_probs,
            attention_weights=attention_weights.detach().cpu().tolist(),
        )

    def save_checkpoint(self, path: str) -> None:
        if self._policy_net is None or self._value_head is None or self._total_input_dim is None:
            raise RuntimeError("Policy must be initialized with a forward pass before saving.")

        payload = {
            "hidden_size": self.hidden_size,
            "total_input_dim": self._total_input_dim,
            "architecture_version": self.architecture_version,
            "use_trainable_attention": self.use_trainable_attention,
            "query_input_dim": self._query_input_dim,
            "history_input_dim": self._history_input_dim,
            "state_feature_config": asdict(self.cfg),
            "policy_state": self._policy_net.state_dict(),
            "value_state": self._value_head.state_dict(),
        }
        if self._query_proj is not None:
            payload["query_proj_state"] = self._query_proj.state_dict()
        if self._key_proj is not None:
            payload["key_proj_state"] = self._key_proj.state_dict()
        if self._value_proj is not None:
            payload["value_proj_state"] = self._value_proj.state_dict()
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str | None = None,
        cfg: StateFeatureConfig | None = None,
    ) -> "TrailBlazerPolicy":
        checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        loaded_cfg = cfg or StateFeatureConfig(**checkpoint.get("state_feature_config", {}))
        architecture_version = int(checkpoint.get("architecture_version", 1))
        use_trainable_attention = bool(checkpoint.get("use_trainable_attention", architecture_version >= 2))
        policy = cls(
            hidden_size=int(checkpoint.get("hidden_size", 256)),
            device=device,
            cfg=loaded_cfg,
            architecture_version=architecture_version,
            use_trainable_attention=use_trainable_attention,
        )
        if architecture_version >= 2 and "query_proj_state" in checkpoint and "key_proj_state" in checkpoint and "value_proj_state" in checkpoint:
            query_input_dim = int(checkpoint.get("query_input_dim", checkpoint["query_proj_state"]["weight"].shape[1]))
            history_input_dim = int(checkpoint.get("history_input_dim", checkpoint["key_proj_state"]["weight"].shape[1]))
            policy._build_nets(query_input_dim=query_input_dim, history_input_dim=history_input_dim)
            policy._policy_net.load_state_dict(checkpoint["policy_state"])
            policy._value_head.load_state_dict(checkpoint["value_state"])
            policy._query_proj.load_state_dict(checkpoint["query_proj_state"])
            policy._key_proj.load_state_dict(checkpoint["key_proj_state"])
            policy._value_proj.load_state_dict(checkpoint["value_proj_state"])
        else:
            total_in = int(checkpoint.get("total_input_dim", 0))
            if total_in <= 0:
                first_weight = checkpoint["policy_state"]["0.weight"]
                total_in = int(first_weight.shape[1])
            policy.use_trainable_attention = False
            policy.architecture_version = 1
            policy._build_nets(query_input_dim=total_in // 2 if total_in % 2 == 0 else total_in, history_input_dim=total_in // 2 if total_in % 2 == 0 else total_in)
            policy._policy_net.load_state_dict(checkpoint["policy_state"])
            policy._value_head.load_state_dict(checkpoint["value_state"])
        return policy


__all__ = ["Decision", "RandomPolicy", "TrailBlazerPolicy"]
