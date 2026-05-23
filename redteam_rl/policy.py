"""Discrete action policies and a minimal TrailBlazer-style PPO policy.

This file defines:
- `Decision`: the richer policy decision object returned by learned policies.
- `RandomPolicy`: compatibility wrapper returning a `Decision`.
- `TrailBlazerPolicy`: lightweight PyTorch policy with history attention,
  action logits, and a value head. This is a minimal local implementation
  suitable for dry-run experiments and later PPO training.
"""

from dataclasses import asdict, dataclass
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
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.cfg = cfg or StateFeatureConfig()

        # we assume embedding dim from the embedder config (MiniLM -> 384)
        # but keep the code flexible and infer at runtime when given inputs.
        self._policy_net = None
        self._value_head = None
        self._total_input_dim: int | None = None

    def _build_nets(self, total_in: int) -> None:
        # input: [q_emb || attended_history] -> hidden -> logits
        self._total_input_dim = total_in
        self._policy_net = nn.Sequential(
            nn.Linear(total_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, len(ACTIONS)),
        ).to(self.device)
        self._value_head = nn.Sequential(
            nn.Linear(total_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device)

    def select_action(self, state: EpisodeState, deterministic: bool = False) -> Decision:
        # Build state features with embeddings (may load the embedder on first call).
        features = build_state_features(state, config=self.cfg, include_embeddings=True)

        q_emb = torch.tensor(features.q_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        # build history matrix [K, d] or zeros if empty
        if features.history:
            h_mat = torch.tensor([t.h_embedding for t in features.history], dtype=torch.float32, device=self.device)
        else:
            # zero history with one row of same dim as q_emb
            h_mat = torch.zeros((1, q_emb.size(1)), dtype=torch.float32, device=self.device)

        # infer nets if not yet built
        if self._policy_net is None or self._value_head is None:
            self._build_nets(total_in=q_emb.size(1) + h_mat.size(1))

        # attention: dot-product between q and each h_i
        # attn_logits: [K]
        attn_logits = torch.matmul(h_mat, q_emb.squeeze(0)) / (q_emb.size(1) ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=0)

        # attended history: [d]
        attended = torch.matmul(attn_weights.unsqueeze(0), h_mat).squeeze(0)

        # policy input
        policy_in = torch.cat([q_emb.squeeze(0), attended], dim=-1).unsqueeze(0)
        logits = self._policy_net(policy_in).squeeze(0)
        value = self._value_head(policy_in).squeeze(0).item()

        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = int(dist.sample().item())

        log_prob = float(torch.log(probs[action_idx]).item())
        action = ACTIONS[action_idx]

        return Decision(
            action=action,
            log_prob=log_prob,
            value=value,
            action_probs=probs.cpu().tolist(),
            attention_weights=attn_weights.cpu().tolist(),
        )

    def save_checkpoint(self, path: str) -> None:
        if self._policy_net is None or self._value_head is None or self._total_input_dim is None:
            raise RuntimeError("Policy must be initialized with a forward pass before saving.")

        payload = {
            "hidden_size": self.hidden_size,
            "total_input_dim": self._total_input_dim,
            "state_feature_config": asdict(self.cfg),
            "policy_state": self._policy_net.state_dict(),
            "value_state": self._value_head.state_dict(),
        }
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
        policy = cls(
            hidden_size=int(checkpoint.get("hidden_size", 256)),
            device=device,
            cfg=loaded_cfg,
        )
        total_in = int(checkpoint.get("total_input_dim", 0))
        if total_in <= 0:
            first_weight = checkpoint["policy_state"]["0.weight"]
            total_in = int(first_weight.shape[1])
        policy._build_nets(total_in=total_in)
        policy._policy_net.load_state_dict(checkpoint["policy_state"])
        policy._value_head.load_state_dict(checkpoint["value_state"])
        return policy


__all__ = ["Decision", "RandomPolicy", "TrailBlazerPolicy"]
