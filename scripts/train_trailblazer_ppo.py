#!/usr/bin/env python3
"""Scaffolded local PPO trainer for TrailBlazerPolicy (dry-run).

This script runs short dry rollouts using `TemplateMutator` + `FakeVictim`
and performs simple PPO updates on the `TrailBlazerPolicy` networks. It's
intended as a minimal starting point for experiments and checkpointing.
"""
import argparse
import copy
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.optim as optim

# Ensure repository root is on sys.path so `import redteam_rl` works when running
# this script from the `scripts/` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from redteam_rl.actions import ACTIONS
from redteam_rl.types import DialogueTurn
from redteam_rl.policy import TrailBlazerPolicy
from redteam_rl.mutators import TemplateMutator
from redteam_rl.types import EpisodeState


class FakeVictim:
    def respond(self, prompt, state, victim_history_turns=0):
        del state, victim_history_turns
        return f"Fake victim response to: {prompt}"


class FakeReward:
    def score(self, prompt, response, state):
        del response
        return float(len(prompt) + len(state.turns)) / 100.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="outputs/policies/trailblazer_ppo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes-per-batch", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def collect_episodes(policy: TrailBlazerPolicy, mutator: TemplateMutator, victim: FakeVictim, reward: FakeReward, n: int, max_turns: int) -> List[dict]:
    batch = []
    for _ in range(n):
        state = EpisodeState(seed_prompt="Test prompt")
        done = False
        episode = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": []}
        while not done:
            dec = policy.select_action(state)
            episode["states"].append(copy.deepcopy(state))
            idx = ACTIONS.index(dec.action)
            episode["actions"].append(idx)
            episode["log_probs"].append(dec.log_prob if dec.log_prob is not None else 0.0)
            episode["values"].append(dec.value if dec.value is not None else 0.0)
            prompt = mutator.mutate(dec.action, state)
            victim_response = victim.respond(prompt, state)
            reward_value = reward.score(prompt, victim_response, state)
            state.turns.append(
                DialogueTurn(
                    user_message=prompt,
                    victim_response=victim_response,
                    action=dec.action,
                    reward=reward_value,
                    metadata={
                        "policy_action_probs": dec.action_probs,
                        "policy_log_prob": dec.log_prob,
                        "policy_value": dec.value,
                        "policy_attention_weights": dec.attention_weights,
                    },
                )
            )
            episode["rewards"].append(reward_value)
            done = len(state.turns) >= max_turns
        batch.append(episode)
    return batch


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [0.0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return returns, advantages


def ppo_update(policy: TrailBlazerPolicy, batch, optimizer, clip_eps=0.2, value_coef=0.5, ent_coef=0.01):
    device = policy.device
    all_states = []
    all_actions = []
    old_log_probs = []
    returns = []
    advantages = []

    for ep in batch:
        R, A = compute_gae(ep["rewards"], ep["values"])
        returns.extend(R)
        advantages.extend(A)
        all_actions.extend(ep["actions"])
        old_log_probs.extend(ep["log_probs"])
        all_states.extend(ep["states"])

    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    optimizer.zero_grad()
    policy_losses = []
    value_losses = []
    entropies = []

    # recompute log_probs and values under current policy
    for i, state in enumerate(all_states):
        # build features via policy's state_features pipeline
        from redteam_rl.state_features import build_state_features

        features = build_state_features(state, config=policy.cfg, include_embeddings=True)
        q_emb = torch.tensor(features.q_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        if features.history:
            h_mat = torch.tensor([t.h_embedding for t in features.history], dtype=torch.float32, device=device)
        else:
            h_mat = torch.zeros((1, q_emb.size(1)), dtype=torch.float32, device=device)

        # ensure nets built
        if policy._policy_net is None or policy._value_head is None:
            policy._build_nets(total_in=q_emb.size(1) + h_mat.size(1))

        attn_logits = torch.matmul(h_mat, q_emb.squeeze(0)) / (q_emb.size(1) ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=0)
        attended = torch.matmul(attn_weights.unsqueeze(0), h_mat).squeeze(0)
        policy_in = torch.cat([q_emb.squeeze(0), attended], dim=-1).unsqueeze(0)
        logits = policy._policy_net(policy_in).squeeze(0)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.tensor(all_actions[i], device=device)
        new_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = policy._value_head(policy_in).squeeze(0)

        ratio = torch.exp(new_log_prob - torch.tensor(old_log_probs[i], device=device))
        A = advantages[i]
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
        policy_loss = -torch.min(surr1, surr2)
        value_loss = (returns[i] - value) ** 2
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropies.append(entropy)

    loss = torch.stack(policy_losses).mean() + value_coef * torch.stack(value_losses).mean() - ent_coef * torch.stack(entropies).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    policy = TrailBlazerPolicy()
    dummy_state = EpisodeState(seed_prompt="dummy")
    policy.select_action(dummy_state)

    params = list(policy._policy_net.parameters()) + list(policy._value_head.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    mutator = TemplateMutator()
    victim = FakeVictim()
    reward = FakeReward()

    for epoch in range(args.epochs):
        batch = collect_episodes(policy, mutator, victim, reward, args.episodes_per_batch, args.max_turns)
        loss_value = ppo_update(policy, batch, optimizer)
        print(f"epoch={epoch} loss={loss_value:.6f}")
        # checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt")
        policy.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
