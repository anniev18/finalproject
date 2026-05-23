"""Modal entrypoint for training the TrailBlazer PPO policy.

This is a lightweight training path for milestone experiments. It uses the
existing dry-run scaffold (template mutator + fake victim/reward) so it does
not require Hugging Face access or a local GPU.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import modal


app = modal.App("cs224r-trailblazer-train")
volume = modal.Volume.from_name("cs224r-redteam-rl-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-gpu.txt")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("redteam_rl", remote_path="/root/redteam_rl")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_trailblazer_remote(
    save_dir: str = "/root/outputs/policies/trailblazer_ppo",
    epochs: int = 10,
    episodes_per_batch: int = 8,
    max_turns: int = 3,
    lr: float = 3e-4,
    wandb_project: str | None = None,
) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import sys

    sys.path.insert(0, "/root")

    import torch
    import torch.optim as optim

    from redteam_rl.actions import ACTIONS
    from redteam_rl.mutators import TemplateMutator
    from redteam_rl.policy import TrailBlazerPolicy
    from redteam_rl.types import DialogueTurn, EpisodeState

    class FakeVictim:
        def respond(self, prompt, state, victim_history_turns=0):
            del state, victim_history_turns
            return f"Fake victim response to: {prompt}"

    class FakeReward:
        def score(self, prompt, response, state):
            del response
            return float(len(prompt) + len(state.turns)) / 100.0

    def collect_episodes(policy: TrailBlazerPolicy, n: int) -> list[dict]:
        mutator = TemplateMutator()
        victim = FakeVictim()
        reward = FakeReward()
        batch = []
        for _ in range(n):
            state = EpisodeState(seed_prompt="Test prompt")
            done = False
            episode = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": []}
            while not done:
                decision = policy.select_action(state)
                episode["states"].append(copy.deepcopy(state))
                action_index = ACTIONS.index(decision.action)
                episode["actions"].append(action_index)
                episode["log_probs"].append(decision.log_prob if decision.log_prob is not None else 0.0)
                episode["values"].append(decision.value if decision.value is not None else 0.0)
                prompt = mutator.mutate(decision.action, state)
                victim_response = victim.respond(prompt, state)
                reward_value = reward.score(prompt, victim_response, state)
                state.turns.append(
                    DialogueTurn(
                        user_message=prompt,
                        victim_response=victim_response,
                        action=decision.action,
                        reward=reward_value,
                        metadata={
                            "policy_action_probs": decision.action_probs,
                            "policy_log_prob": decision.log_prob,
                            "policy_value": decision.value,
                            "policy_attention_weights": decision.attention_weights,
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

        for episode in batch:
            episode_returns, episode_advantages = compute_gae(episode["rewards"], episode["values"])
            returns.extend(episode_returns)
            advantages.extend(episode_advantages)
            all_actions.extend(episode["actions"])
            old_log_probs.extend(episode["log_probs"])
            all_states.extend(episode["states"])

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        entropies = []

        from redteam_rl.state_features import build_state_features

        for index, state in enumerate(all_states):
            features = build_state_features(state, config=policy.cfg, include_embeddings=True)
            q_embedding = torch.tensor(features.q_embedding, dtype=torch.float32, device=device).unsqueeze(0)
            if features.history:
                history_matrix = torch.tensor([turn.h_embedding for turn in features.history], dtype=torch.float32, device=device)
            else:
                history_matrix = torch.zeros((1, q_embedding.size(1)), dtype=torch.float32, device=device)

            if policy._policy_net is None or policy._value_head is None:
                policy._build_nets(total_in=q_embedding.size(1) + history_matrix.size(1))

            attn_logits = torch.matmul(history_matrix, q_embedding.squeeze(0)) / (q_embedding.size(1) ** 0.5)
            attn_weights = torch.nn.functional.softmax(attn_logits, dim=0)
            attended_history = torch.matmul(attn_weights.unsqueeze(0), history_matrix).squeeze(0)
            policy_input = torch.cat([q_embedding.squeeze(0), attended_history], dim=-1).unsqueeze(0)

            logits = policy._policy_net(policy_input).squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)

            action = torch.tensor(all_actions[index], device=device)
            new_log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            value = policy._value_head(policy_input).squeeze(0)

            ratio = torch.exp(new_log_prob - torch.tensor(old_log_probs[index], device=device))
            advantage = advantages_tensor[index]
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_losses.append(-torch.min(surrogate_1, surrogate_2))
            value_losses.append((returns_tensor[index] - value) ** 2)
            entropies.append(entropy)

        loss = (
            torch.stack(policy_losses).mean()
            + value_coef * torch.stack(value_losses).mean()
            - ent_coef * torch.stack(entropies).mean()
        )
        loss.backward()
        optimizer.step()
        return float(loss.item())

    os.makedirs(save_dir, exist_ok=True)
    policy = TrailBlazerPolicy()
    policy.select_action(EpisodeState(seed_prompt="dummy"))
    optimizer = optim.Adam(list(policy._policy_net.parameters()) + list(policy._value_head.parameters()), lr=lr)

    wandb_run = None
    if wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name="trailblazer_modal_train")
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    run_summary = {"epochs": [], "save_dir": save_dir}
    for epoch in range(epochs):
        batch = collect_episodes(policy, episodes_per_batch)
        loss_value = ppo_update(policy, batch, optimizer)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        policy.save_checkpoint(checkpoint_path)
        print(f"epoch={epoch} loss={loss_value:.6f} checkpoint={checkpoint_path}")
        if wandb_run is not None:
            try:
                wandb_run.log({"epoch": epoch, "loss": loss_value, "checkpoint_path": checkpoint_path}, step=epoch)
            except Exception:
                pass
        run_summary["epochs"].append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "checkpoint_path": checkpoint_path,
            }
        )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"wandb logging failed: {exc}")

    volume.commit()
    return run_summary


@app.local_entrypoint()
def main(
    save_dir: str = "outputs/policies/trailblazer_ppo",
    epochs: int = 10,
    episodes_per_batch: int = 8,
    max_turns: int = 3,
    lr: float = 3e-4,
    wandb_project: str | None = None,
) -> None:
    result = train_trailblazer_remote.remote(
        save_dir="/root/outputs/policies/trailblazer_ppo",
        epochs=epochs,
        episodes_per_batch=episodes_per_batch,
        max_turns=max_turns,
        lr=lr,
        wandb_project=wandb_project,
    )
    print(json.dumps(result, indent=2))
    print("\nsaved TrailBlazer checkpoints to Modal volume cs224r-redteam-rl-data")
    print("remote checkpoint path: /root/outputs/policies/trailblazer_ppo")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        local_summary_path = Path(save_dir) / "modal_trailblazer_train_summary.json"
        with local_summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved local summary to {local_summary_path}")
