# Design Log

This file records current implementation choices for the new `redteam_rl`
project.

## Current Architecture

The environment is multi-turn. Each episode starts from a `seed_prompt`, then
repeats:

```text
policy chooses action -> mutator writes attacker message -> victim responds -> judge assigns reward
```

The attacker is split into:

- RL policy: chooses a discrete strategy/action.
- Mutator LLM: converts the chosen action and dialogue history into the next user message.

The policy is currently a placeholder `RandomPolicy`; PPO/GFlowNet training is
out of scope for this environment setup.

The victim receives only normal chat context: the last `victim_history_turns`
attacker/victim turns plus the current attacker message. It does not see
actions, rewards, judge labels, or attention weights. Those richer fields are
only for the attacker/mutator state.

Policy-facing state features are formatted separately from the mutator prompt.
`redteam_rl/state_features.py` defines `q_t` as the clean current prompt and
`h_i` as compact history records containing attacker/victim text, action ID,
reward, judge label, refusal/length features, recency, and optional embeddings.
The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

Attacker evolution is supported only as optional LoRA adapter management. The
mutator can load or switch attacker LoRA adapters manually, and
`EvolvingAttacker.should_update_after_episode()` exposes a periodic update hook
for an external trainer. We do not currently train the attacker or implement
GFlowNet/PPO updates in this code path.

`TemplateMutator` is a deterministic fallback that maps each action to a simple
hardcoded prompt template. It is used for cheap smoke tests when we do not want
to load the mutator LLM.

## Model Roles

Default config:

- Victim: `Qwen/Qwen2.5-1.5B-Instruct`
- Mutator: `Qwen/Qwen2.5-1.5B-Instruct`
- Main judge: `Qwen/Qwen2.5-1.5B-Instruct`
- Optional auxiliary judge: `meta-llama/Llama-Prompt-Guard-2-86M`
- Optional faithful judge: `meta-llama/Meta-Llama-Guard-2-8B`

## Rewards

Default reward backend is `qwen_safety_judge`.

The judge sees:

```text
attacker prompt + victim response
```

It returns:

```text
UNSAFE -> reward 1.0
SAFE   -> reward 0.0
```

Prompt Guard is supported as an auxiliary score on attacker prompts. Llama Guard
is supported for later offline or in-loop scoring, but is compute-heavy.

## Victim Evolution

The victim evolves by training a LoRA adapter on collected successful attacks.
After fine-tuning, `EvolvingVictim` switches future victim responses to use the
new adapter.

The current code provides the fine-tuning loop, but the orchestration schedule
is still external. Example future schedule:

```text
collect N episodes -> fine-tune victim adapter -> continue with evolved victim
```

## Execution

Local runs should use dry mode:

```bash
python scripts/run_episode.py --dry-run
```

Modal is used for real model runs:

```bash
modal run modal_run_episode.py
```

Default Modal run uses the Qwen victim, Qwen mutator, and Qwen safety judge. Use
`--use-template-mutator` only for cheaper deterministic smoke tests. Use
`--reward-backend=llama_guard` only when testing the heavier faithful judge.

## Milestone Experiment

For the project milestone, use Qwen for the real comparison runs and keep the
smoke-test path separate.

Use Qwen when you want a realistic end-to-end result:

- victim: `Qwen/Qwen2.5-1.5B-Instruct`
- mutator: `Qwen/Qwen2.5-1.5B-Instruct`
- judge: `Qwen/Qwen2.5-1.5B-Instruct`

Use `TemplateMutator` only for fast local wiring checks. It is not the final
milestone result because it does not reflect the Qwen-based attack pipeline.

Recommended milestone comparison:

1. Baseline smoke test: local dry run with `--dry-run`.
2. Real baseline: Modal episode run with the default Qwen victim/mutator/judge.
3. Learned policy: Modal episode run with `--policy-checkpoint=/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_9.pt`.

Example commands:

```bash
python scripts/run_episode.py --dry-run
modal run modal_run_episode.py --reward-backend=qwen_safety_judge
modal run modal_run_episode.py --policy-checkpoint=/root/outputs/policies/trailblazer_ppo/checkpoint_epoch_9.pt
```

For the milestone writeup, report:

- average episode reward
- judge label distribution (`SAFE` vs `UNSAFE`)
- one or two representative trajectories for each condition
- whether the PPO checkpoint changes the action mix or attack style relative to the baseline

Keep the comparison fixed on the same seed prompt set and the same `max_turns`
so the conditions are directly comparable.
