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

`TemplateMutator` is a deterministic fallback that maps each action to a simple
hardcoded prompt template. It is used for cheap smoke tests when we do not want
to load the mutator LLM.

## Model Roles

Default config:

- Victim: `Qwen/Qwen2.5-1.5B-Instruct`
- Mutator: `Qwen/Qwen2.5-1.5B-Instruct`
- Main judge: `Qwen/Qwen2.5-0.5B-Instruct`
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

Default Modal run uses Qwen victim, template mutator, and Qwen safety judge. Use
`--reward-backend=llama_guard` only when testing the heavier faithful judge.
Use `--use-llm-mutator` to replace the template mutator with the Qwen mutator.
