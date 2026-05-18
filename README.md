# CS224R Final Project

This repository contains a new implementation of a multi-turn RL framework for
LLM red-teaming.

## Layout

- `redteam_rl/`: new project code for the proposed system.
- `active_attacks_reference/`: reference copy of the cloned Active Attacks code.

The new implementation should not depend on `active_attacks_reference` directly.
Use that directory only for implementation ideas such as victim generation,
toxicity scoring, replay buffers, and evolving-victim training rounds.

## Local Environment

Use a virtual environment for local work. This setup is for editing, imports,
and dry runs that do not load real model weights.

```bash
cd CS-224r-final-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Run a local smoke test:

```bash
python scripts/run_episode.py --dry-run
```

By default, runners sample seed prompts from `data/seed_prompts.json`. To use a
specific prompt:

```bash
python scripts/run_episode.py --dry-run --seed-prompt "Explain how safety policies work."
```

Do not install `vllm` directly on a laptop unless you know your machine has a
compatible GPU/runtime. Real model runs should use a CUDA GPU environment such
as Modal.

## GPU Environment

For a CUDA GPU machine or remote environment:

```bash
pip install -r requirements-gpu.txt
```

Then run:

```bash
python scripts/run_episode.py --max_turns 3
```

Real model runs require Hugging Face access to the configured models in
`configs/default.json`.

## Modal

Modal is the recommended path for running the real vLLM-backed episode without
installing GPU dependencies on your laptop.

Install the Modal client in your local environment:

```bash
pip install modal
modal setup
```

Create a Modal secret named `huggingface-secret` with an `HF_TOKEN` value that
has access to the configured Hugging Face models, especially Llama Guard.

Then run:

```bash
modal run modal_run_episode.py
```

By default this uses the template mutator and a small Qwen 0.5B judge reward
model, which is much lighter than Llama Guard while still judging the victim
response.

For the cheapest remote smoke test, use a fake reward:

```bash
modal run modal_run_episode.py --reward-backend=fake
```

To load Llama Guard for real reward scoring:

```bash
modal run modal_run_episode.py --reward-backend=llama_guard
```

Other reward backends:

```bash
modal run modal_run_episode.py --reward-backend=fake
modal run modal_run_episode.py --reward-backend=prompt_guard
modal run modal_run_episode.py --reward-backend=qwen_safety_judge
```

To log Prompt Guard as an auxiliary score while keeping the default Qwen judge
as the main reward:

```bash
modal run modal_run_episode.py --aux-reward-backend=prompt_guard
```

To run Llama Guard offline on saved trajectories:

```bash
python scripts/score_trajectories.py \
  --input outputs/episodes.jsonl \
  --output outputs/llama_guard_scored.jsonl \
  --reward-backend llama_guard
```

If Llama Guard does not fit on the default L40S setup, use the larger-GPU
entrypoint:

```bash
modal run modal_run_episode.py --big-gpu=true
```

To also load the Qwen mutator model:

```bash
modal run modal_run_episode.py --use-llm-mutator
```
