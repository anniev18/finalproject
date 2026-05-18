"""Modal entrypoint for running a multi-turn red-team episode on GPU.

Run from the repo root:

    modal run modal_run_episode.py

For the first test, the default uses the template mutator and Prompt Guard.
Set `reward_backend="llama_guard"` to load Llama Guard, and set
`use_llm_mutator=True` to load the Qwen mutator too.
"""

import json
import os

import modal


app = modal.App("cs224r-redteam-rl")

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
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_episode_remote(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    config_path: str = "/root/configs/default.json",
    use_template_mutator: bool = True,
    reward_backend: str | None = None,
    aux_reward_backends: list[str] | None = None,
) -> dict:
    return _run_episode_impl(
        seed_prompt=seed_prompt,
        seed_prompt_file=seed_prompt_file,
        seed=seed,
        max_turns=max_turns,
        config_path=config_path,
        use_template_mutator=use_template_mutator,
        reward_backend=reward_backend,
        aux_reward_backends=aux_reward_backends,
    )


def _run_episode_impl(
    seed_prompt: str | None,
    seed_prompt_file: str,
    seed: int | None,
    max_turns: int,
    config_path: str,
    use_template_mutator: bool,
    reward_backend: str | None,
    aux_reward_backends: list[str] | None,
) -> dict:
    import sys

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    sys.path.insert(0, "/root")

    from redteam_rl.attacker import Attacker
    from redteam_rl.config import load_config
    from redteam_rl.env import EnvConfig, RedTeamEnv
    from redteam_rl.mutators import LLMMutator, TemplateMutator
    from redteam_rl.policy import RandomPolicy
    from redteam_rl.rewards import build_reward_model
    from redteam_rl.seed_prompts import sample_seed_prompt
    from redteam_rl.victims import EvolvingVictim, VLLMVictim

    cfg = load_config(config_path)
    selected_seed_prompt = seed_prompt or sample_seed_prompt(seed_prompt_file, seed=seed)
    policy = RandomPolicy()
    mutator = TemplateMutator() if use_template_mutator else LLMMutator(cfg.mutator_config())
    attacker = Attacker(policy=policy, mutator=mutator)
    victim = EvolvingVictim(VLLMVictim(cfg.victim_config()))
    selected_reward_backend = reward_backend or cfg.reward_backend()
    reward_model = build_reward_model(
        selected_reward_backend,
        prompt_guard_config=cfg.prompt_guard_config(),
        qwen_judge_config=cfg.qwen_judge_config(),
        llama_guard_config=cfg.llama_guard_config(),
    )
    auxiliary_reward_models = {
        backend: build_reward_model(
            backend,
            prompt_guard_config=cfg.prompt_guard_config(),
            qwen_judge_config=cfg.qwen_judge_config(),
            llama_guard_config=cfg.llama_guard_config(),
        )
        for backend in (aux_reward_backends or [])
    }

    env = RedTeamEnv(
        config=EnvConfig(max_turns=max_turns),
        attacker=attacker,
        victim=victim,
        reward_model=reward_model,
        auxiliary_reward_models=auxiliary_reward_models,
    )

    state = env.reset(selected_seed_prompt)
    done = False
    while not done:
        state, _, done, _ = env.step()

    return {
        "seed_prompt": state.seed_prompt,
        "turns": [
            {
                "action": turn.action.value if turn.action else None,
                "user_message": turn.user_message,
                "victim_response": turn.victim_response,
                "reward": turn.reward,
                "metadata": turn.metadata,
            }
            for turn in state.turns
        ],
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_episode_llama_guard_remote(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    config_path: str = "/root/configs/default.json",
    use_template_mutator: bool = True,
    aux_reward_backends: list[str] | None = None,
) -> dict:
    return _run_episode_impl(
        seed_prompt=seed_prompt,
        seed_prompt_file=seed_prompt_file,
        seed=seed,
        max_turns=max_turns,
        config_path=config_path,
        use_template_mutator=use_template_mutator,
        reward_backend="llama_guard",
        aux_reward_backends=aux_reward_backends,
    )


@app.local_entrypoint()
def main(
    seed_prompt: str | None = None,
    seed_prompt_file: str = "/root/data/seed_prompts.json",
    seed: int | None = None,
    max_turns: int = 2,
    use_template_mutator: bool = True,
    use_llm_mutator: bool = False,
    reward_backend: str | None = None,
    aux_reward_backend: str | None = None,
    big_gpu: bool = False,
) -> None:
    aux_reward_backends = [aux_reward_backend] if aux_reward_backend else None
    selected_use_template_mutator = use_template_mutator and not use_llm_mutator
    if big_gpu:
        result = run_episode_llama_guard_remote.remote(
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=seed,
            max_turns=max_turns,
            use_template_mutator=selected_use_template_mutator,
            aux_reward_backends=aux_reward_backends,
        )
    else:
        result = run_episode_remote.remote(
            seed_prompt=seed_prompt,
            seed_prompt_file=seed_prompt_file,
            seed=seed,
            max_turns=max_turns,
            use_template_mutator=selected_use_template_mutator,
            reward_backend=reward_backend,
            aux_reward_backends=aux_reward_backends,
        )
    print(json.dumps(result, indent=2))
