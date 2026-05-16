import os
import shlex
import subprocess

import modal

app = modal.App("active-attacks")

REPO_PATH = "/workspace"
OUTPUT_PATH = "/output"
CACHE_PATH = "/cache"

output_volume = modal.Volume.from_name("active-attacks-output", create_if_missing=True)
cache_volume = modal.Volume.from_name("active-attacks-cache", create_if_missing=True)
hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ.get("HF_TOKEN"),
    "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN"),
    "HUGGINGFACE_HUB_TOKEN": os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
})

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "cmake",
        "gcc",
        "g++",
        "make",
        "ninja-build",
        "libglib2.0-0",
        "libsm6",
        "libxrender-dev",
        "curl",
    )
    .run_commands([
        "python -m pip install -U pip setuptools wheel",
        "python -m pip install torch==2.7.1",
        "python -m pip install modal transformers==4.53.3 accelerate==1.8.1 bitsandbytes==0.47.0 peft==0.15.2 sentencepiece==0.2.0 safetensors==0.5.3 datasets==3.6.0 huggingface-hub==0.34.4 wandb==0.20.1 jsonlines tqdm numpy pandas scikit-learn scipy regex pyyaml protobuf",
    ])
    .add_local_dir(
        ".",
        remote_path=REPO_PATH,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache"],
    )
)

def _to_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(OUTPUT_PATH, path)


def _normalize_paths(args: list[str]) -> list[str]:
    normalized = []
    skip_next = False
    output_keys = {"--save_dir", "--log_dir", "--attack_ckpt", "--sft_ckpt", "--output_dir"}
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in output_keys and i + 1 < len(args):
            normalized.append(arg)
            normalized.append(_to_output_path(args[i + 1]))
            skip_next = True
        else:
            normalized.append(arg)
    if "--save_dir" not in normalized:
        normalized += ["--save_dir", os.path.join(OUTPUT_PATH, "save")]
    if "--log_dir" not in normalized:
        normalized += ["--log_dir", os.path.join(OUTPUT_PATH, "logs")]
    return normalized


def _prepare_env():
    os.chdir(REPO_PATH)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_PATH, "transformers")
    os.environ["HF_HOME"] = os.path.join(CACHE_PATH, "huggingface")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_PATH, "datasets")
    os.environ["XDG_CACHE_HOME"] = CACHE_PATH

    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def _run_script(script: str, cmd_args: str):
    _prepare_env()
    args = _normalize_paths(shlex.split(cmd_args))
    command = ["python", script] + args
    print("Running command:", " ".join(command))
    try:
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} failed with exit code {result.returncode}. "
                "Scroll up in the Modal logs for the Python traceback above this message."
            )
    finally:
        output_volume.commit()
        cache_volume.commit()


@app.function(
    image=image,
    gpu="A10",
    secrets=[hf_secret],
    timeout=36000,
    volumes={OUTPUT_PATH: output_volume, CACHE_PATH: cache_volume},
)
def train(cmd_args: str = ""):
    _run_script("main.py", cmd_args)


@app.function(
    image=image,
    gpu="A10",
    secrets=[hf_secret],
    timeout=36000,
    volumes={OUTPUT_PATH: output_volume, CACHE_PATH: cache_volume},
)
def collect_attacks(cmd_args: str = ""):
    _run_script("collect_attacks.py", cmd_args)


if __name__ == "__main__":
    app.deploy()
