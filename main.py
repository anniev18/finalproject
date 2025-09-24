import os
from typing import Literal

from tap import Tap

from trainers.gfn_trainer import GFNTrainer
# from trainers.gfn_trainer_new import GFNTrainerDebug
from trainers.rl_trainer import RLTrainer
from trainers.mle_trainer import MLETrainer
from trainers.mle_parallel_trainer import MLEParallelTrainer
from trainers.safety_trainer import SafetyTrainer
from trainers.sft_trainer import SFTTrainer
from trainers.ppo_trainer import PPOTrainer
from utils import load_victim_config, seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Argument(Tap):
    baseline: bool = False
    mode: Literal["gfn", "gfn_debug", "sft", "mle", "mle_parallel", "safety", "reinforce", "ppo"] = "gfn"
    model_name: str = "gpt2"
    victim_model: str = "vicgalle/gpt2-alpaca"
    sft_ckpt: str = "save/gpt2-sft-position-final/latest"
    victim_ckpt: str = None
    save_dir: str = "./save"
    log_dir: str = "./logs"

    prompt_file: str = "prompts/attack_prompt.jsonl"
    few_shot_file: str = "prompts/sft_dataset.json"

    epochs: int = 1
    lr: float = 1e-4
    max_norm: float = 1.0
    weight_decay: float = 0.1

    num_warmup_steps: int = 100
    train_steps: int = 5000
    batch_size: int = 16
    grad_acc_steps: int = 8

    len_norm: bool = False
    max_len: int = 20
    min_len: int = 5

    victim_top_p: float = 0.92
    victim_max_len: int = 30
    victim_temp: float = 0.7
    use_4bit: bool = False

    load_buffer: bool = False
    buffer_size: int = 1000
    sim_tolerance: float = 0.4
    prioritization: Literal["c_reward", "reward", "uniform"] = "c_reward"
    buffer_ckpt: str = ""
    compare: str = "reward"
    metric: Literal["edit", "cosine"] = "edit"

    dtype: str = "float32"
    seed: int = 42

    eval_period: int = 500
    eval_batch_size: int = 1024
    # lora hparams
    lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # reward scaling
    beta: float = 0.1
    lm_sched_end: float = 1.0
    lm_sched_start: float = 1.0
    lm_sched_horizon: int = 2000

    # reward temperature
    reward_sched_start: float = 2.0
    reward_sched_end: float = 1.0
    reward_sched_horizon: int = 500

    # sampling temperature
    temp_low: float = 0.5
    temp_high: float = 2.0

    # victim model
    num_r_samples: int = 5
    do_sample: bool = True

    # wandb
    exp_name: str = "debug"
    wandb_project: str = "red-team-dq"
    
    # sync interval
    sync_interval: int = 10
    
    # gpu memory utilization
    model_gpu_memory_utilization: float = 0.2
    victim_gpu_memory_utilization: float = 0.3
    toxicity_gpu_memory_utilization: float = 0.25
    toxicity_version: int = 3
    toxicity_fn: str = "llama"
    
    # iteration
    iteration: int = 1
    
    # exploration
    exploration: bool = False
    exploration_type: Literal["rnd", "cosine"] = "rnd"
    exploration_lamb: float = 0.1
    rnd_train_steps: int = 200
    rnd_batch_size: int = 128
    rnd_lr: float = 1e-4
    
    # reweighting
    reweighting: bool = False
    
    # ppo
    clip_ratio: float = 0.2
    

if __name__ == "__main__":
    args = Argument(explicit_bool=True).parse_args()
    # load_victim_config(args)
    seed(args.seed)
    if args.mode == "gfn":
        load_victim_config(args)
        trainer = GFNTrainer(args)
    elif args.mode == "gfn_debug":
        load_victim_config(args)
        trainer = GFNTrainerDebug(args)
    elif args.mode == "reinforce":
        load_victim_config(args)
        trainer = RLTrainer(args)
    elif args.mode == "ppo":
        load_victim_config(args)
        trainer = PPOTrainer(args)
    elif args.mode == "mle":
        trainer = MLETrainer(args)
    elif args.mode == "mle_parallel":
        trainer = MLEParallelTrainer(args)
    elif args.mode == "safety":
        load_victim_config(args)
        trainer = SafetyTrainer(args)
    else:
        trainer = SFTTrainer(args)
    trainer.train()
