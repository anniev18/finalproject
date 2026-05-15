import argparse

from trainers.sft_trainer import SFTTrainer
from trainers.gfn_trainer import GFNTrainer
from trainers.safety_trainer import SafetyTrainer
from trainers.mle_trainer import MLETrainer


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="sft", choices=["sft", "redteam", "mle", "safety"])
    args.add_argument("--model_name", type=str, default="qwen2.5-1.5b")
    args.add_argument("--victim_model", type=str, default="qwen2.5-1.5b-instruct")
    args.add_argument("--classifier_model", type=str, default="llama")
    args.add_argument("--sft_ckpt", type=str, default="save/qwen2.5-1.5b-sft/latest")
    args.add_argument("--lr", type=float, default=3e-5)
    args.add_argument("--max_norm", type=float, default=1.0)
    args.add_argument("--weight_decay", type=float, default=0.1)
    
    args.add_argument("--num_warmup_steps", type=int, default=200)
    args.add_argument("--train_steps", type=int, default=200)
    args.add_argument("--grad_acc_steps", type=int, default=32)
    args.add_argument("--batch_size", type=int, default=1024)
    
    args.add_argument("--min_len", type=int, default=5)
    args.add_argument("--max_len", type=int, default=20)
    
    args.add_argument("--victim_top_p", type=float, default=0.92)
    args.add_argument("--victim_max_len", type=int, default=30)
    args.add_argument("--victim_temp", type=float, default=0.7)
    args.add_argument("--use_4bit", action="store_true")
    
    args.add_argument("--buffer_size", type=int, default=1000)
    args.add_argument("--sim_tolerance", type=float, default=0.4)
    args.add_argument("--prioritization", type=str, default="c_reward")
    args.add_argument("--compare", type=str, default="reward")
    args.add_argument("--metric", type=str, default="edit")
    
    args.add_argument("--dtype", type=str, default="float32")
    args.add_argument("--seed", type=int, default=42)
    
    args.add_argument("--lora", action="store_true")
    args.add_argument("--lora_r", type=int, default=32)
    args.add_argument("--lora_alpha", type=int, default=16)
    args.add_argument("--lora_dropout", type=float, default=0.0)
    
    args.add_argument("--beta", type=float, default=0.1)
    args.add_argument("--lm_sched_end", type=float, default=1.0)
    args.add_argument("--lm_sched_start", type=float, default=1.0)
    args.add_argument("--lm_sched_horizon", type=int, default=2000)
    
    args.add_argument("--reward_sched_start", type=float, default=2.0)
    args.add_argument("--reward_sched_end", type=float, default=1.0)
    args.add_argument("--reward_sched_horizon", type=int, default=500)
    
    args.add_argument("--temp_low", type=float, default=0.5)
    args.add_argument("--temp_high", type=float, default=2.0)
    
    args.add_argument("--num_r_samples", type=int, default=5)
    args.add_argument("--do_sample", action="store_true")
    
    args.add_argument("--exp_name", type=str, default="active-attacks")
    args.add_argument("--wandb_project", type=str, default="active-attacks")
    
    args.add_argument("--model_gpu_memory_utilization", type=float, default=0.2)
    args.add_argument("--victim_gpu_memory_utilization", type=float, default=0.3)
    args.add_argument("--toxicity_gpu_memory_utilization", type=float, default=0.25)
    
    args.add_argument("--prompt_file", type=str, default="./prompts/attack_prompt.jsonl")
    args.add_argument("--few_shot_file", type=str, default="./prompts/sft_dataset.json")
    args.add_argument("--attack_ckpt", type=str, default="")
    args.add_argument("--log_dir", type=str, default="logs")
    args.add_argument("--save_dir", type=str, default="save")
    
    
    
    ####################################Active Attacks##############################
    args.add_argument("--active_attacks", action="store_true")
    args.add_argument("--interval", type=int, default=1000)
    ####################################Active Attacks##############################
    args = args.parse_args()
    
    # convert model name to huggingface format
    model_name = args.model_name
    if model_name == "qwen2.5-1.5b":
        model_name_hf = "Qwen/Qwen2.5-1.5B"
    elif model_name == "llama3.2-3b":
        model_name_hf = "meta-llama/Llama-3.2-3B"
    else:
        raise ValueError(f"Model name {model_name} not supported")
    args.model_name_hf = model_name_hf
    
    # convert victim model name to huggingface format
    victim_model = args.victim_model
    if victim_model == "qwen2.5-1.5b-instruct":
        victim_model_hf = "Qwen/Qwen2.5-1.5B-Instruct"
    elif victim_model == "llama3.2-3b-instruct":
        victim_model_hf = "meta-llama/Llama-3.2-3B-Instruct"
    elif victim_model == "gemma3-4b-it":
        victim_model_hf = "google/gemma-3-4b-it"
    elif victim_model == "mistral-7b-instruct":
        victim_model_hf = "mistralai/Mistral-7B-Instruct"
    else:
        raise ValueError(f"Victim model name {victim_model} not supported")
    args.victim_model_hf = victim_model_hf
    
    if args.mode == "sft":
        trainer = SFTTrainer(args)
        trainer.train()
    elif args.mode == "redteam":
        if args.active_attacks:
            num_rounds = args.train_steps // args.interval
            for round in range(num_rounds):
                args.round = round + 1
                trainer = GFNTrainer(args)
                trainer.train()
                
                if round < num_rounds - 1:
                    safety_trainer = SafetyTrainer(args)
                    safety_trainer.train()
        else:
            trainer = GFNTrainer(args)
            trainer.train()
    elif args.mode == "mle":
        trainer = MLETrainer(args)
        trainer.train()
    elif args.mode == "safety":
        trainer = SafetyTrainer(args)
        trainer.train()
    else:
        raise ValueError(f"Mode {args.mode} not supported")
