import os

import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from utils import InfIterator, get_decay_parameter_names
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_model

class SafetyTrainer(object):
    def __init__(self, args) -> None:
        self.args = args

        wandb.init(reinit=True, config=args.as_dict(),
                   project=args.wandb_project, name=args.exp_name)

        # self.device = torch.cuda.current_device()
        # accelerator = Accelerator(mixed_precision="bf16")  # uses bfloat16 if available
        # self.device = accelerator.device

        # if args.iteration == 1:
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         args.model_name,
        #         torch_dtype=torch.bfloat16,
        #         device_map=self.device)
        # else:
        #     # we need to load the model from the previous iteration with lora adapter
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            # device_map=self.device)
            device_map="auto")
        # self.model = PeftModel.from_pretrained(self.model, args.sft_ckpt)
        # self.model = self.model.merge_and_unload()
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()


        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        decay_parameters = get_decay_parameter_names(self.model)
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],

                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, args.num_warmup_steps, args.train_steps)
        
        self.dataloader = get_dataloader("safety-tuning", self.tokenizer, 
                                        prompt_file=args.prompt_file, 
                                        batch_size=args.batch_size,
                                        reweighting=args.reweighting)
        self.train_iter = InfIterator(self.dataloader)
        
    
    def get_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Merge LoRA weights with base model and save the full model
        # merged_model = self.model.merge_and_unload()
        # make sure weights are tied (safe, even if already tied)
        # if hasattr(merged_model, "tie_weights"):
        #     merged_model.tie_weights()
        # merged_model.save_pretrained(output_dir)
        # save_model(merged_model, os.path.join(output_dir, "model.safetensors"))
        # merged_model.config.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def train(self):
        t = tqdm(range(1, self.args.train_steps+1), desc="training", dynamic_ncols=True, leave=False)
        self.model.train()
        # global_step = 1
        # for epoch in trange(self.args.epochs):
        #     for batch in tqdm(self.dataloader, dynamic_ncols=True):
        for global_step in t:
            batch = next(self.train_iter)
            batch_loss = []
            chunks = {k:torch.chunk(v, self.args.grad_acc_steps, dim=0) for k,v in batch.items()}
            num_chunks = len(chunks["input_ids"])
            self.model.zero_grad()
            self.model.train()
            for i in tqdm(range(num_chunks), desc="gradient step", dynamic_ncols=True, leave=False):
                mini_batch = {k:v[i].to(self.model.device) for k,v in chunks.items()}    
                
                # need expllicit position_ids because of left-padding
                loss = self.model(**mini_batch).loss
                loss = loss / self.args.grad_acc_steps
                
                loss.backward()
                batch_loss.append(loss.item())
            
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_norm)

            self.optimizer.step()
            self.scheduler.step()

            # logging
            wandb.log({"ce-loss/train": sum(batch_loss)}, step=global_step)

            t.set_description(
                f"Step {global_step}: {sum(batch_loss): .4f}")

            # global_step += 1
        
        output_dir = os.path.join(self.args.save_dir, self.args.exp_name, "latest")
        self.save(output_dir)
        wandb.finish()


