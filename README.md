# CS-224r-final-project

## Red-teaming LLMs via Adaptive Environments

### Installation of Dependencies

```bash
conda env create -n active_attacks python=3.10 -y
conda activate active_attacks
pip install -r requirements.txt
```

### Warm-up SFT

Similar to prior work, we warm-up the attacker LLM with SFT using pre-collected dataset.

```bash
python main.py \
    --mode sft \
    --model_name $ATTACKER_NAME \
    --lr 3e-5 \
    --train_steps 100 \
    --grad_acc_steps 32 \
    --batch_size 1024 \
    --few_shot_file ./prompts/sft_dataset.json \
    --exp_name attacker-$ATTACKER_NAME/sft \
    --save_dir $SFT_SAVE_DIR
```

### GFlowNet + Active Attacks

Active attacks is a plug-and-play module that seamlessly integrates into existing RL objectives. In implementation, we can turn on/off active attacks by using argument.

```bash
python main.py \
    --mode redteam \
    --model_name $ATTACKER_NAME \
    --victim_model $VICTIM_NAME \
    --toxicity_fn $CLASSIFIER_NAME
    --lr 1e-4 \
    --train_steps 5000 \
    --grad_acc_steps 8 \
    --batch_size 16 \
    --seed 0 \
    --exp_name attacker-$ATTACKER_NAME-victim-$VICTIM_NAME-classifier-$CLASSIFIER_NAME/seed$seed \
    --log_dir $ATTACK_LOG_DIR \
    --save_dir $ATTACK_SAVE_DIR \
    --sft_ckpt $SFT_SAVE_DIR
    --lora
    // Active Attacks argument
    --active_attacks \
    --interval 1000
```

### MLE smoothing for attack LLM

Given collected prompt dataset, we can finally obtain MLE smoothed attacker LLM

```bash
python main.py \
    --mode mle \
    --model_name $ATTACKER_NAME \
    --lr 3e-5 \
    --train_steps 200 \
    --num_warmup_steps 0 \
    --grad_acc_steps 32 \
    --batch_size 1024 \
    --seed 0 \
    --exp_name attacker-$ATTACKER_NAME-victim-$VICTIM_NAME-classifier-$CLASSIFIER_NAME/seed$seed \
    --log_dir $MLE_LOG_DIR \
    --save_dir $MLE_SAVE_DIR \
    --attack_ckpt $ATTACK_SAVE_DIR
    // Active Attacks argument
    --active_attacks \
    --interval 1000
```

### Safety fine-tuned victim LLM

Given collected prompt dataset, we can finally safety fine-tune victim LLM.

```bash
python main.py \
    --mode safety \
    --model_name $VICTIM_NAME \
    --lr 3e-5 \
    --train_steps 200 \
    --num_warmup_steps 0 \
    --grad_acc_steps 32 \
    --batch_size 1024 \
    --seed 0 \
    --exp_name attacker-$ATTACKER_NAME-victim-$VICTIM_NAME-classifier-$CLASSIFIER_NAME/seed$seed \
    --log_dir $SAFETY_LOG_DIR \
    --save_dir $SAFETY_SAVE_DIR \
    --attack_ckpt $ATTACK_SAVE_DIR
    // Active Attacks argument
    --active_attacks \
    --interval 1000
```

## Running on Modal

This repository now includes `modal_setup.py`, a Modal application that can launch training on a GPU and persist both cache and saved artifacts.
The current Modal image is focused on SFT and attack collection. Red-team training uses `vllm` and should use a separate compatible image.

### Local setup

```bash
pip install modal
```

### Optional: set a Hugging Face token

If your model downloads require authentication, export your token before running Modal:

```bash
export HF_TOKEN=<your-token>
```

### Optional: enable Weights & Biases logging

SFT already logs `ce-loss/train` every step and `ce-loss/validation` every 10 steps. To send those logs to W&B from Modal, export your API key locally before running:

```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

### Run a training job on Modal

```bash
modal run modal_setup.py::train --cmd-args "--mode sft --model_name qwen2.5-1.5b --lr 3e-5 --train_steps 100 --grad_acc_steps 32 --batch_size 32 --lora --wandb_project active-attacks --exp_name attacker-qwen2.5-1.5b/sft-lora"
```

### Run attack collection on Modal

```bash
modal run modal_setup.py::collect_attacks --cmd-args "--attack_type ICL --attack_model Qwen/Qwen2.5-1.5B-Instruct --num_samples 1024 --batch_size 16"
```

### Important paths

- repository code: `/workspace`
- persistent outputs: `/output`
- Hugging Face cache: `/cache`

For chained runs, pass explicit `/output` paths for checkpoints, for example `--sft_ckpt /output/save/attacker-qwen2.5-1.5b/sft/latest`.
