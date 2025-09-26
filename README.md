# Red-teaming LLMs via Adaptive Environments

## Installation of Dependencies
```bash
conda env create -n active_attacks python=3.10 -y
conda activate active_attacks
pip install -r requirements.txt
```

## Warm-up SFT
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

## GFlowNet + Active Attacks
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

## MLE smoothing for attack LLM
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

## Safety fine-tuned victim LLM
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


