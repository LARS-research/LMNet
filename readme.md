# Boosting Collective Intelligence through Densely Communicated Large Language Models

##  Scaling for General Intelligence
* To scale for general intelligence by pre-training a LMNet on large-sclae datasets, run "python qwen_llama/pretrain_qwen.py" after customizing the dataset and training_args.

##  Customizing with Limited Data
* To customize with limited data, run "python qwen_llama/mmlu_qwen.py" or "python qwen_llama/mmlu_llama.py".
###  PEFT on E2E dataset
* get public code "LoRA" (https://github.com/microsoft/LoRA), and prepare enviroment and datasets following the corresponding instructions;
* copy "gpt/lmn_beam.py", "gpt/lmn_ft.py", "gpt/model_net.py" to "LoRA/examples/NLG/src/", and run 
```
torchrun --nproc_per_node=1 --master_port=1334 src/lmn_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 0 \
    --max_epoch 15 \
    --save_interval 9999999 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/lmn121 \
    --random_seed 110
```
* To evaluate, run the following to beam search (step 2), then follow the same instruction in LoRA to decode (step 3) and evaluate (step 4).
```
torchrun --nproc_per_node=1 --master_port=12318 src/lmn_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/lmn121/model.last_step_number.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/lmn121 \
    --output_file predict.last_step_number.b10p09r4.jsonl
```



