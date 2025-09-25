
import torch
from transformers import  Trainer, TrainingArguments,EvalPrediction,DefaultDataCollator,LlamaTokenizer,AutoTokenizer
from datasets import load_dataset,concatenate_datasets
import os
import numpy as np
from safetensors.torch import load_file
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8'
# Load dataset
#DS_SKIP_CUDA_CHECK=1 deepspeed --num_gpus=4 src/mllama.py
dataset = load_dataset("cais/mmlu", "all")#['test', 'validation', 'dev', 'auxiliary_train'])


# Check if GPU is available

from transformers_llama.modeling_llama import LlamaForSequenceClassification,LMNetForSequenceClassification,LMNetForSequenceClassificationS
from transformers_llama.configuration_llama import LlamaConfig

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=4)
tmodel=LlamaForSequenceClassification.from_pretrained(model_name,num_labels=4)
model = LMNetForSequenceClassification(tmodel.config,layer_list=[1,2,1],pl=0)#.from_pretrained(model_name)
model.model=tmodel.model
#model.freeze_transformer()
'''model = LMNetForSequenceClassificationS(tmodel.config,layer_list=[2,1],pl=0)#.from_pretrained(model_name)
for m in model.ms:
    m.load_state_dict(tmodel.model.state_dict())
del tmodel'''
model.config.pad_token_id=tokenizer.pad_token_id


data_collator = DefaultDataCollator()
def preprocess_function(examples, max_length=512):
    inputs = []
    labels = []

    for question, choices, answer in zip(examples["question"], examples["choices"], examples["answer"]):
        input_text = f"Question: {question}\nOptions:\n"
        for idx, option in enumerate(choices):
            input_text += f"{idx}. {option}\n"
        
        label = answer
        
        inputs.append(input_text)
        labels.append(label)

    # 使用分词器编码输入文本
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = labels

    return tokenized_inputs

def preprocess_logits_for_metrics(logits,labels):
    predicted_answers = torch.argmax(logits, axis=-1) 
    accuracy = (predicted_answers == labels).float().mean()
    return accuracy.detach()


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    return {"accuracy": predictions.mean()}




# 对数据集进行预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./LMN_mmlu_llama/ls21_ft",
    evaluation_strategy="epoch",
    #eval_accumulation_steps=8,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=1e-6,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=1000,
    logging_strategy="steps",
    save_strategy="epoch",
    bf16=True,  
    #deepspeed=''
)

#td=tokenized_datasets['test'].shard(num_shards=200, index=0).select([0,1,2,3,4,5,6,7])#
ts=concatenate_datasets([tokenized_datasets['validation'],tokenized_datasets['dev'],tokenized_datasets['auxiliary_train']])
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ts,
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

print(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
#print(td[0])
#eval_results = trainer.evaluate()
#print(f"Evaluation results: {eval_results}")

print(training_args.output_dir)
trainer.train()
