
import torch
from transformers import  Trainer, TrainingArguments,EvalPrediction,DefaultDataCollator,LlamaTokenizer,AutoTokenizer
from datasets import load_dataset,concatenate_datasets
import os
import numpy as np
from safetensors.torch import load_file
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
# Load dataset
#DS_SKIP_CUDA_CHECK=1 deepspeed --master_port=12327 --include localhost:0,1,2,3 src/mqw.py
dataset = load_dataset("cais/mmlu", "all")#['test', 'validation', 'dev', 'auxiliary_train'])

dataset = concatenate_datasets([dataset['validation'],dataset['dev'],dataset['auxiliary_train']])

testset=load_dataset("cais/mmlu", "all")['test']
# Check if GPU is available
from transformers_qwen2.modeling_qwen2 import Qwen2ForSequenceClassification,LMNetForSequenceClassification,LMNetForSequenceClassificationS
from transformers_qwen2.configuration_qwen2 import Qwen2Config

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#model = Qwen2ForSequenceClassification.from_pretrained(model_name, num_labels=4)
tmodel=Qwen2ForSequenceClassification.from_pretrained(model_name,num_labels=4)
model = LMNetForSequenceClassification(tmodel.config,layer_list=[1,2,1],pl=0)#.from_pretrained(model_name)
model.model=tmodel.model
#model.freeze_transformer()
'''model = LMNetForSequenceClassificationS(tmodel.config,layer_list=[1,2,1],pl=0)#.from_pretrained(model_name)
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



tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets_test = testset.map(preprocess_function, batched=True)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./LMN_mmlu_qwen/0.5B_l121_ft",
    evaluation_strategy="epoch",
    #eval_steps=100,
    #eval_accumulation_steps=1,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=1e-6,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=1000,
    logging_strategy="steps",
    save_strategy="epoch",
    #save_steps=20000,
    bf16=True,  
    #deepspeed=''
)

#model=LMNetForSequenceClassification.from_pretrained()
#model.melt_transformer()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,#['auxiliary_train'],
    eval_dataset=tokenized_datasets_test,
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

