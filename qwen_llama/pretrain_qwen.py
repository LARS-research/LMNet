
import torch
from transformers import  Trainer, TrainingArguments,EvalPrediction,DefaultDataCollator,AutoTokenizer,DataCollatorForLanguageModeling,Qwen2Tokenizer,DataCollatorForSeq2Seq
from datasets import load_dataset
import os
import numpy as np
from safetensors.torch import load_file
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# Load dataset
#DS_SKIP_CUDA_CHECK=1 deepspeed --master_port=2327 --include localhost:0,1 src/qw_pt.py
dataset = None

from transformers_qwen2.modeling_qwen2 import Qwen2ForCausalLM,LMNetForCausalLM,LMNetForCausalLMS
from transformers_qwen2.configuration_qwen2 import Qwen2Config
import re
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#model = Qwen2ForCausalLM.from_pretrained(model_name)
tmodel=Qwen2ForCausalLM.from_pretrained(model_name)
model = LMNetForCausalLM(tmodel.config,layer_list=[1,4,4,4,1],pl=0)#.from_pretrained(model_name)
model.model=tmodel.model
model.lm_head=tmodel.lm_head
#model.freeze_transformer()
'''model = LMNetForSequenceClassificationS(tmodel.config,layer_list=[1,2,1],pl=0)#.from_pretrained(model_name)
for m in model.ms:
    m.load_state_dict(tmodel.model.state_dict())
del tmodel'''
model.config.pad_token_id=tokenizer.pad_token_id

data_collator=DefaultDataCollator()
'''data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)'''
#data_collator = DataCollatorForSeq2Seq(tokenizer,model)

from transformers import Trainer, TrainingArguments

training_args = None

#tmodel=LMNetForCausalLM.from_pretrained('1')
#model=LMNetForCausalLM(tmodel.config,layer_list=[1,2,1],pl=0)
#model.assign_model(tmodel)

#model=LMNetForCausalLM.from_pretrained('')
#model.freeze_transformer()
#model.melt_transformer()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
#trainer.evaluate()
print(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
#for i in range(512):
#    print(td[0]["input_ids"][i],td[0]["labels"][i])
#eval_results = trainer.evaluate()
#print(f"Evaluation results: {eval_results}")

print(training_args.output_dir)
trainer.train()

