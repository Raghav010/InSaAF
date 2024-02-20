import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys
from typing import List

import fire
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
import json

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

import sys
sys.path.append('../')
from alpaca.utils.prompter import Prompter
from alpaca.utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "../Llama-2-7b-hf", 
    output_dir: str = "../models/llama2_without_id_withExtraVal", 
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 2,
    num_epochs: float = 30,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    quantized=False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training llama2 with identity model with params:\n"
            f"base_model: {base_model}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            # f"max_steps: {max_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=quantized,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    class legalDataset(Dataset):
        def __init__(self,prompts,tokenizer,prompter,isdataset=False,cutoffDB=None):
            self.prompts=prompts
            self.tokenizer=tokenizer
            self.prompter=prompter
            self.isdataset=isdataset
            self.cutoffDB=cutoffDB

        def __len__(self):
            if self.cutoffDB==None:
                return len(self.prompts)
            else:
                return self.cutoffDB
        
        def tokenize(self,prompt, add_eos_token=True):
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result
        
        def generate_and_tokenize_prompt(self,data_point):

            if self.isdataset:
                full_prompt=data_point['sentence']
            else:
                full_prompt = self.prompter.generate_prompt(
                    data_point["instruction"],
                    data_point["input"],
                    data_point["output"],
                )
            tokenized_full_prompt = self.tokenize(full_prompt)
            if not train_on_inputs:
                user_prompt = self.prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
                tokenized_user_prompt = self.tokenize(
                    user_prompt, add_eos_token=add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            return tokenized_full_prompt

        def __getitem__(self,idx):
            if self.isdataset:
                return self.generate_and_tokenize_prompt(self.prompts.__getitem__(idx))
            else:
                return self.generate_and_tokenize_prompt(self.prompts[idx])
            


    if quantized:   
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()  

    th=open('../datasets/finetune/bsr_withoutID/train_data.jsonl','r')  
    vh=open('../datasets/finetune/bsr_withoutID/val_data.jsonl','r')    

    train_data=legalDataset(json.load(th),tokenizer,prompter)
    val_data=legalDataset(json.load(vh),tokenizer,prompter)
    
    th.close()
    vh.close()

    # adding normal language modeling data
    languageModel_dataset = load_dataset("ptb_text_only",split='validation')
    lm_data=legalDataset(languageModel_dataset.shuffle(seed=7),tokenizer,prompter,True,400)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset={'validation':val_data,'baseline':lm_data},
        args=transformers.TrainingArguments(
            auto_find_batch_size=True,
            gradient_accumulation_steps=32,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_strategy="epoch",
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()

    os.mkdir(output_dir+'/last_epoch_model/')
    trainer.model.save_pretrained(output_dir+'/last_epoch_model/')

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)



