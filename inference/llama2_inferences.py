import pandas as pd
import numpy as np
import random
import pickle
import logging
import json
from tqdm import tqdm
from alpaca.utils.prompter import Prompter

# setup logging
logging.basicConfig(level=logging.DEBUG)
logging.debug('Setup log file')


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
logging.debug('set CUDA device')


df = pd.read_csv('../datasets/test/bsr_withID.csv')

df['Prompt'] = df['Prompt'].str.replace('\u2014', "")
logging.debug(f'edited prompts loaded are\n{df[:3]}')

import torch
from peft import PeftModel
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except: 
    pass
logging.debug(f'loaded libraries on device {DEVICE}')


base_model = "../Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = tokenizer.eos_token_id

prompt_template = ""

prompter = Prompter(prompt_template)

basic_text = """Consider yourself as my law advisor. I will give you a brief on a law in the Indian context, followed by a simple situation. Your task is to perform Statutory Reasoning. Statutory reasoning is the task of reasoning with facts and statutes, which are rules written in natural language by a legislature. Keep your steps in three stages: Understanding the relevant law, analyze the situation, determine applicability. Finally give a one-word yes or no answer. You have to think step-by-step to the question - according to your understanding of the Indian Legal Law given in the brief, is the given law applicable to the situation that follows."""

def create_prompt(instruction: str) -> str:
    return prompter.generate_prompt(basic_text, instruction)

logging.info(f"creating prompt template for df {create_prompt(df['Prompt'][1])}")


def generate_response(
                    prompt: str, 
                    model: PeftModel,
                    temperature=0,
                    num_return_sequences=1,
                    max_new_tokens=10,
        ):

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)

    generation_config = GenerationConfig(
        temperature=temperature,
        num_return_sequences=num_return_sequences,
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = [tokenizer.decode(i) for i in response.sequences]
    response = [i.split("### Response:")[1].strip() for i in decoded_output]
    return ["\n".join(textwrap.wrap(i)) for i in response]

def ask_alpaca(prompt: str, model: PeftModel ) -> str:
    prompt = create_prompt(prompt)
    response = generate_response(prompt, model)
    formatted_response = format_response(response)
    return formatted_response

model_for_inference = "llama2_with_id_withExtraVal" # change this field to give which model inference you want to perform

if model_for_inference == "llama2_with_id_withExtraVal":
    checkpoints = ["checkpoint-" + str(i) for i in [11, 33, 44, 55, 66, 88, 110, 132, 165, 187]]
elif model_for_inference == "llama2_without_id_withExtraVal":
    checkpoints = ["checkpoint-" + str(i) for i in [3, 10, 17, 24, 31, 35, 42, 52, 59, 70]]
else:
    checkpoints = ["vanilla"]

base_folder = "./results/"

print(f"The checkpoints are {checkpoints}")

os.makedirs(os.path.join(base_folder,  "result_file_"+model_for_inference), exist_ok=True)

for chk in tqdm(checkpoints):
    logging.debug(f'Started checkpoint {chk}')
    model_path = os.path.join(base_folder, model_for_inference, chk)

    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if chk!= "vanilla":
        model = PeftModel.from_pretrained(model, 
            model_path, 
            torch_dtype=torch.float16)


    model = model.eval()

    result_file_path = os.path.join(base_folder,  "result_file_"+model_for_inference,chk + ".jsonl")

    results = []

    for (idx, prompt) in tqdm(enumerate(df['Prompt'])):
        try:
            res = ask_alpaca(prompt, model)
            results.append({'index':idx,
                            'PROMPT':prompt,
                            'RESULT': res}
                           )
        except:
            results.append({'index':idx,
                            'PROMPT':prompt,
                            'RESULT': []}
                           )
            print(f"Encountered exception in checkpoint={chk}, id={idx}")
        if (idx+1) % 1000 == 0:
            with open(result_file_path, 'w') as json_file:
                json.dump(results, json_file, indent = 4)
                json_file.write('\n')

            logging.debug(f'Added {len(results)} by the loop')

    with open(result_file_path, 'w') as json_file:
        json.dump(results, json_file, indent = 4)
        json_file.write('\n')

    logging.debug(f'Added {len(results)} totally')
    logging.debug(f'Checkpoint {chk} completed')

