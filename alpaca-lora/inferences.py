import pandas as pd
import numpy as np
import random
import pickle
import logging
import json
from tqdm import tqdm
from utils.prompter import Prompter

# setup logging
logging.basicConfig(level=logging.DEBUG)
logging.debug('Setup log file')


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
logging.debug('set CUDA device')


df = pd.read_csv('../Final_test_file/Truncated_Test_final.csv')
logging.debug(f'prompts loaded are\n{df[:3]}')

df['Prompt'] = df['Prompt'].str.replace('\u2014', "")

# df = df[:10]

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
except:  # noqa: E722
    pass
logging.debug(f'loaded libraries on device {DEVICE}')


base_model = "decapoda-research/llama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(base_model)

prompt_template = ""

prompter = Prompter(prompt_template)

basic_text = """Consider yourself as my law advisor. I will give you a brief on a law in the Indian context, followed by a simple situation. Your task is to perform Statutory Reasoning. Statutory reasoning is the task of reasoning with facts and statutes, which are rules written in natural language by a legislature. Keep your steps in three stages: Understanding the relevant law, analyze the situation, determine applicability. Finally give a one-word yes or no answer. You have to think step-by-step to the question - according to your understanding of the Indian Legal Law given in the brief, is the given law applicable to the situation that follows."""

def create_prompt(instruction: str) -> str:
    return prompter.generate_prompt(basic_text, instruction)

# logging.info(f"creating prompt template check {create_prompt('what is the meaning of life')}")

logging.info(f"creating prompt template for df {create_prompt(df['Prompt'][1])}")


def generate_response(
                    prompt: str, 
                    model: PeftModel,
                    temperature=0,
                    # top_p=0.5,
                    # top_k=10,
                    # num_beams=3,
                    num_return_sequences=1,
                    max_new_tokens=10,
                    # do_sample=True
        ):

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)

    generation_config = GenerationConfig(
        temperature=temperature,
        # top_p=top_p,
        # top_k=top_k,
        # num_beams=num_beams, ### added on my own
        num_return_sequences=num_return_sequences,
        # do_sample=do_sample
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

model_for_inference = "llama_without_id_withExtraVal"

if model_for_inference == "llama_with_id_withExtraVal":
    # checkpoints = ["checkpoint-" + str(i) for i in [11, 33, 55, 110, 187]]
    checkpoints = ["checkpoint-" + str(i) for i in [44, 66, 88, 132, 165]]
elif model_for_inference == "llama_without_id_withExtraVal":
    # checkpoints = ["checkpoint-" + str(i) for i in [3, 24, 42, 59, 70]]
    checkpoints = ["checkpoint-" + str(i) for i in [10, 17, 31, 35, 52]]
else:
    checkpoints = ["vanilla"]

base_folder = "/home/gsk/debiasing/final_finetuning/"
# checkpoints = os.listdir(os.path.join(base_folder, model_for_inference))

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

    if chk != "vanilla":
        model = PeftModel.from_pretrained(model, 
            model_path, 
            torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    # model = torch.compile(model)


    result_file_path = os.path.join(base_folder,  "result_file_"+model_for_inference,chk + ".jsonl")


    results = []

    # promptList=prompt_list[:10]

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
        # print(results[-1])
        # write to file every 100th iteration
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

