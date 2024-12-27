import sys
import os
from AutoEE.Pure.EE_model import EEModel
# from EE_model import EEModel
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ['TORCH_USE_CUDA_DSA'] = "1"
import torch
from fastchat.model import get_conversation_template

import time
from tqdm import trange
import json
from AutoEE.Pure.model_llama_ee import MLP
from typing import Optional
import pandas as pd
import pyarrow.parquet as pq

five_shot_prompt = """Here are some questions.You should give your answers.Your answer must be among A,B,C,D and E.
The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
A. ignore
B. enforce
C. authoritarian
D. yell at
E. avoid
Answer: A

Sammy wanted to go to where the people were.  Where might he go?
A. race track
B. populated areas
C. the desert
D. apartment
E. roadblock
Answer: B

To locate a choker not located in a jewelry box or boutique where would you go?
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Answer: A

Google Maps and other highway and street GPS services have replaced what?
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Answer: D

The fox walked from the city into the forest, what was it looking for?
A. pretty flowers.
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer: C

"""
def get_prompt(question,options,answers):
    prompt = five_shot_prompt
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += answers[i]
        prompt += '\n'
    # prompt += 'Answer: '
    return prompt


file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
dataset = pq.read_table(file_path).to_pandas()
model = EEModel.from_pretrained(
    base_model_path='/share/datasets/public_models/Llama-2-7b-chat-hf',
    ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # is_offload = False,
    # skip_model = "/home/xujiaming/xujiaming/research/ASPLOS-24/skip_layer/model.txt",
)
model.eval()
correct = 0
total = 0
exit_layer_id_list=[]
for _, row in dataset.iterrows():
    question = row['question']
    choices = row['choices']
    options = choices['label']
    answers = choices['text']
    correct_answer = row['answerKey'].strip()
    prompt = get_prompt(question,options,answers)
    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model(input_ids,max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    answer_start_index = len(prompt+"Answer:")     
    try:
        predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
    except:
        predicted_answer = "N/A"
    # print(predicted_answer)
    # print(predicted_answer == correct_answer)
    if predicted_answer == correct_answer:
        correct += 1
    total += 1
accuracy = correct / total
print(f"模型在commomsense_qa上的准确率为: {accuracy:.2%}")
print(sum(exit_layer_id_list)/len(exit_layer_id_list))