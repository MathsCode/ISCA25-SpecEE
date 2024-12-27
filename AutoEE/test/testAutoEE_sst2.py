import sys
import os
# 将路径添加到搜索路径中
path_to_add = '.'
absolute_path = os.path.abspath(path_to_add)
sys.path.append(absolute_path)
from AutoEE.Pure.EE_model import EEModel
import torch
from fastchat.model import get_conversation_template
import os
import time
from tqdm import trange
import json
from AutoEE.Pure.model_llama_ee import MLP
from typing import Optional
import pandas as pd
import pyarrow.parquet as pq
five_shot_prompt = """Here are some sentences.You need to answer the sentiment of these sentences.If the sentiment of a sentence is positive,you need to output 1,otherwise 0.
Sentence: hide new secretions from the parental units
Sentiment: 0

Sentence: contains no wit , only labored gags
Sentiment: 0

Sentence: that loves its characters and communicates something rather beautiful about human nature 
Sentiment: 1

Sentence: remains utterly satisfied to remain the same throughout 
Sentiment: 0

Sentence: demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop . 
Sentiment: 1

"""
def get_prompt(sentence):
    prompt = five_shot_prompt
    prompt += "Sentence: "
    prompt += sentence
    prompt += '\n'
    prompt += 'Sentiment'
    return prompt
file_path = "/home/xujiaming/xujiaming/train_machine/PJY/benchmark/sst2/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
dataset = pq.read_table(file_path).to_pandas()
model = EEModel.from_pretrained(
    base_model_path='/home/xujiaming/xujiaming/train_machine/dataset/Llama-2-7B-Chat-AWQ',
    ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # is_offload = False,
    # skip_model = "/home/xujiaming/xujiaming/research/ASPLOS-24/skip_layer/model.txt",
)
exit_layer_id_list=[]
model.eval()
total_time = 0
output_ids_tot = 0
total = 0
correct = 0
total = 0
for _, row in dataset.iterrows():
    # print("正常工作")
    sentence = row['sentence']
    label = str(row['label']).strip()
    prompt = get_prompt(sentence)
    st = time.time()
    inputs = model.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = torch.as_tensor(inputs).cuda()
    seqlen = len(inputs[0])
    outputs = model(input_ids, max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
    output_ids_tot += len(outputs[0]) - seqlen
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    ed = time.time()
    total_time += ed-st
    answer_start_index = len(prompt+":")
    try:
        predicted_answer = generated_text[answer_start_index:].strip()[0]
    except:
        predicted_answer = "N/A"
    # print(predicted_answer)
    if predicted_answer == label:   
        correct += 1 
    total += 1 
print("准确率: ",correct/total)