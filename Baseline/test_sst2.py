import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
file_path = "/home/xujiaming/xujiaming/train_machine/PJY/benchmark/sst2/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
dataset = pq.read_table(file_path).to_pandas()
model_name = "/share/datasets/public_models/Llama-2-70b-chat-hf"  # 替换为您想使用的LLaMA 2模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")
five_shot_prompt = """Here are some sentences.You need to identify the sentiment of these sentences.If the sentiment of a sentence is positive,you need to output 1,otherwise 0.
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
    prompt += 'Sentiment:'
    return prompt
  
model.eval()
total_time = 0
output_ids_tot = 0
total = 0
correct = 0
total = 0
for _, row in dataset.iterrows():
    sentence = row['sentence']
    label = str(row['label']).strip()
    prompt = get_prompt(sentence)
    st = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    seqlen = len(inputs[0])
    outputs = model.generate(inputs, max_new_tokens=2)
    output_ids_tot += len(outputs[0]) - seqlen
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ed = time.time()
    total_time += ed-st
    answer_start_index = len(prompt)
    try:
        predicted_answer = generated_text[answer_start_index:].strip()[0]
    except:
        predicted_answer = "N/A"
    if predicted_answer == label:   
        print("True")
        correct += 1 
    else:
        print("False")
    total += 1 
print('sum Time:', total_time)
print('average time ',total_time/output_ids_tot)
print("准确率: ",correct/total)