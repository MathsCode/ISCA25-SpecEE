import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
file_path = "/home/xujiaming/xujiaming/train_machine/PJY/ASPLOS-arranged/benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
dataset = pq.read_table(file_path).to_pandas()

model_name = "/share/datasets/public_models/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map='auto')
five_shot_prompt = """The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
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
    prompt = ""
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += answers[i]
        prompt += '\n'
    prompt += 'Answer: '
    return prompt


model.eval()
total_time = 0
output_ids_tot = 0
total = 0
correct = 0
for _, row in dataset.iterrows():
    question = row['question']
    choices = row['choices']
    options = choices['label']
    answers = choices['text']
    correct_answer = row['answerKey'].strip()
    prompt = get_prompt(question,options,answers)
    st = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    seqlen = len(inputs[0])
    outputs = model.generate(inputs, max_new_tokens=256,output_hidden_states=True)
    output_ids_tot += len(outputs[0]) - seqlen
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ed = time.time()
    total_time += ed-st
    # answer_start_index = len(prompt)
    # print(generated_text[answer_start_index:])
    # # breakpoint()
    # try:
    #     predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
    # except:
    #     predicted_answer = "N/A"
    # print(predicted_answer)
    # print(predicted_answer == correct_answer)
    # if predicted_answer == correct_answer:
    #     correct += 1
    # total += 1    
print('sum Time:', total_time)
print('average time ',total_time/output_ids_tot)
print(output_ids_tot/total_time)
# accuracy = correct / total
# print(f"模型在commomsense_qa上的准确率为: {accuracy:.2%}")
