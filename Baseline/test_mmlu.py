import torch
import os
import time
import json
from typing import Optional
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
five_shot_prompt = """What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

"""
def get_prompt(question,options,answers):
    prompt = ""
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += str(answers[i])
        prompt += '\n'
    prompt += 'Answer:'
    return prompt


model_name = "/share/datasets/public_models/Llama-2-13b-chat-hf" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto")
test_directory = "/home/xujiaming/xujiaming/train_machine/PJY/ASPLOS-arranged/benchmark/mmlu/data/test/"
test_files = os.listdir(test_directory)
print(test_files)
model.eval()
total_time = 0
output_ids_tot = 0
acc = 0
total = 0
correct = 0
for file in test_files:
    print(file)
    file_path = test_directory + file
    mmlu_dataset = pd.read_csv(file_path, header=None)
    task_name = file[:-9]
    for idx, row in mmlu_dataset.iterrows():
        question = row[0]
        options = ["A", "B", "C", "D"]
        answers = [row[1], row[2], row[3], row[4]]
        correct_answer = row[5].strip()
        prompt = get_prompt(question,options,answers)
        
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        seqlen = len(inputs[0])
        st = time.time()
        outputs = model.generate(inputs, max_new_tokens=256)
        ed = time.time()
        output_ids_tot += len(outputs[0]) - seqlen
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        total_time += ed-st
        # answer_start_index = len(prompt)
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
# accuracy = correct / total
# print(f"模型在mmlu上的准确率为: {accuracy:.2%}")