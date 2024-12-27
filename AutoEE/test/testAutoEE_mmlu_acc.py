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
five_shot_prompt = """Here are some questions.You should give your answers.Your answer must be among A,B,C and D.
What is the embryological origin of the hyoid bone?
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
    prompt = five_shot_prompt
    prompt += question
    prompt += '\n'
    for i in range(len(options)):
        prompt += options[i]
        prompt += '. '
        prompt += str(answers[i])
        prompt += '\n'
    # prompt += 'Answer:'
    return prompt


test_directory = "/home/xujiaming/xujiaming/train_machine/PJY/benchmark/mmlu/data/test/"
test_files = os.listdir(test_directory)
print(test_files)
model = EEModel.from_pretrained(
    base_model_path='/home/xujiaming/xujiaming/train_machine/dataset/Llama-2-7B-Chat-AWQ',
    ea_model_path="/home/xujiaming/xujiaming/datasets/LLM_models/EAGLE-llama2-chat-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # is_offload = False,
    # skip_model = "/home/xujiaming/xujiaming/research/ASPLOS-24/skip_layer/model.txt",
)

all_total = 0
all_correct = 0
with open("1.txt",'w') as result_file:
    for file in test_files:
        file_path = test_directory + file
        mmlu_dataset = pd.read_csv(file_path, header=None)
        task_name = file[:-9]
        model.eval()
        correct = 0
        total = 0
        exit_layer_id_list=[]
        for idx, row in mmlu_dataset.iterrows():
            # 提取问题和选项
            question = row[0]
            options = ["A", "B", "C", "D"]
            answers = [row[1], row[2], row[3], row[4]]
            correct_answer = row[5].strip()

            # 构建输入文本，将问题和选项组合在一起
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
            # print(predicted_answer == correct_answer)
            # print(predicted_answer)
            if predicted_answer == correct_answer:
                correct += 1
                all_correct += 1
            if predicted_answer in ["A", "B", "C", "D"]:
                total += 1
                all_total += 1
        accuracy = correct / total
        print("模型在" + task_name + f"上的准确率为: {accuracy:.2%}")
        result_file.write(task_name)
        result_file.write('    ')
        result_file.write(str(accuracy))
        result_file.write('\n')
    all_accuracy = all_correct/all_total
    print(f"模型在mmlu上的准确率为: {all_accuracy:.2%}")
    result_file.write("mmlu")
    result_file.write('    ')
    result_file.write(str(all_accuracy))
    result_file.write('\n')
    result_file.close()