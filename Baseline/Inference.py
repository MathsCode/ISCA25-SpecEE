import torch
from fastchat.model import get_conversation_template
import os
import time
from tqdm import trange
import json
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

import GPUtil
def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions
question_list = load_questions('/home/xujiaming/xujiaming/train_machine/PJY/ASPLOS-24/EAGLE/eagle/data/mt_bench/question.jsonl',begin=0,end=80)
model_name = "/share/datasets/public_models/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager")
model.eval()
output_ids_tot = 0
st = time.time()
for i in trange(80):
    torch.cuda.empty_cache()
    # print("===================== Question Id = ",question_list[i]['question_id']," ======================")
    # message = "What is the capital of France?"
    message = question_list[i]['turns'][0]
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "    
    input_ids=tokenizer([prompt]).input_ids
    seqlen = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model.generate(input_ids,max_new_tokens=256,temperature=1e-6)
    output_ids_tot += len(output_ids[0]) - seqlen
    output=tokenizer.decode(output_ids[0])
ed = time.time()
print('time:', (ed -st))
print('average time ',(ed-st)/output_ids_tot)
print(output_ids_tot/(ed-st))