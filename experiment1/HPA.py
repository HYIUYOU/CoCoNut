import argparse
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import sys
import os
import random
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
import math
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from typing import List, Optional, Tuple, Union
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM
import accelerate
from accelerate import dispatch_model
from utils.utils import create_mtiHPA_copy_forward,copy_module,m_inference_time
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
logger = logging.get_logger(__name__)

from utils.dataset.questions import questions
from utils.llama_chat import chat
# from utils.utils import create_mtiHPA_copy_forward, copy_module
import time
import copy
all_time_avg = []
deploy_time_avg = []
inference_time_avg = []

all_time_max = []
deploy_time_max = []
inference_time_max = []

# 使用argparse模块从命令行接受参数，示例：python 1.py --rps 50 --batch_size 10 --max_new_tokens 128 
parser = argparse.ArgumentParser(description='参数注入')
parser.add_argument('--rps', type=int, default=10, help='每秒请求数')
parser.add_argument('--model', type=str, default="../Models/Llama-2-13b-chat-hf", help='模型ID')
parser.add_argument('--batch_size', type=int, default=10, help='批处理大小')
parser.add_argument('--max_new_tokens', type=int, default=128, help='生成的最大新token数')
parser.add_argument('--time', type=int, default=5, help='持续时间')
args = parser.parse_args()

rps = args.rps*args.time
model_id = args.model
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens

batch_size = 4

config = AutoConfig.from_pretrained(model_id)
with accelerate.init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

device_map = {"": "cuda:0"}
for j in list(range(4, 5, 4)):
    all_time_1 = []
    deploy_time_1 = []
    inference_time_1 = []
    for i in range(1):
        start = start_event.record()
        # 加载模型和tokenizer
       # model = LlamaForCausalLM.from_pretrained(model_id,offload_folder="offload").bfloat16().to(device0)
        model =LlamaForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        # load_in_8bit=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True
        )
        # 加载模型和tokenizer
        #model = dispatch_model(model, device_map={"": "cuda:0"})
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        #model.model.layers[4] = model.model.layers[4].to(device1)
        tokenizer.pad_token = tokenizer.eos_token
        # module_id = 4
        # module_copy = copy.deepcopy(model.model.layers[module_id])
        # module_copy = module_copy.to(device1)
        module_copy = copy_module(4,device1,model)
        module_copy_1 = copy_module(4,device1,model)
        module_copy_2 = copy_module(8,device1,model)
        module_copy_3 = copy_module(14,device1,model)
        module = {
            4:module_copy,
            8:module_copy_2,
            14:module_copy_3
        }
        LlamaModel.forward = create_mtiHPA_copy_forward(module,device0,device1)
        part1 = end_event.record()
        torch.cuda.synchronize()


        deploy = start_event.elapsed_time(end_event) / 1000  # 转换为秒
        deploy_time_1.append(deploy)
        
        prompts = questions[:j]
        # truncation=True：确保输入序列不会超过模型的最大长度，超长序列会被截断。
        part2 = start_event.record()
        # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")  # 将输入转移到GPU
        history_dict = {}
        chat(rps = j/2,model=model,tokenizer=tokenizer,prompts=prompts,batch_size = batch_size,max_new_tokens=max_new_tokens,history_dict=history_dict,device=device0)

        # Generate
        #generate_ids = model.generate(inputs.input_ids, max_length=1280)

        # 解码并输出生成结果
        #outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        end_event.record()
        torch.cuda.synchronize()
        inference = start_event.elapsed_time(end_event) / 1000
        print("inference time",inference,"\n")
        print("=="*50,"\n")
        inference_time_1.append(inference)
        all_time_1.append(deploy+inference)
        del model
        del tokenizer
        torch.cuda.empty_cache()  # 清空未使用的显存
        gc.collect()  # 进行垃圾回收
        
        print(f"Iteration {i+1} complete, memory cleared.")
    all_time_avg.append(sum(all_time_1)/len(all_time_1))
    deploy_time_avg.append(sum(deploy_time_1)/len(deploy_time_1))
    inference_time_avg.append(sum(inference_time_1)/len(inference_time_1))

    all_time_max.append(max(all_time_1))
    deploy_time_max.append(max(deploy_time_1))
    inference_time_max.append(max(inference_time_1))

m_inference_time_avg =[]
l = len(m_inference_time)
m_inference_time_avg = [np.mean(m_inference_time[i:i+l]) for i in range(0, len(m_inference_time), l)]
m_inference_time_avg = [float(val) for val in m_inference_time_avg]
   
# 生成文件名
file_index = 1
while os.path.exists(f"{file_index}_bs{batch_size}_tokens{max_new_tokens}.txt"):
    file_index += 1
file_name = f" interference_HPA_{file_index}_time{5}_bs{batch_size}_tokens{max_new_tokens}.txt"
file_name_1 = f"{file_index}_time{5}_bs{batch_size}_tokens{max_new_tokens}"


# with open('HPA Module.txt', 'w') as file:
    #file.write(f"xlabel = {list(range(2, 25, 2))}\n")
   


with open(file_name, 'w') as file:
    file.write(f"==============llama2-13b HPA 1 {file_name_1}===================\n")
    file.write(f"xlabel = {list(range(2, 21, 2))}\n")
    file.write(f"all_time_avg = {all_time_avg}\n")
    file.write(f"deploy_time_avg = {deploy_time_avg}\n")
    file.write(f"inference_time_avg = {inference_time_avg}\n")
    file.write(f"all_time_max = {all_time_max}\n")
    file.write(f"deploy_time_max = {deploy_time_max}\n")
    file.write(f"inference_time_max = {inference_time_max}\n")
    file.write(f"======================= module time ========================\n")
    file.write(f"inference_time_avg = {m_inference_time_avg}\n")