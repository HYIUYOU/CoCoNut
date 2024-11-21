import argparse
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

config = AutoConfig.from_pretrained(model_id)
with accelerate.init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

device_map = {"": "cuda:0"}

all_time_1 = []
deploy_time_1 = []
inference_time_1 = []

# 加载模型和tokenizer
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

cpu_start = time.time()
start_event.record()
model = LlamaForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    offload_folder="offload",
    offload_state_dict=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

end_event.record()
torch.cuda.synchronize()


deploy = start_event.elapsed_time(end_event) / 1000  # 转换为秒
deploy_time_1.append(deploy)

def send_requests():
    global count
    count += 1
    from datetime import datetime
    print(f"请求发送时间：{datetime.now()}, 发送轮次：{count}")
    prompts = random.sample(questions, rps)
    cpu_start = time.time()
    start_event.record()
    history_dict = {}
    chat(model=model, tokenizer=tokenizer, prompts=prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, history_dict=history_dict, device=device0)
    end_event.record()
    torch.cuda.synchronize()
    cpu_end = time.time()
    inference = start_event.elapsed_time(end_event) / 1000  # 转换为秒
    cpu_inference = cpu_end - cpu_start
    inference_time_1.append(inference)
    
    print("inference time (GPU):", inference, "\n")
    print("inference time (CPU):", cpu_inference, "\n")
    print("=="*50, "\n")
    
    all_time_1.append(deploy + inference)
        

# 持续5秒随机发送请求
start_time = time.time()
count = 0
threads = []
for _ in range(1):
    t = threading.Thread(target=send_requests)
    t.start()
    #time.sleep(1)
    threads.append(t)

for t in threads:
    t.join()

del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()

all_time_avg.append(sum(all_time_1) / len(all_time_1))
deploy_time_avg.append(sum(deploy_time_1) / len(deploy_time_1))
inference_time_avg.append(sum(inference_time_1) / len(inference_time_1)/rps)

all_time_max.append(max(all_time_1))
deploy_time_max.append(max(deploy_time_1))
inference_time_max.append(max(inference_time_1)/rps)

# 生成文件名
file_index = 1
while os.path.exists(f"{file_index}_rps{rps}_bs{batch_size}_tokens{max_new_tokens}.txt"):
    file_index += 1
file_name = f"{file_index}_rps{rps/args.time}_time{args.time}_bs{batch_size}_tokens{max_new_tokens}.txt"
file_name_1 = f"{file_index}_rps{rps/args.time}_time{args.time}_bs{batch_size}_tokens{max_new_tokens}"

with open(file_name, 'w') as file:
    file.write(f"==============llama2-13b {file_name_1}===================\n")
    file.write(f"all_time_avg = {all_time_avg}\n")
    file.write(f"deploy_time_avg = {deploy_time_avg}\n")
    file.write(f"inference_time_avg = {inference_time_avg}\n")
    file.write(f"all_time_max = {all_time_max}\n")
    file.write(f"deploy_time_max = {deploy_time_max}\n")
    file.write(f"inference_time_max = {inference_time_max}\n")