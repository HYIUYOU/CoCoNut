from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import sys
import os
import csv
from torch import nn
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb,LlamaDecoderLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelForCausalLM
import accelerate
from accelerate import dispatch_model
utils_path = os.path.abspath("/root/heyiyuan")
sys.path.insert(0, utils_path)
from utils.utils import *
from utils.dataset.questions import questions
from utils.llama_chat import chat
import time
import argparse

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
logger = logging.get_logger(__name__)

# 定义常量
NUM_LAYERS = 40

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 使用argparse模块从命令行接受参数，示例：python 1.py --rps 50 -bs 10 -mnt 128 
def parse_device(device_str):
    """辅助函数:将输入的设备编号转换为CUDA设备字符串。"""
    return f"cuda:{device_str}" if device_str.isdigit() else device_str

parser = argparse.ArgumentParser(description='参数注入')
parser.add_argument('-rps', '--rps', type=int, default=20, help='每秒请求数')
parser.add_argument('-m', '--model', type=str, default="/root/heyiyuan/Models/Llama-2-13b-chat-hf", help='模型ID')
parser.add_argument('-bs', '--batch_size', type=int, default=10, help='批处理大小')
parser.add_argument('-mnt', '--max_new_tokens', type=int, default=128, help='生成的最大新token数')
parser.add_argument('-p', '--presist', type=int, default=1, help='持续时间')
parser.add_argument('-d','--device', type=str, default="2", help='设置模型的计算设备,输入数字代表cuda设备编号,例如0代表cuda:0')
parser.add_argument('-cd','--copy_device', type=str, default="3", help='设置module的计算设备,输入数字代表cuda设备编号,例如0代表cuda:0')

args = parser.parse_args()
rps = args.rps
model_id = args.model
batch_size = args.batch_size
max_new_tokens = args.max_new_tokens
presist = args.presist  
device = parse_device(args.device)
copy_device = parse_device(args.copy_device)

# 使用转换后的设备参数
device_map = {"": device}
module_device = device
module_copy_device = copy_device

config = AutoConfig.from_pretrained(model_id)
with accelerate.init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


for j in list(range(rps*presist, rps*presist+1, batch_size)):
    all_time_1 = []
    deploy_time_1 = []
    inference_time_1 = []
    for i in range(1):
        start = start_event.record()
        model =LlamaForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        #LlamaModel.forward = create_mtiHPA_copy_forward({},module_device,module_copy_device,half_idx=10)
        LlamaModel.forward = create_monitor_modules_forward()
        decode_forward = create_monitor_atten_ffn_forward()
        LlamaDecoderLayer.forward = decode_forward.__get__(None,LlamaDecoderLayer)
        prompts = questions[:j]
        start =  time.time()
        history_dict = {}
        chat(rps = j/presist,model=model,tokenizer=tokenizer,prompts=prompts,batch_size = batch_size,max_new_tokens=max_new_tokens,history_dict=history_dict,device=module_device)
        batch_num = j/batch_size
        print("inference time:",(time.time()-start)/batch_num,"\n")
        del model
        del tokenizer
        torch.cuda.empty_cache()  # 清空未使用的显存
        gc.collect()  # 进行垃圾回收
        print(f"Iteration {i+1} complete, memory cleared.")




mean_values = []
max_values = []
max_index = []
mean_values_attn = []
max_values_attn = []
max_index_attn = []
mean_values_ffn = []
max_values_ffn = []
max_index_ffn = []
mean_values_norm = []
max_values_norm = []
max_index_norm = []
modules = [[] for _ in range(40)]
modules_atten = [[] for _ in range(40)]
modules_ffn = [[] for _ in range(40)]
modules_norm = [[] for _ in range(40)]


def get_modules_time(inference_time, num_layers=NUM_LAYERS):
    """获取每个modules的时间"""
    batch = [[] for _ in range(num_layers)]
    for i in range(len(inference_time)):
        batch[i % num_layers].append(inference_time[i])
    return batch

def get_mean_max_time(batch):
    """计算每个modules的平均值、最大值和最大值索引"""
    mean_values = [sum(batch[i]) / len(batch[i]) for i in range(len(batch))]
    max_values = [max(batch[i]) for i in range(len(batch))]
    max_index = [batch[i].index(max(batch[i])) for i in range(len(batch))]
    return mean_values, max_values, max_index

print("m_inference_time:",len(m_inference_time))
print("atten_time:",len(atten_time))
print("ffn_time:",len(ffn_time))
print("norm_time:",len(norm_time))

modules = get_modules_time(m_inference_time)
print("batch:",len(modules))

mean_values,max_values,max_index = get_mean_max_time(get_modules_time(m_inference_time))
mean_values_attn,max_values_attn,max_index_attn = get_mean_max_time(get_modules_time(atten_time))
mean_values_ffn,max_values_ffn,max_index_ffn = get_mean_max_time(get_modules_time(ffn_time))
mean_values_norm,max_values_norm,max_index_norm = get_mean_max_time(get_modules_time(norm_time))

# 生成文件名
file_index = 1
main_filename =  f"module_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"
while os.path.exists(f"results/{file_index}_module_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"):
    file_index += 1
file_name = f"results/{file_index}_"+ main_filename
file_name_1 = main_filename[:len(".txt")]


with open(file_name, 'w') as file:
    file.write(f"==============llama2-13b {file_name_1}===================\n")
    
    file.write(f"======================= module time ========================\n")
    file.write(f"Mean_values = {mean_values}\n")
    file.write(f"Max_values = {max_values}\n")
    file.write(f"Max_index = {max_index}\n")

    file.write(f"======================= atten time ========================\n")
    file.write(f"Mean_values_attn = {mean_values_attn}\n")
    file.write(f"Max_values_attn = {max_values_attn}\n")
    file.write(f"Max_index_attn = {max_index_attn}\n")

    file.write(f"======================= ffn time ========================\n")
    file.write(f"Mean_values_ffn = {mean_values_ffn}\n")
    file.write(f"Max_values_ffn = {max_values_ffn}\n")
    file.write(f"Max_index_ffn = {max_index_ffn}\n")

    file.write(f"======================= norm time ========================\n")
    file.write(f"Mean_values_norm = {mean_values_norm}\n")
    file.write(f"Max_values_norm = {max_values_norm}\n")
    file.write(f"Max_index_norm = {max_index_norm}\n")





main_filename_csv =  f"module_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.csv"
while os.path.exists(f"results/{file_index}_module_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.csv"):
    file_index += 1
file_name_csv = f"results/{file_index}_"+ main_filename_csv
file_name_1_csv = main_filename_csv[:len(".csv")]

# 写入CSV文件
with open(file_name_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(["idx", "mean", "max"] + list(range(len(modules[0]))))

    # 写入数据行
    for idx in range(len(mean_values)):
        row = [idx, mean_values[idx], max_values[idx]] + modules[idx]
        writer.writerow(row)

print(f"Data saved to {file_name_csv} in CSV format.")

