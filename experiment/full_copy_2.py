from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import sys
import os
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

from utils.dataset.questions import  questions
from utils.llama_chat import chat
import time

all_time_avg = []
deploy_time_avg = []
inference_time_avg = []

all_time_max = []
deploy_time_max = []
inference_time_max = []

model_id = "../Models/Llama-2-13b-chat-hf"
config = AutoConfig.from_pretrained(model_id)
with accelerate.init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

# device_map = accelerate.infer_auto_device_map(dummy_model, max_memory={
#     0: "40GiB",   # 分配40GiB给GPU 0
#     1: "0GiB",    # 禁用GPU 1
#     2: "0GiB",    # 禁用GPU 2
#     3: "0GiB",    # 禁用GPU 3
#     "cpu": "30GiB" # 分配30GiB给CPU，允许offload
# })

def deploy(device,model_id):
    device_map = {"": device}
    model =LlamaForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        # load_in_8bit=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True
        )
        
    #model = dispatch_model(model, device_map={"": "cuda:0"})
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #model.model.layers[4] = model.model.layers[4].to(device1)
    tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer


device_map = {"": "cuda:2"}
for j in list(range(98, 99, 3)):
    all_time_1 = []
    deploy_time_1 = []
    inference_time_1 = []
    for i in range(5):
        start = time.time()
        # 加载模型和tokenizer
        #model = LlamaForCausalLM.from_pretrained(model_id,offload_folder="offload").bfloat16().to(device0)
        model =LlamaForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        # load_in_8bit=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.bfloat16,
        offload_folder="offload",
        offload_state_dict=True
        )
        
        #model = dispatch_model(model, device_map={"": "cuda:0"})
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        #model.model.layers[4] = model.model.layers[4].to(device1)
        tokenizer.pad_token = tokenizer.eos_token
        part1 = time.time()
        deploy = part1 - start
        deploy_time_1.append(deploy)
        
        prompts = questions[:j]
        # truncation=True：确保输入序列不会超过模型的最大长度，超长序列会被截断。
        part2 = time.time()
        # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")  # 将输入转移到GPU
        history_dict = {}
        chat(model=model,tokenizer=tokenizer,prompts=prompts,batch_size = 3,max_new_tokens=64,history_dict=history_dict,device=device2)

        # Generate
        #generate_ids = model.generate(inputs.input_ids, max_length=1280)

        # 解码并输出生成结果
        #outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        inference = time.time() - part2
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


with open('fullcopy_2_1.txt', 'w') as file:
    file.write(f"==============llama2-13b bs = 3 ===================\n")
    #file.write(f"xlabel = {list(range(10, 91, 3))}\n")
    file.write(f"all_time_avg = {all_time_avg}\n")
    file.write(f"deploy_time_avg = {deploy_time_avg}\n")
    file.write(f"inference_time_avg = {inference_time_avg}\n")

    file.write(f"all_time_max = {all_time_max}\n")
    file.write(f"deploy_time_max = {deploy_time_max}\n")
    file.write(f"inference_time_max = {inference_time_max}\n")



    