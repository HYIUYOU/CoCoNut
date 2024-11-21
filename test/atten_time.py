import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import sys
import os
import random
import threading
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
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
from utils.utils import *
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
logger = logging.get_logger(__name__)

q_time = []
k_time = []
v_time = []

rotary_emb_time = []
apply_rotary_pos_emb_time = []
attn_weights_matmul_time = []
attn_weights_softmax_time = []
attn_output_matmul_time = []
attn_output_linear_time = []

kv_cache_time = []

total_time = []
def create_monitor_atten_op_forward():
    global q_time, k_time ,v_time , rotary_emb_time ,apply_rotary_pos_emb_time ,attn_weights_matmul_time ,\
          attn_weights_softmax_time ,attn_output_linear_time, total_time,kv_cache_time,attn_output_matmul_time
   
    def my_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        total_start_time = time.time()
        bsz, q_len, _ = hidden_states.size()

        q_start_time = time.time()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        q_time.append(time.time() - q_start_time)

        k_start_time = time.time()
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        k_time.append(time.time() - k_start_time)

        v_start_time = time.time()
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        v_time.append(time.time() - v_start_time)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        r_start_time = time.time()
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        torch.cuda.synchronize()
        rotary_emb_time.append(time.time() - r_start_time)

        apply_start_time = time.time()
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        torch.cuda.synchronize()
        apply_rotary_pos_emb_time.append(time.time() - apply_start_time)
        # [bsz, nh, t, hd]
        kv_cache_start_time = time.time()
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        torch.cuda.synchronize()    
        kv_cache_time.append(time.time() - kv_cache_start_time)
        
            # print(f"key_states device {key_states.device},shape:{key_states.shape}\n")
            # print(f"value_states device {value_states.device},shape:{value_states.shape}\n")
            # for i, tensor in enumerate(past_key_value):
            #     print(f"Tensor {i} device: {tensor.device}, shape: {tensor.shape}\n")
        past_key_value = (key_states, value_states) if use_cache else None
        
        
        attn_weights_start_time = time.time()
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        torch.cuda.synchronize()
        attn_weights_matmul_time.append(time.time() - attn_weights_start_time)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights_softmax_start_time  =time.time()
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_softmax_time.append(time.time() - attn_weights_softmax_start_time)

        attn_output_matmul_start_time = time.time()
        attn_output = torch.matmul(attn_weights, value_states)
        torch.cuda.synchronize()
        attn_output_matmul_time.append(time.time() - attn_output_matmul_start_time)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output_linear_start_time  = time.time()
        attn_output = self.o_proj(attn_output)
        attn_output_linear_time.append(time.time() - attn_output_linear_start_time)

        if not output_attentions:
            attn_weights = None
        torch.cuda.synchronize()
        total_time.append(time.time() - total_start_time)
        return attn_output, attn_weights, past_key_value
    return my_forward

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

batch_size = 2

config = AutoConfig.from_pretrained(model_id)
with accelerate.init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

device_map = {"": "cuda:2"}
presist = 1
module_device = device2
module_copy_device = device3


for j in list(range(20, 21, 10)):
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
        # module_copy = copy_module(4,module_copy_device,model)
        # # module_copy_1 = copy_module(4,device1,model)
        # # module_copy_2 = copy_module(8,device1,model)
        # # module_copy_3 = copy_module(14,device1,model)
        # module = {
        #     4:module_copy,
        #     # 8:module_copy_2,
        #     # 14:module_copy_3
        # }
        # #LlamaModel.forward = create_mtiHPA_copy_forward(module,module_device,module_copy_device,half_idx=10)
        # LlamaModel.forward = create_mtiHPA_copy_forward({},module_device,module_copy_device,half_idx=10)
        # decode_forward = create_monitor_atten_ffn_forward()
        # LlamaDecoderLayer.forward = decode_forward.__get__(None,LlamaDecoderLayer)
        # part1 = end_event.record()
        # torch.cuda.synchronize()
        attten_forward = create_monitor_atten_op_forward()
        LlamaAttention.forward = attten_forward.__get__(None,LlamaAttention)

        # deploy = start_event.elapsed_time(end_event) / 1000  # 转换为秒
        # deploy_time_1.append(deploy)
        
        prompts = questions[:j]
        # # truncation=True：确保输入序列不会超过模型的最大长度，超长序列会被截断。
        # part2 = start_event.record()
        start =  time.time()
        # # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")  # 将输入转移到GPU
        history_dict = {}
        chat(rps = j/presist,model=model,tokenizer=tokenizer,prompts=prompts,batch_size = batch_size,max_new_tokens=max_new_tokens,history_dict=history_dict,device=module_device)
        

        batch_num = j/batch_size
        print("inference time:",(time.time()-start)/batch_num,"\n")
        # Generate
        #generate_ids = model.generate(inputs.input_ids, max_length=1280)

        # 解码并输出生成结果
        #outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # end_event.record()
        # torch.cuda.synchronize()
        # inference = start_event.elapsed_time(end_event) / 1000
        # print("inference time",inference,"\n")
        # print("=="*50,"\n")
        # inference_time_1.append(inference)
        # all_time_1.append(deploy+inference)
        del model
        del tokenizer
        torch.cuda.empty_cache()  # 清空未使用的显存
        gc.collect()  # 进行垃圾回收
        
        print(f"Iteration {i+1} complete, memory cleared.")
# 生成文件名
file_index = 1
main_filename =  f"atten_time_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"
while os.path.exists(f"{file_index}_atten_time_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"):
    file_index += 1
file_name = f"{file_index}_"+ main_filename
file_name_1 = main_filename[:len(".txt")]




# batch = [[] for _ in range(40)]
# batch_atten = [[] for _ in range(40)]
# batch_ffn = [[] for _ in range(40)]
# batch_norm = [[] for _ in range(40)]


def get_batch_time(inference_time):
    batch = [[] for _ in range(40)]
    for i in range(len(inference_time)):
        batch[i % 40].append(inference_time[i])
    return batch

def get_mean_max_time(batch):
    mean_values = []
    max_values = []
    max_index = []
    for i in range(40):
        mean_values.append(sum(batch[i]) /len(batch[i]))    
        max_values.append(max(batch[i]))
        max_index.append(batch[i].index(max(batch[i])))
    return mean_values,max_values,max_index

print("q_time:",len(q_time))
print("k_time:",len(k_time))
print("v_time:",len(v_time))
print("rotary_emb_time:",len(rotary_emb_time))

# batch = get_batch_time(m_inference_time)
# print("batch:",len(batch))
mean_total,max_total,max_total_index = get_mean_max_time(get_batch_time(total_time))

mean_q,max_q,max_q_index = get_mean_max_time(get_batch_time(q_time))
mean_k,max_k,max_k_index = get_mean_max_time(get_batch_time(k_time))
mean_v,max_v,max_v_index = get_mean_max_time(get_batch_time(v_time))

mean_kvcache,max_kvcache,max_kvcache_index = get_mean_max_time(get_batch_time(kv_cache_time))
#batch_atten = get_batch_time(atten_time)
mean_rotary_emb,max_rotary_emb,max_rotary_emb_index = get_mean_max_time(get_batch_time(rotary_emb_time))
mean_apply_rotary_pos_emb,max_apply_rotary_pos_emb,max_apply_rotary_pos_emb_index = get_mean_max_time(get_batch_time(apply_rotary_pos_emb_time))
mean_attn_weights_matmul,max_attn_weights_matmul,max_attn_weights_matmul_index = get_mean_max_time(get_batch_time(attn_weights_matmul_time))
mean_attn_weights_softmax,max_attn_weights_softmax,max_attn_weights_softmax_index = get_mean_max_time(get_batch_time(attn_weights_softmax_time))
mean_attn_output_linear,max_attn_output_linear,max_attn_output_linear_index = get_mean_max_time(get_batch_time(attn_output_linear_time))

mean_attn_output_matmul,max_attn_output_matmul,max_attn_output_matmul_index = get_mean_max_time(get_batch_time(attn_output_matmul_time))


# print(f"batch len: {len(batch)}")
# 输出结果
# print("Mean values:", mean_values)
# print("Max values:", max_values)

# with open('HPA Module.txt', 'w') as file:
    #file.write(f"xlabel = {list(range(2, 25, 2))}\n")
   


with open(file_name, 'w') as file:
    file.write(f"==============llama2-13b {file_name_1}===================\n")
    file.write(f"\n")

    file.write(f"======================= atten total time ========================\n")
   #file.write(f"inference_time_avg = {m_inference_time}\n")
    file.write(f"Mean_total_values = {mean_total}\n")
    file.write(f"Max_total_values = {max_total}\n")
    file.write(f"Max_total_index = {max_total_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten KV cache time ========================\n")
   #file.write(f"inference_time_avg = {m_inference_time}\n")
    file.write(f"Mean_kvcache_values = {mean_kvcache}\n")
    file.write(f"Max_kvcache_values = {max_kvcache}\n")
    file.write(f"Max_kvcache_index = {max_kvcache_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten KV time ========================\n")
   #file.write(f"inference_time_avg = {m_inference_time}\n")
    file.write(f"Mean_q_values = {mean_q}\n")
    file.write(f"Max_q_values = {max_q}\n")
    file.write(f"Max_q_index = {max_q_index}\n")

    file.write(f"\n")

    file.write(f"Mean_k_values = {mean_k}\n")
    file.write(f"Max_k_values = {max_k}\n")
    file.write(f"Max_k_index = {max_k}\n")

    file.write(f"\n")

    file.write(f"Mean_v_values = {mean_v}\n")
    file.write(f"Max_v_values = {max_v}\n")
    file.write(f"Max_v_index = {max_v_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten rotary time ========================\n")
    file.write(f"Mean_rotary_emb_values = {mean_rotary_emb}\n")
    file.write(f"Max_rotary_emb_values = {max_rotary_emb}\n")
    file.write(f"Max_rotary_emb_index = {max_rotary_emb_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten apply rotary time ========================\n")
    file.write(f"Mean_apply_rotary_emb_values = {mean_apply_rotary_pos_emb}\n")
    file.write(f"Max_apply_rotary_emb_values = {max_apply_rotary_pos_emb}\n")
    file.write(f"Max_apply_rotary_emb_index = {max_apply_rotary_pos_emb_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten weights matmul time ========================\n")
    file.write(f"Mean_attn_weights_matmul_values = {mean_attn_weights_matmul}\n")
    file.write(f"Max_attn_weights_matmul_values = {max_attn_weights_matmul}\n")
    file.write(f"Max_attn_weights_matmul_index = {max_attn_weights_matmul_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten weights softmax time ========================\n")
    file.write(f"Mean_attn_weights_softmax_values = {mean_attn_weights_softmax}\n")
    file.write(f"Max_attn_weights_softmax_values = {max_attn_weights_softmax}\n")
    file.write(f"Max_attn_weights_softmax_index = {max_attn_weights_softmax_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten output matmul time ========================\n")
    file.write(f"Mean_attn_output_matmul_values = {mean_attn_output_matmul}\n")
    file.write(f"Max_attn_output_matmul_values = {max_attn_output_matmul}\n")
    file.write(f"Max_attn_output_matmul_index = {max_attn_output_matmul_index}\n")

    file.write(f"\n")

    file.write(f"======================= atten output linear time ========================\n")
    file.write(f"Mean_attn_output_linear_values = {mean_attn_output_linear}\n")
    file.write(f"Max_attn_output_linear_values = {max_attn_output_linear}\n")
    file.write(f"Max_attn_output_linear_index = {max_attn_output_linear_index}\n")

    # file.write(f"module_0_time = {batch[0]}\n")
    # file.write(f"module_1_time = {batch[1]}\n")
    # file.write(f"module_2_time = {batch[2]}\n")
    # file.write(f"module_3_time = {batch[3]}\n")
    # file.write(f"module_14_time = {batch[14]}\n")
    # file.write(f"module_15_time = {batch[15]}\n")
    # file.write(f"module_16_time = {batch[16]}\n")
    # file.write(f"module__36_time = {batch[36]}\n")
    # file.write(f"module__37_time = {batch[37]}\n")
    # file.write(f"module__38_time = {batch[38]}\n")
    # file.write(f"module__39_time = {batch[39]}\n")