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


# def create_decodelayer_forward(device):
#     def my_forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#         """
#         hidden_states = hidden_states.to(device)
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask.to(device),
#             position_ids=position_ids.to(device),
#             past_key_value=move_to_device(past_key_value,device),
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs
#     return my_forward


# def create_mtiHPA_copy_forward(module:dict,module_device,module_copy_device,half_idx = 3):
#     def my_forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

#         seq_length_with_past = seq_length
#         past_key_values_length = 0
        
        
#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length

#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)
#         # embed positions
#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
#         )

#         hidden_states = inputs_embeds

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None

#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             past_key_value = past_key_values[idx] if past_key_values is not None else None

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, output_attentions, None)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                     None,
#                 )
#             else:
#                 if idx in module:
#                      # 假设第二维度表示序列长度，可以根据你的实际情况调整
#                     def run_layer_1(decoder_layer,half_idx, hidden_states, attention_mask, position_ids, past_key_value):
#                         hidden_states_1 = hidden_states[:half_idx, :, :].contiguous().to(module_device)  # 第一半
#                         attention_mask_1 = attention_mask[:half_idx, :, :].to(module_device)
#                         position_ids_1 = position_ids[:half_idx, :].to(module_device)
#                         if past_key_value is not None:
#                         #print("past_key_value is not None\n")
#                             past_key_value_1 = (past_key_value[0][:half_idx, :, :, :], past_key_value[1][:half_idx, :, :, :])
#                             #past_key_value_2 = (past_key_value[0][half_idx:, :, :, :], past_key_value[1][half_idx:, :, :, :])
#                         else:
#                             past_key_value_1 = None
#                             #past_key_value_2 = None
#                         past_key_value_1 = move_to_device(past_key_value_1,module_device)
#                         forward1 = create_decodelayer_forward(module_device)
#                         decoder_layer.forward = forward1.__get__(decoder_layer,LlamaDecoderLayer)
#                         layer_outputs_1 = decoder_layer(
#                             hidden_states_1,
#                             attention_mask=attention_mask_1,  # 更新对应的 attention_mask
#                             position_ids=position_ids_1,  # 更新 position_ids
#                             past_key_value=past_key_value_1,
#                             output_attentions=output_attentions,
#                             use_cache=use_cache,
#                         )
#                         return layer_outputs_1
#                     def run_layer_2(module,idx, half_idx,  hidden_states, attention_mask, position_ids, past_key_value):
#                         hidden_states_2 = hidden_states[half_idx:, :, :].contiguous().to(module_copy_device)  # 第二半
#                         attention_mask_2 = attention_mask[half_idx:, :, :].to(module_copy_device)
#                         position_ids_2 = position_ids[half_idx:, :].to(module_copy_device)
#                         if past_key_value is not None:
#                             past_key_value_2 = (past_key_value[0][half_idx:, :, :, :], past_key_value[1][half_idx:, :, :, :])
#                         else:
#                             past_key_value_2 = None
#                         past_key_value_2 = move_to_device(past_key_value_2,module_copy_device)
#                         forward2 = create_decodelayer_forward(module_copy_device)
#                         module[idx].forward = forward2.__get__(module[idx],LlamaDecoderLayer)
#                         layer_outputs_2 = module[idx](
#                             hidden_states_2,
#                             attention_mask=attention_mask_2,  # 更新对应的 attention_mask
#                             position_ids=position_ids_2,  # 更新 position_ids
#                             past_key_value=past_key_value_2,
#                             output_attentions=output_attentions,
#                             use_cache=use_cache,
#                         )
#                         return layer_outputs_2
                    
#                     from concurrent.futures import wait
#                     with ThreadPoolExecutor(max_workers=2) as executor:
#                         # 记录开始的时间
#                         t_start_1 = time.time()
#                         # 提交第一个任务
#                         future1 = executor.submit(run_layer_1, decoder_layer, half_idx, hidden_states, attention_mask, position_ids, past_key_value)
#                         t_end_1 = time.time()  # 记录第一个任务提交后的时间

#                         # 提交第二个任务
#                         t_start_2 = time.time()
#                         future2 = executor.submit(run_layer_2, module, idx, half_idx, hidden_states, attention_mask, position_ids, past_key_value)
#                         t_end_2 = time.time()  # 记录第二个任务提交后的时间

#                         # 获取第一个任务的结果，并记录结束时间
#                         layer_outputs_1 = future1.result()
#                         t_finish_1 = time.time()

#                         # 获取第二个任务的结果，并记录结束时间
#                         layer_outputs_2 = future2.result()
#                         t_finish_2 = time.time()

#                     # 计算每个任务的运行时间
#                     execution_time_1 = t_finish_1 - t_start_1
#                     execution_time_2 = t_finish_2 - t_start_2

#                     # 打印每个任务的执行时间
#                     print(f"Thread 1 execution time (including waiting time for submission): {execution_time_1:.4f} seconds")
#                     print(f"Thread 2 execution time (including waiting time for submission): {execution_time_2:.4f} seconds")
#                     total_time = max(t_finish_1, t_finish_2) - min(t_start_1, t_start_2)
#                     print(f"Total time for both threads: {total_time:.4f} seconds")
#                     print("=="*50,"\n")
#                     assert len(layer_outputs_1) == len(layer_outputs_2), "两个输出的长度不一致"
#                     layer_outputs = recursive_concat(layer_outputs_1, layer_outputs_2, module_device,dim=0)
#                     LlamaAttention.forward = create_att_forward()
#                 else:
                    
#                     layer_outputs = decoder_layer(
#                         hidden_states,
#                         attention_mask=attention_mask,
#                         position_ids=position_ids,
#                         past_key_value=past_key_value,
#                         output_attentions=output_attentions,
#                         use_cache=use_cache,
#                     )
#             hidden_states = layer_outputs[0]
            
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
        
#         hidden_states = self.norm(hidden_states)
#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
    
#     return my_forward


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

batch_size = 20

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
        module_copy = copy_module(4,module_copy_device,model)
        # module_copy_1 = copy_module(4,device1,model)
        # module_copy_2 = copy_module(8,device1,model)
        # module_copy_3 = copy_module(14,device1,model)
        module = {
            4:module_copy,
            # 8:module_copy_2,
            # 14:module_copy_3
        }
        #LlamaModel.forward = create_mtiHPA_copy_forward(module,module_device,module_copy_device,half_idx=10)
       # LlamaModel.forward = create_mtiHPA_copy_forward({},module_device,module_copy_device,half_idx=10)
        LlamaModel.forward = create_monitor_modules_forward()
        decode_forward = create_monitor_atten_ffn_forward()
        LlamaDecoderLayer.forward = decode_forward.__get__(None,LlamaDecoderLayer)
        # part1 = end_event.record()
        # torch.cuda.synchronize()


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
main_filename =  f"interference_HPA_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"
while os.path.exists(f"{file_index}_interference_HPA_persist{presist}_bs{batch_size}_tokens{max_new_tokens}.txt"):
    file_index += 1
file_name = f"{file_index}_"+ main_filename
file_name_1 = main_filename[:len(".txt")]


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
batch = [[] for _ in range(40)]
batch_atten = [[] for _ in range(40)]
batch_ffn = [[] for _ in range(40)]
batch_norm = [[] for _ in range(40)]


def get_batch_time(inference_time):
    batch = [[] for _ in range(40)]
    for i in range(len(m_inference_time)):
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

print("m_inference_time:",len(m_inference_time))
print("atten_time:",len(atten_time))
print("ffn_time:",len(ffn_time))
print("norm_time:",len(norm_time))

batch = get_batch_time(m_inference_time)
print("batch:",len(batch))
mean_values,max_values,max_index = get_mean_max_time(get_batch_time(m_inference_time))
#batch_atten = get_batch_time(atten_time)
mean_values_attn,max_values_attn,max_index_attn = get_mean_max_time(get_batch_time(atten_time))
mean_values_ffn,max_values_ffn,max_index_ffn = get_mean_max_time(get_batch_time(ffn_time))
mean_values_norm,max_values_norm,max_index_norm = get_mean_max_time(get_batch_time(norm_time))
# for i in range(len(m_inference_time)):
#     # batch = m_inference_time[i:i+40]
#     # mean_values.append(sum(batch) / len(batch))
#     # max_values.append(max(batch))
#     batch[i % 40].append(m_inference_time[i])
#     batch_atten[i % 40].append(atten_time)
#     batch_ffn[i % 40].append(ffn_time)
#     batch_norm[i % 40].append(norm_time)

# for i in range(40):
#     mean_values.append(sum(batch[i]) /len(batch[i]))    
#     max_values.append(max(batch[i]))
#     max_index.append(batch[i].index(max(batch[i])))

#     mean_values_ffn.append(sum(batch_ffn[i]) /len(batch[i]))    
#     max_values_ffn.append(max(batch[i]))
#     max_index_ffn.append(batch[i].index(max(batch[i])))


# print(f"batch len: {len(batch)}")
# 输出结果
# print("Mean values:", mean_values)
# print("Max values:", max_values)

# with open('HPA Module.txt', 'w') as file:
    #file.write(f"xlabel = {list(range(2, 25, 2))}\n")
   


with open(file_name, 'w') as file:
    file.write(f"==============llama2-13b {file_name_1}===================\n")
    
    file.write(f"======================= module time ========================\n")
   #file.write(f"inference_time_avg = {m_inference_time}\n")
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