from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from torch import nn
import math
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb,LlamaDecoderLayer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from typing import List, Optional, Tuple, Union
import time


device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
logger = logging.get_logger(__name__)


import copy

def copy_module(module_id,device,model):
    module_copy = copy.deepcopy(model.model.layers[module_id])
    module_copy = module_copy.to(device)
    return module_copy
def recursive_concat(output1, output2, device,dim=0):
    # 如果是元组，递归处理每个元素
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        return tuple(recursive_concat(o1, o2,device, dim=dim) for o1, o2 in zip(output1, output2))
    # 如果是张量，直接拼接
    elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.cat([output1.to(device), output2.to(device)], dim=dim)
    else:
        raise TypeError("Elements must be tensors or tuples of tensors")

def chunk_list(lst, batch_size=10):
    # 使用列表切片，将列表按照chunk_size进行分组
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def move_to_device(outputs, device):
    if isinstance(outputs, tuple):
        # 递归处理 tuple 中的每个元素
        return tuple(move_to_device(tensor, device) for tensor in outputs)
    elif isinstance(outputs, list):
        # 递归处理 list 中的每个元素
        return [move_to_device(tensor, device) for tensor in outputs]
    elif isinstance(outputs, torch.Tensor):
        # 如果是张量，则检查它是否已经在目标设备上
        if outputs.device != device:
            return outputs.to(device)  # 只有当张量不在目标设备上时，才移动
        return outputs  # 如果张量已经在目标设备上，直接返回
    else:
        # 其他类型保持不变 (比如 None)
        return outputs


def create_att_forward():
    def att_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            device = hidden_states.device
            bsz, q_len, _ = hidden_states.size()
            # self.q_proj = self.q_proj.to(device)
            # self.k_proj = self.k_proj.to(device)
            # self.v_proj = self.v_proj.to(device)
            # attention_mask = attention_mask.to(device)
            # print("===== 1 ====\n")
            #hidden_states = hidden_states.float()
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # print("===== 2 ====\n")
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                # print("===== 3 ====\n")
                past_key_value = move_to_device(past_key_value,device)
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # cos = cos.to(position_ids.device)
            # sin = sin.to(position_ids.device)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            # [bsz, nh, t, hd]
            
            if past_key_value is not None:
                # reuse k, v, self_attention
                # print("=="*50,"\n")
                # print("past_key_value[0] shape:", past_key_value[0].shape)
                # print("past_key_value[1] shape:", past_key_value[1].shape)
                # print("key_states shape:", key_states.shape)
                key_states = move_to_device(key_states,device)
                value_states = move_to_device(value_states,device)
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                # print("past_key_value[1] shape:", past_key_value[1].shape)
                # print("past_key_value[0] shape:", past_key_value[0].shape)
                # print("key_states shape:", key_states.shape)
                # print("=="*50,"\n")
                
            past_key_value = (key_states, value_states) if use_cache else None
            
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.to(device)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )
            
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights.to(device)
            value_states.to(device)
            # print('attn_weights device',attn_weights.device,"\n")
            # print('value_states device',value_states.device,"\n")
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            
            # self.o_proj = self.o_proj.to(device)
            # attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
    return att_forward


def create_VPA_offload_forward(module_id=4,self_device = device0 ,aim_device=device1):
    def my_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                if idx == module_id-1:
                    LlamaAttention.forward = create_att_forward(aim_device)
                    layer_outputs = move_to_device(layer_outputs,aim_device)
                    position_ids = position_ids.to(aim_device)
                    past_key_value = move_to_device(past_key_value,aim_device)
                    #output_attentions = output_attentions.to(device2) if output_attentions is not None else None
                if idx == module_id:
                    
                    LlamaAttention.forward = create_att_forward(self_device)
                    layer_outputs = move_to_device(layer_outputs,self_device)
                    position_ids = position_ids.to(self_device)
                    past_key_value = move_to_device(past_key_value,self_device)
            # print("model idx",idx)
            # print('--'*50)
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    return my_forward



def create_HPA_copy_forward(module_id,module_device,module_copy,module_copy_device):
    def my_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                
                # if idx == 3:
                #    LlamaAttention.forward = create_att_forward(device1)
                #    layer_outputs = move_to_device(layer_outputs,device1)
                #    position_ids = position_ids.to(device1)
                #    past_key_value = move_to_device(past_key_value,device1)
                #output_attentions = output_attentions.to(device2) if output_attentions is not None else None
                if idx == module_id:
                #    LlamaAttention.forward = create_att_forward(device0)
                #    layer_outputs = move_to_device(layer_outputs,device0)
                #    position_ids = position_ids.to(device0)
                #    past_key_value = move_to_device(past_key_value,device0)
                    # 分割 hidden_states 数据，将其一部分送入 layer1，另一部分送入 layer2
                    half_idx_1 = hidden_states.size(0) // 2  # 假设第二维度表示序列长度，可以根据你的实际情况调整
                    half_idx_2 = hidden_states.size(0) - half_idx_1
                    hidden_states_1 = hidden_states[:half_idx_1, :, :].contiguous().to(module_device)  # 第一半
                    hidden_states_2 = hidden_states[:half_idx_2, :, :].contiguous().to(module_copy_device)  # 第二半    
                    attention_mask_1 = attention_mask[:half_idx_1, :, :].to(device0)
                    attention_mask_2 = attention_mask[:half_idx_2, :, :].to(device1)
                    # 按照 batch_size 分割 position_ids
                    position_ids_1 = position_ids[:half_idx_1, :].to(module_device)
                    position_ids_2 = position_ids[half_idx_2:, :].to(module_copy_device)
                    # 并行执行两个 layers
                    # 将 past_key_value 也分割成两部分
                    if past_key_value is not None:
                        past_key_value_1 = (past_key_value[0][:half_idx_1, :, :, :], past_key_value[1][:half_idx_1, :, :, :])
                        past_key_value_2 = (past_key_value[0][half_idx_2:, :, :, :], past_key_value[1][half_idx_2:, :, :, :])
                    else:
                        past_key_value_1 = None
                        past_key_value_2 = None
                    layer_outputs_1 = decoder_layer(
                        hidden_states_1,
                        attention_mask=attention_mask_1,  # 更新对应的 attention_mask
                        position_ids=position_ids_1,  # 更新 position_ids
                        past_key_value=past_key_value_1,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    LlamaAttention.forward = create_att_forward(module_copy_device)
                    layer_outputs_2 = module_copy(
                        hidden_states_2,
                        attention_mask=attention_mask_2,  # 更新对应的 attention_mask
                        position_ids=position_ids_2,  # 更新 position_ids
                        past_key_value=past_key_value_2,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    # # 从返回的元组中提取 hidden_states
                    # hidden_states_1 = layer_outputs_1[0]  # 提取 layer_outputs_1 的第一个元素，即 hidden_states
                    # hidden_states_2 = layer_outputs_2[0]  # 提取 layer_outputs_2 的第一个元素，即 hidden_states

                    # # 然后对 hidden_states 进行拼接
                    # layer_outputs = torch.cat([hidden_states_1, hidden_states_2], dim=0).to(module_device)

                    # 将两个输出合并
                    assert len(layer_outputs_1) == len(layer_outputs_2), "两个输出的长度不一致"

                    # 对元组中的每个元素进行拼接
                    # layer_outputs = tuple(
                    #     torch.cat([move_to_device(elem1,module_device), move_to_device(elem2,module_device)], dim=0) 
                    #     for elem1, elem2 in zip(layer_outputs_1, layer_outputs_2)
                    # )
                    layer_outputs = recursive_concat(layer_outputs_1, layer_outputs_2, module_device,dim=0)
                    #layer_outputs = torch.cat([layer_outputs_1, layer_outputs_2], dim=1).to(module_device)
                    LlamaAttention.forward = create_att_forward(module_device)
                else:
                    
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
            # print("model idx",idx)
            # print('--'*50)
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    return my_forward

m_inference_time = []
#m_inference_time_avg = []


from concurrent.futures import ThreadPoolExecutor

# def create_mtiHPA_copy_forward(module:dict,module_device,module_copy_device):
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
#         global m_inference_time
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
                
#                 # if idx == 3:
#                 #    LlamaAttention.forward = create_att_forward(device1)
#                 #    layer_outputs = move_to_device(layer_outputs,device1)
#                 #    position_ids = position_ids.to(device1)
#                 #    past_key_value = move_to_device(past_key_value,device1)
#                 #output_attentions = output_attentions.to(device2) if output_attentions is not None else None
#                 if idx in module:
#                     # print(idx,"\n")
#                     # print(next(module[idx].parameters()).device,"\n")
#                     # print("="*30,"\n")
                    
#                 #    LlamaAttention.forward = create_att_forward(device0)
#                 #    layer_outputs = move_to_device(layer_outputs,device0)
#                 #    position_ids = position_ids.to(device0)
#                 #    past_key_value = move_to_device(past_key_value,device0)
#                     # 分割 hidden_states 数据，将其一部分送入 layer1，另一部分送入 layer2
#                     half_idx_1 = hidden_states.size(0) // 2  # 假设第二维度表示序列长度，可以根据你的实际情况调整
#                     #half_idx_1 = 1
#                     half_idx_2 = hidden_states.size(0) - half_idx_1
#                     hidden_states_1 = hidden_states[:half_idx_1, :, :].contiguous()
#                     # hidden_states_1 = hidden_states[:half_idx_1, :, :].contiguous().to(module_device)  # 第一半
#                     hidden_states_2 = hidden_states[:half_idx_2, :, :].contiguous().to(module_copy_device)  # 第二半    
#                     attention_mask_1 = attention_mask[:half_idx_1, :, :]
#                     # attention_mask_1 = attention_mask[:half_idx_1, :, :].to(module_device)
#                     attention_mask_2 = attention_mask[:half_idx_2, :, :].to(module_copy_device)
#                     # 按照 batch_size 分割 position_ids
#                     position_ids_1 = position_ids[:half_idx_1, :]
#                    # position_ids_1 = position_ids[:half_idx_1, :].to(module_device)
#                     position_ids_2 = position_ids[half_idx_2:, :].to(module_copy_device)
#                     # 并行执行两个 layers
#                     # 将 past_key_value 也分割成两部分
#                     if past_key_value is not None:
#                         print("past_key_value is not None\n")
#                         past_key_value_1 = (past_key_value[0][:half_idx_1, :, :, :], past_key_value[1][:half_idx_1, :, :, :])
#                         past_key_value_2 = (past_key_value[0][half_idx_2:, :, :, :], past_key_value[1][half_idx_2:, :, :, :])
#                     else:
#                         past_key_value_1 = None
#                         past_key_value_2 = None
                    
                    
#                     layer_outputs_1 = decoder_layer(
#                         hidden_states_1,
#                         attention_mask=attention_mask_1,  # 更新对应的 attention_mask
#                         position_ids=position_ids_1,  # 更新 position_ids
#                         past_key_value=past_key_value_1,
#                         output_attentions=output_attentions,
#                         use_cache=use_cache,
#                     )
#                     LlamaAttention.forward = create_att_forward(module_copy_device)

                 
                  
#                     layer_outputs_2 = module[idx](
#                         hidden_states_2,
#                         attention_mask=attention_mask_2,  # 更新对应的 attention_mask
#                         position_ids=position_ids_2,  # 更新 position_ids
#                         past_key_value=past_key_value_2,
#                         output_attentions=output_attentions,
#                         use_cache=use_cache,
#                     )
                    

#                     # # 从返回的元组中提取 hidden_states
#                     # hidden_states_1 = layer_outputs_1[0]  # 提取 layer_outputs_1 的第一个元素，即 hidden_states
#                     # hidden_states_2 = layer_outputs_2[0]  # 提取 layer_outputs_2 的第一个元素，即 hidden_states

#                     # # 然后对 hidden_states 进行拼接
#                     # layer_outputs = torch.cat([hidden_states_1, hidden_states_2], dim=0).to(module_device)

#                     # 将两个输出合并
#                     assert len(layer_outputs_1) == len(layer_outputs_2), "两个输出的长度不一致"

#                     # 对元组中的每个元素进行拼接
#                     # layer_outputs = tuple(
#                     #     torch.cat([move_to_device(elem1,module_device), move_to_device(elem2,module_device)], dim=0) 
#                     #     for elem1, elem2 in zip(layer_outputs_1, layer_outputs_2)
#                     # )
#                     layer_outputs = recursive_concat(layer_outputs_1, layer_outputs_2, module_device,dim=0)
#                     #layer_outputs = torch.cat([layer_outputs_1, layer_outputs_2], dim=1).to(module_device)
#                     LlamaAttention.forward = create_att_forward(module_device)
#                 else:
                    
#                     layer_outputs = decoder_layer(
#                         hidden_states,
#                         attention_mask=attention_mask,
#                         position_ids=position_ids,
#                         past_key_value=past_key_value,
#                         output_attentions=output_attentions,
#                         use_cache=use_cache,
#                     )
#             # print("model idx",idx)
#             # print('--'*50)
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


def create_decodelayer_forward(device):
    def my_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        hidden_states = hidden_states.to(device)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask.to(device),
            position_ids=position_ids.to(device),
            past_key_value=move_to_device(past_key_value,device),
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    return my_forward

def create_mtiHPA_copy_forward(module:dict,module_device,module_copy_device,half_idx = 3):
    def my_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        #LlamaAttention.forward = create_monitor_atten_ffn_forward()
        for idx, decoder_layer in enumerate(self.layers):
            #print("idx",idx)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                if idx in module:
                     # 假设第二维度表示序列长度，可以根据你的实际情况调整
                    def run_layer_1(decoder_layer,half_idx, hidden_states, attention_mask, position_ids, past_key_value):
                        hidden_states_1 = hidden_states[:half_idx, :, :].contiguous().to(module_device)  # 第一半
                        attention_mask_1 = attention_mask[:half_idx, :, :].to(module_device)
                        position_ids_1 = position_ids[:half_idx, :].to(module_device)
                        if past_key_value is not None:
                        #print("past_key_value is not None\n")
                            past_key_value_1 = (past_key_value[0][:half_idx, :, :, :], past_key_value[1][:half_idx, :, :, :])
                            #past_key_value_2 = (past_key_value[0][half_idx:, :, :, :], past_key_value[1][half_idx:, :, :, :])
                        else:
                            past_key_value_1 = None
                            #past_key_value_2 = None
                        past_key_value_1 = move_to_device(past_key_value_1,module_device)
                        forward1 = create_decodelayer_forward(module_device)
                        decoder_layer.forward = forward1.__get__(decoder_layer,LlamaDecoderLayer)
                        layer_outputs_1 = decoder_layer(
                            hidden_states_1,
                            attention_mask=attention_mask_1,  # 更新对应的 attention_mask
                            position_ids=position_ids_1,  # 更新 position_ids
                            past_key_value=past_key_value_1,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )
                        return layer_outputs_1
                    def run_layer_2(module,idx, half_idx,  hidden_states, attention_mask, position_ids, past_key_value):
                        hidden_states_2 = hidden_states[half_idx:, :, :].contiguous().to(module_copy_device)  # 第二半
                        attention_mask_2 = attention_mask[half_idx:, :, :].to(module_copy_device)
                        position_ids_2 = position_ids[half_idx:, :].to(module_copy_device)
                        if past_key_value is not None:
                            past_key_value_2 = (past_key_value[0][half_idx:, :, :, :], past_key_value[1][half_idx:, :, :, :])
                        else:
                            past_key_value_2 = None
                        past_key_value_2 = move_to_device(past_key_value_2,module_copy_device)
                        forward2 = create_decodelayer_forward(module_copy_device)
                        module[idx].forward = forward2.__get__(module[idx],LlamaDecoderLayer)
                        layer_outputs_2 = module[idx](
                            hidden_states_2,
                            attention_mask=attention_mask_2,  # 更新对应的 attention_mask
                            position_ids=position_ids_2,  # 更新 position_ids
                            past_key_value=past_key_value_2,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )
                        return layer_outputs_2
                    
                    from concurrent.futures import wait
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        # 记录开始的时间
                       
                        # 提交第一个任务
                        future1 = executor.submit(run_layer_1, decoder_layer, half_idx, hidden_states, attention_mask, position_ids, past_key_value)
                      

                        # 提交第二个任务
                      
                        future2 = executor.submit(run_layer_2, module, idx, half_idx, hidden_states, attention_mask, position_ids, past_key_value)
                      

                        # 获取第一个任务的结果，并记录结束时间
                        layer_outputs_1 = future1.result()
                        

                        # 获取第二个任务的结果，并记录结束时间
                        layer_outputs_2 = future2.result()
                        


                    # 打印每个任务的执行时间
                    #print(f"Thread 1 execution time (including waiting time for submission): {execution_time_1:.4f} seconds")
                    #print(f"Thread 2 execution time (including waiting time for submission): {execution_time_2:.4f} seconds")
                    #total_time = max(t_finish_1, t_finish_2) - min(t_start_1, t_start_2)
                    # print(f"Total time for both threads: {total_time:.4f} seconds")
                    # print("=="*50,"\n")
                    assert len(layer_outputs_1) == len(layer_outputs_2), "两个输出的长度不一致"
                    layer_outputs = recursive_concat(layer_outputs_1, layer_outputs_2, module_device,dim=0)
                    #LlamaAttention.forward = create_att_forward()
                else:
                    
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                   
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    return my_forward


def create_monitor_modules_forward():
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                start = time.time()
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                torch.cuda.synchronize()
                m_inference_time.append(time.time()-start)
                

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward



# 定义一个函数来测量显存使用
def measure_memory_usage(device):
    torch.cuda.synchronize()  # 确保所有操作都完成
    return torch.cuda.memory_allocated(device)


norm_time = []
atten_time = []
ffn_time = []

def create_monitor_atten_ffn_forward():
#def create_monitor_atten_ffn_forward(device_id,norm_time,norm_mem,atten_time,atten_mem,ffn_time,ffn_mem):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        start_time = time.time()
        #start_mem =  measure_memory_usage(device_id)
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.synchronize()
        norm_time.append(time.time()-start_time)
        #norm_mem.append(measure_memory_usage(device_id)-start_mem)
        # Self Attention
        start_time = time.time()
        #start_mem =  measure_memory_usage(device_id)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        torch.cuda.synchronize()
        atten_time.append(time.time()-start_time)
        #atten_mem.append(measure_memory_usage(device_id)-start_mem)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        start_time = time.time()
        #start_mem =  measure_memory_usage(device_id)
        hidden_states = self.mlp(hidden_states)
        torch.cuda.synchronize()
        ffn_time.append(time.time()-start_time)
        #ffn_mem.append(measure_memory_usage(device_id)-start_mem)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    return forward