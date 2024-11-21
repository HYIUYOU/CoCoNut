from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from torch import nn
import math
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from typing import List, Optional, Tuple, Union
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
logger = logging.get_logger(__name__)


def recursive_concat(output1, output2, device,dim=0):
    # 如果是元组，递归处理每个元素
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        return tuple(recursive_concat(o1, o2,device, dim=dim) for o1, o2 in zip(output1, output2))
    # 如果是张量，直接拼接
    elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        return torch.cat([output1.to(device), output2.to(device)], dim=dim)
    else:
        raise TypeError("Elements must be tensors or tuples of tensors")

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


def create_att_forward(device):
    def att_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                past_key_value = move_to_device(past_key_value,device)
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            # [bsz, nh, t, hd]

            if past_key_value is not None:
                # reuse k, v, self_attention
                
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                print("past_key_value[0].shape:", past_key_value[0].shape)
                print("key_states.shape:", key_states.shape)
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
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
    return att_forward

def VPA_HPA(module_id,module_device,module_copy,module_copy_device,VPA_id,VPA_device):
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
                    attention_mask_1 = attention_mask[:half_idx_1, :, :].to(module_device)
                    attention_mask_2 = attention_mask[:half_idx_2, :, :].to(module_copy_device)
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
                    LlamaAttention.forward = create_att_forward(module_device)
                    
                    #layer_outputs = torch.cat([layer_outputs_1, layer_outputs_2], dim=1).to(module_device)
                else:
                    #LlamaAttention.forward = create_att_forward(module_device)
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    if idx == VPA_id-1:
                        LlamaAttention.forward = create_att_forward(VPA_device)
                        layer_outputs = move_to_device(layer_outputs,VPA_device)
                        position_ids = position_ids.to(VPA_device)
                        past_key_value = move_to_device(past_key_value,VPA_device)
                        #output_attentions = output_attentions.to(device2) if output_attentions is not None else None
                    if idx == VPA_id:
                        LlamaAttention.forward = create_att_forward(module_device)
                        layer_outputs = move_to_device(layer_outputs,module_device)
                        position_ids = position_ids.to(module_device)
                        past_key_value = move_to_device(past_key_value,module_device)
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



# 加载模型和tokenizer
module_id = 4
model = LlamaForCausalLM.from_pretrained("../Models/Llama-2-7b-chat-hf").half().to(device1)
tokenizer = AutoTokenizer.from_pretrained("../Models/Llama-2-7b-chat-hf")
model.model.layers[20] = model.model.layers[20].to(device2)
import copy
module_copy = copy.deepcopy(model.model.layers[module_id])
module_copy = module_copy.to(device3)
tokenizer.pad_token = tokenizer.eos_token

LlamaModel.forward = VPA_HPA(module_id,device1,module_copy,device3,20,device2)



# 提供prompt并将输入移到GPU上
prompts = [
    "Hey, are you conscious? Can you talk to me?",
    "What is the meaning of life?",
    "Tell me a joke.",
    "Explain quantum mechanics in simple terms."
]
# truncation=True：确保输入序列不会超过模型的最大长度，超长序列会被截断。
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device1)  # 将输入转移到GPU

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)

# 解码并输出生成结果
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# 打印所有结果
for i, output in enumerate(outputs):
    print(f"Prompt {i + 1}: {prompts[i]}")
    print(f"Response {i + 1}: {output}")
    print('-' * 30)