import logging,copy,torch
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb,LlamaDecoderLayer


logger = logging.get_logger(__name__)

class COPY:
    @staticmethod
    def copy_module(module,device):
        """复制模型的指定模块, 默认复制到CPU
        Args:
            module (nn.Module): 待复制的模块
            device (str): 设备名称
        Returns:
            nn.Module: 复制的模块
        """
        if device == None:
            device = 'cpu'

        module_copy = copy.deepcopy(module)
        module_copy = module_copy.to(device)
        return module_copy
    @staticmethod
    def copy_module_with_id(module_id,device,model):
        """复制模型的指定模块, 默认复制到CPU
        Args:
            module_id (int): 待复制模块的索引
            device (str): 设备名称
            model (nn.Module): 模型
        Returns:
            nn.Module: 复制的模块
        """
        if device == None:
            device = 'cpu'
        module = model._modules[module_id]
        module_copy = copy.deepcopy(module)
        module_copy = module_copy.to(device)
        return module_copy

class U_ops:
    """
    通用操作类, 包含一些通用的操作方法
    universal_concat: 万能concat
    chunk_list: 将列表按照batch_size进行分组
    move_to_device: 万能将输入移动到指定设备
    """
    @staticmethod
    def universal_concat(input1, input2, device,dim=0):
        """万能concat: 拼接两个张量或元组
        Args:
            input1 (Union[torch.Tensor, Tuple[torch.Tensor]]): 输入1
            input2 (Union[torch.Tensor, Tuple[torch.Tensor]]): 输入2
            dim (int): 拼接维度
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: 拼接后的张量或元组
        """
        # 如果是元组，递归处理每个元素
        if isinstance(input1, tuple) and isinstance(input2, tuple):
            return tuple(U_ops.universal_concat(i1, i2,device, dim=dim) for i1, i2 in zip(input1, input2))
        # 如果是张量，直接拼接
        elif isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
            return torch.cat([input1.to(device), input2.to(device)], dim=dim)
        else:
            raise TypeError("Elements must be tensors or tuples of tensors")
    
    @staticmethod
    def chunk_list(lst, batch_size=10):
        """将列表按照batch_size进行分组
        Args:
            lst (List): 待分组的列表
            batch_size (int): 分组大小
        Returns:
            List: 分组后的列表
        """
    # 使用列表切片，将列表按照chunk_size进行分组
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    
    @staticmethod
    def move_to_device(inputs, device):
        """
        万能将输入移动到指定设备,适用于张量、元组和列表以及其他类型
        Args:
            inputs (Union[torch.Tensor, Tuple, List]): 输入
            device (torch.device): 目标设备
        Returns:
            Union[torch.Tensor, Tuple, List]: 移动到目标设备的输入
        """
        if isinstance(inputs, tuple):
            # 递归处理 tuple 中的每个元素
            return tuple(U_ops.move_to_device(tensor, device) for tensor in inputs)
        elif isinstance(inputs, list):
            # 递归处理 list 中的每个元素
            return [U_ops.move_to_device(tensor, device) for tensor in inputs]
        elif isinstance(inputs, torch.Tensor):
            # 如果是张量，则检查它是否已经在目标设备上
            if inputs.device != device:
                return inputs.to(device)  # 只有当张量不在目标设备上时，才移动
            return inputs  # 如果张量已经在目标设备上，直接返回
        else:
            # 其他类型保持不变 (比如 None)
            return inputs


#以下是进行HMA和Migration等操作时经常需要使用的代码，为简洁使用，所以将其放在了utils.py中

class create_forward:
    """
    创建自定义的forward方法
    注意，该类中只有静态方法，不需要实例化
    """
    @staticmethod
    def create_att_forward():
        """
        创建自定义的Attention的forward方法
        """
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
                    past_key_value = U_ops.move_to_device(past_key_value,device)
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
                    key_states = U_ops.move_to_device(key_states,device)
                    value_states = U_ops.move_to_device(value_states,device)
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

    @staticmethod
    def create_decodelayer_forward(device):
        """
        创建自定义的DecoderLayer的forward方法
        """
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
                past_key_value=U_ops.move_to_device(past_key_value,device),
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