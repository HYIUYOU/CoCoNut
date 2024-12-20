from coconut.utils import *
from typing import Optional, Tuple, Union, List
from concurrent.futures import ThreadPoolExecutor
from transformers.modeling_outputs import BaseModelOutputWithPast

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
                        past_key_value_1 = U_ops.move_to_device(past_key_value_1,module_device)
                        forward1 = create_forward.create_decodelayer_forward(module_device)
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
                        past_key_value_2 = U_ops.move_to_device(past_key_value_2,module_copy_device)
                        forward2 = create_forward.create_decodelayer_forward(module_copy_device)
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
                    layer_outputs = U_ops.universal_concat(layer_outputs_1, layer_outputs_2, module_device,dim=0)
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

