import torch
def chunk_list(lst, batch_size=10):
    # 使用列表切片，将列表按照chunk_size进行分组
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
# 批量处理对话的函数
def batch_chat(model, tokenizer, prompts: list, history_dict: dict, max_new_tokens,device = 'cuda:0'):
    # 初始化带系统消息的对话
    system_message = "<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> "
    full_prompts = []
    
    # 为每个输入拼接对应的历史
    for i, prompt in enumerate(prompts):
        # 检查是否已有对话历史
        if i in history_dict:
            # 拼接已有的历史和新的输入
            full_prompt = history_dict[i] + f"[INST] {prompt} [/INST] "
        else:
            # 如果没有历史，初始化新的对话
            full_prompt = system_message + f"{prompt} [/INST] "
        # print("full prompt:", full_prompt,"\n")
        # print("==="*50,"\n")
        full_prompts.append(full_prompt)
    # print("full prompts:", full_prompts,"\n")
    # print("==="*50,"\n")
    # 将多个prompt处理为批量输入
    inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids,  max_new_tokens = max_new_tokens, do_sample=True,use_cache=True, temperature=0.7)

    # 解码生成的响应
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 更新每个输入的对话历史
    for i, response in enumerate(responses):
        # 提取助手的回答部分
        chatbot_response = response[len(full_prompts[i]):].strip()
        # print("chatbot_response:", chatbot_response,"\n")
        # print("==="*50,"\n")
        # 更新对应的对话历史
        if i in history_dict:
            history_dict[i] += f"[INST] {prompts[i]} [/INST] assistant: {chatbot_response} "
        else:
            history_dict[i] = system_message + f"{prompts[i]} [/INST] assistant: {chatbot_response} "
        # if i in history_dict:
        #     history_dict[i] += f"[INST] {prompts[i]} [/INST] assistant: {chatbot_response} "
        # else:
        #     history_dict[i] = f"[INST] {prompts[i]} [/INST] assistant: {chatbot_response} "
        # print("history:", history_dict,"\n")
        # print("==="*50,"\n")
    return responses, history_dict

import time

def chat(rps,model, tokenizer, prompts: list, history_dict: dict, batch_size=10, max_new_tokens=64, device='cuda:0'):
    responses = []
    inter = batch_size/rps
    batched_prompts = chunk_list(prompts,batch_size)
    # if inter > 1:
    #     # time.sleep(inter)
    for i in range(len(batched_prompts)):
        start = time.time()
        response, history = batch_chat(model,tokenizer,batched_prompts[i],history_dict,max_new_tokens,device)
        all = time.time() - start
        # if all < inter:
            # time.sleep(inter-all)
        responses.append(response)

    