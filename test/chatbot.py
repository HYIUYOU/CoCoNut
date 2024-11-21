class ChatbotBatch:
    def __init__(self, llama_instance):
        self.llama = llama_instance  # llama 模型实例
    
    def add_message(self, dialog, role, content):
        """
        添加消息到对话
        dialog: 当前对话的历史列表
        role: 'user' 或 'assistant'
        content: 消息的内容
        """
        dialog.append({"role": role, "content": content})
    
    def generate_batch_responses(self, dialog_batches):
        """
        为多个对话批量生成助手回复
        dialog_batches: List[List[Dict]] 对话历史的列表，其中每个对话历史是一个列表
        
        返回: 包含每个对话助手回复的列表
        """
        # 调用 chat_completion 方法生成批量回复
        responses = self.llama.chat_completion(
            dialogs=dialog_batches,  # 传递多个对话历史
            temperature=0.6,         # 生成的随机性控制
            top_p=0.9,               # 采样的概率控制
            max_gen_len=100          # 回复的最大长度
        )
        
        # 从每个生成结果中提取助手回复
        assistant_responses = [
            response["generation"]["content"] for response in responses
        ]
        
        return assistant_responses
    
    def chat_batch(self, user_inputs):
        """
        实现批量聊天：根据不同的用户输入生成对应的助手回复
        user_inputs: List[str] 批量用户输入
        
        返回: List[str] 批量生成的助手回复
        """
        dialog_batches = []  # 存储多个对话历史
        
        # 模拟每个对话历史
        for user_input in user_inputs:
            # 为每个用户输入创建一个新的对话历史
            dialog = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
            dialog_batches.append(dialog)
        
        # 调用生成批量回复的函数
        batch_responses = self.generate_batch_responses(dialog_batches)
        
        return batch_responses

from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig,AutoModelForCausalLM
import torch
import accelerate

# 加载模型和分词器
model_id = "../Models/Llama-2-13b-chat-hf"  # 替换为你自己的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 初始化 Accelerate 分布式加载和 offload
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

config = AutoConfig.from_pretrained(model_id)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
tokenizer.pad_token = tokenizer.eos_token
# 动态分配设备，设置 max_memory 来控制模型的显存使用
#device_map = infer_auto_device_map(model, max_memory={0: "40GiB"})
device_map = {"": "cuda:0"} 
print(device_map)


# 加载模型，并开启 offload 到 CPU
model = LlamaForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    offload_folder="offload",  # offload 到指定的文件夹
    torch_dtype=torch.bfloat16  # 使用 bfloat16 精度减少显存占用
)


# 设置为 eval 模式
model.eval() # 假设 Llama 类已实例化
chatbot_batch = ChatbotBatch(model)

# 假设有三个用户输入
user_inputs = [
    "Hello, how are you?",
    "Can you tell me about Llama2?",
    "What is the weather today?"
]

# 调用批量聊天方法
batch_responses = chatbot_batch.chat_batch(user_inputs)

# 打印每个对话的生成回复
for i, response in enumerate(batch_responses):
    print(f"Response {i+1}: {response}")
