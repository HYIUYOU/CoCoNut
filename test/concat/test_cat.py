import torch
import time
import argparse

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Measure concat operation time")
parser.add_argument('--bsz', type=int, required=True, help="Batch size")  # 从命令行获取 batch size
parser.add_argument('--device', type=str, choices=['cpu', 'cuda:1'], default='cuda:1', help="Device to run the operation on")
args = parser.parse_args()

# 使用命令行输入的 batch size 和设备
bsz = args.bsz  # 从命令行获取的 batch size
device = args.device  # 获取设备 ('cpu' 或 'cuda')

# 设置设备
device = torch.device(device)

q_len = 312  # query sequence length
num_heads = 40  # number of attention heads
head_dim = 128  # head dimension
use_cache = True

# 假设过去的key_value缓存数据大小
past_key_value = (torch.randn(bsz, num_heads, q_len-1, head_dim).to(device), 
                  torch.randn(bsz, num_heads, q_len-1, head_dim).to(device))

# 开始测量 concat 时延


# 生成key_states和value_states
key_states = torch.randn(bsz, num_heads, 1, head_dim).to(device)
value_states = torch.randn(bsz, num_heads, 1, head_dim).to(device)


print(f"key values device: {key_states.device}, shape {key_states.shape},  key_states.element_size: { key_states.element_size()} \n")

start_time = time.time()
# 测量concat操作的时延
if past_key_value is not None:
    key_states = torch.stack([past_key_value[0], key_states], dim=2)
    value_states = torch.stack([past_key_value[1], value_states], dim=2)

# 输出时延
torch.cuda.synchronize()
concat_time = time.time() - start_time

print(f"past key device: {past_key_value[0].device}, shape {past_key_value[0].shape},  element_size: { past_key_value[0].element_size()} \n")

print(f"key values device: {key_states.device}, shape {key_states.shape},  key_states.element_size: { key_states.element_size()} \n")

print(f"{device}: {concat_time:.6f} seconds")
