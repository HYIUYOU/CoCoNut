from py3nvml import py3nvml

# 初始化 NVML
py3nvml.nvmlInit()

# 获取 GPU 数量
device_count = py3nvml.nvmlDeviceGetCount()

# 遍历所有 GPU 获取其信息
for i in range(device_count):
    handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
    
    # 获取 GPU 名称
    name = py3nvml.nvmlDeviceGetName(handle)  # 直接使用返回的字符串
    
    # 获取当前频率 (图形处理频率)
    current_freq = py3nvml.nvmlDeviceGetClockInfo(handle, py3nvml.NVML_CLOCK_GRAPHICS) / 1000  # 单位 MHz
    
    # 获取最大频率
    max_freq = py3nvml.nvmlDeviceGetMaxClockInfo(handle, py3nvml.NVML_CLOCK_GRAPHICS) / 1000  # 单位 MHz
    
    # 获取 GPU 利用率
    utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle).gpu
    
    # 获取内存信息
    memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    memory_used = memory_info.used / (1024 ** 2)  # MB
    memory_total = memory_info.total / (1024 ** 2)  # MB
    
    # 打印每个 GPU 的信息
    print(f"GPU {i}: {name}")
    print(f"  Current Frequency: {current_freq} MHz")
    print(f"  Max Frequency: {max_freq} MHz")
    print(f"  GPU Utilization: {utilization}%")
    print(f"  Memory Used: {memory_used} MB / {memory_total} MB")
    print("-" * 50)
