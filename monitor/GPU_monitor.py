import pynvml,torch
from typing import List, Optional
from dataclasses import dataclass, asdict

@dataclass
class GPUInfo:
    """GPU信息类
    
    Attributes:
        gpu_id (int): GPU ID
        memory_total (float): 总显存大小 (MB)
        memory_used (float): 已使用显存大小 (MB)
        memory_free (float): 剩余显存大小 (MB)
        gpu_util (int): GPU使用率 (%)
        memory_util (int): 显存使用率 (%)
        device (torch.device): GPU设备(torch.device)
    """
    gpu_id: int
    memory_total: float
    memory_used: float
    memory_free: float
    gpu_util: int
    memory_util: int
    device:torch.device 

    def __str__(self) -> str:
        return (f"GPU {self.gpu_id}: "
                f"Memory Total: {self.memory_total:.2f} MB, "
                f"Memory Used: {self.memory_used:.2f} MB, "
                f"Memory Free: {self.memory_free:.2f} MB, "
                f"GPU Util: {self.gpu_util}%, "
                f"Memory Util: {self.memory_util}%,"
                f"Device: {self.device}")

    def to_dict(self) -> dict:
        return asdict(self)
        
class GPUMonitor:
    """GPU监控类
    example:
        try:
            monitor = GPUMonitor()
            gpus = monitor.get_gpu_info()  
            print(gpus[1].gpu_id)
        except Exception as e:
            print(f"Error: {e}")
    """
    
    @staticmethod
    def get_gpu_info() -> List[Optional[GPUInfo]]:
        """获取所有GPU的详细信息
        
        Returns:
            List[Optional[GPUInfo]]: 包含每个GPU信息的列表
                - 使用 Optional 表示可能返回 None 的情况
            
        Raises:
            RuntimeError: 当无法初始化NVML时
        """
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")

        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info = []

            for i in range(gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    gpu_data = GPUInfo(
                        gpu_id=i,
                        memory_total=mem_info.total / 1024**2,
                        memory_used=mem_info.used / 1024**2,
                        memory_free=mem_info.free / 1024**2,
                        gpu_util=util.gpu,
                        memory_util=util.memory,
                        device=torch.device(f"cuda:{i}")
                    )
                    gpu_info.append(gpu_data)
                except Exception as e:
                    print(f"Error getting info for GPU {i}: {e}")
                    gpu_info.append(None)

            return gpu_info
        finally:
            pynvml.nvmlShutdown()

# # # 使用示例
if __name__ == "__main__":

    try:
        monitor = GPUMonitor()
        gpus = monitor.get_gpu_info()  
        print(gpus[1].device)
    except Exception as e:
        print(f"Error: {e}")