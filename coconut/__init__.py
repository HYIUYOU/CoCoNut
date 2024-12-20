# coconut/__init__.py
from .version import __version__

# 导入主要的监控类
from .monitor.GPU_monitor import GPUMonitor
from .monitor.module_monitor import layer_monitor,layer_time,pre_norm_time,post_norm_time,atten_time,ffn_time,atten_ffn_norm_monitor

# 导入操作相关的类
from .ops.HMA import HMA
from .ops.Migration import Migration

# 导入其他的类
from .utils import *

# 设置默认导出
__all__ = [
    '__version__',
    'GPUMonitor',
    'layer_monitor',
    'layer_time',
    'pre_norm_time',
    'post_norm_time',
    'atten_time,ffn_time',
    'atten_ffn_norm_monitor',
    'HMA',
    'Migration',
    'utils',
    'COPY',
    'U_ops',
    'create_forward',
    'logger',
    'torch',
    'LlamaDecoderLayer'
]

# 包的基本信息
__author__ = "HBigo"
__email__ = "hbigopk@gmail.com"
__description__ = "CoCoNut is a drinking buddy for your deep learning models."

# 可以添加一些包级别的初始化代码
def setup_logging():
    """设置日志配置"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# 包导入时自动执行的初始化
setup_logging()
