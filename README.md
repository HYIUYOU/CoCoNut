# CoCoNut
**CoCoNut is a drinking buddy for your deep learning models.**
----------------
## Quick Start

### Installation
```bash
$ git clone https://github.com/HYIUYOU/CoCoNut.git
$ cd CoCoNut
$ pip install -e .
```

### Usage
```python
# deploy a model
from transformers import AutoToenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("llama")
model = LlamaForCausalLM.from_pretrained("llama").to(device)

device1 = torch.device("cuda:1")
```

#### HMA
```python
from coconut.ops import HMA
from coconut.utils import COPY
module_copy_1 = COPY.copy_module_with_id(4,device1,model)
module_copy_2 = COPY.copy_module_with_id(8,device1,model)
module_copy_3 = COPY.copy_module_with_id(14,device1,model)
module = {
    4:module_copy_1,
    8:module_copy_2,
    14:module_copy_3
}

HMA(module,device0,device1,half_idx = 3)
```

#### Migration
```python
# Migration
from coconut.ops import Migration
Migration(model, 1, 'cuda:0', 'cuda:1')
```

#### Monitor

##### 1. GPU Monitor
```python
# gpu monitor
from coconut.monitor import GPU_monitor
try:
    monitor = GPUMonitor()
    gpus = monitor.get_gpu_info()  
    print(gpus[1].gpu_id)
except Exception as e:
    print(f"Error: {e}")
```

##### 2. module Monitor
```python
from coconut.monitor import module_monitor

# get layer time
layer_id = [0,1,2,5,8]
layer_time = module_monitor.layer_time(model,layer_id)

print(layer_time)

# get attention  norm ffn time

pre_norm_time,post_norm_time,atten_time,ffn_time = atten_ffn_norm_monitor()
```


