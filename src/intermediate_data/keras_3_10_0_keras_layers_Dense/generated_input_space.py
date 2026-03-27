import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# 1. 定义valid_test_case变量
valid_test_case = {
    "inputs": np.random.randn(4, 8).astype(np.float32),
    "units": 16,
    "activation": "relu",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "lora_rank": None,
    "lora_alpha": None
}

# 2. 识别影响输出形状的参数（除了"inputs"）
# 根据文档，只有"units"参数直接影响输出形状

# 4. 定义InputSpace类
@dataclass
class InputSpace:
    units: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256])