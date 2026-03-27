import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. 定义valid_test_case
batch_size = 2
height = 10
width = 10
channels = 3
input_tensor = tf.constant(np.random.randn(batch_size, height, width, channels).astype(np.float32))
valid_test_case = {
    'inputs': input_tensor,
    'pooling_ratio': [1.0, 1.44, 1.73, 1.0],
    'pseudo_random': False,
    'overlapping': False,
    'seed': 0,
    'name': None
}

# 2,3,4. 定义InputSpace类
@dataclass
class InputSpace:
    # pooling_ratio: 仅中间两个维度(height, width)影响输出形状
    # 典型值: [1.0, ratio_h, ratio_w, 1.0]
    pooling_ratio: List[List[float]] = field(default_factory=lambda: [
        [1.0, 1.0, 1.0, 1.0],      # 最小边界: 无池化
        [1.0, 1.2, 1.2, 1.0],      # 典型值1
        [1.0, 1.44, 1.73, 1.0],    # 示例值
        [1.0, 2.0, 2.0, 1.0],      # 典型值2
        [1.0, 3.0, 3.0, 1.0]       # 较大边界
    ])