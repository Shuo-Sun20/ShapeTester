import keras
from dataclasses import dataclass, field
from typing import Any, Union, Optional, List

valid_test_case = {
    "inputs": keras.random.uniform((3, 4)),
    "axis": 1
}

@dataclass
class InputSpace:
    axis: List[Optional[Union[int, tuple]]] = field(default_factory=lambda: [
        None,                  # 边界值: 无轴向
        0,                     # 边界值: 第一个维度
        -1,                    # 边界值: 最后一个维度
        1,                     # 典型值: 中间维度 (来自valid_test_case)
        (0, 1),                # 典型值: 多轴向元组
        (0,),                  # 典型值: 单元素元组
        (-2, -1),              # 典型值: 负索引元组
        2,                     # 边界值: 三维张量的第三维度
        (0, 2, 1)              # 典型值: 三轴元组
    ])