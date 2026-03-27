import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'padding': (1, 2, 3, 4, 5, 6),
    'inputs': torch.randn(2, 3, 10, 20, 30)
}

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: [
            # 离散整数值
            0,  # 边界值：无填充
            1,  # 小正整数值
            3,  # 中等正整数值
            7,  # 较大正整数值
            10, # 典型大值
            
            # 六元素元组（6-tuple）类型
            (0, 0, 0, 0, 0, 0),  # 全零填充
            (1, 1, 1, 1, 1, 1),  # 均匀小值填充
            (1, 2, 3, 4, 5, 6),  # 递增非对称填充（包含valid_test_case中的值）
            (10, 0, 0, 10, 0, 0),  # 仅左右非零填充
            (0, 0, 5, 5, 0, 0),  # 仅上下非零填充
            (0, 0, 0, 0, 3, 7),  # 仅前后非零填充
            (2, 3, 2, 3, 2, 3),  # 对称对填充
            (10, 10, 10, 10, 10, 10),  # 均匀大值填充
        ]
    )