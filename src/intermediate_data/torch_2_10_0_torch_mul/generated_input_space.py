import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional

# 1. valid_test_case definition
input_tensor1 = torch.randn(3, 4)
input_tensor2 = torch.randn(3, 4)
valid_test_case = {
    'inputs': [input_tensor1, input_tensor2],
    'other': None,
    'out': None
}

# 2 & 3 & 4. InputSpace dataclass
@dataclass
class InputSpace:
    """Parameters affecting output shape of torch.mul (excluding 'inputs')"""
    other: list = field(default_factory=lambda: [
        None,
        2.5,
        3+0j,
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    ])