import torch
from dataclasses import dataclass
from typing import List

def call_func(inputs):
    return torch.relu_(inputs)

valid_test_case = {
    'inputs': torch.randn(3, 4)
}

@dataclass
class InputSpace:
    pass