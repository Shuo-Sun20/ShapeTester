import torch
import torch.nn as nn
from dataclasses import dataclass

def call_func(inputs, p=0.5, inplace=False):
    dropout = nn.Dropout3d(p=p, inplace=inplace)
    return dropout(inputs)

example_input = torch.randn(20, 16, 4, 32, 32)
valid_test_case = {
    "inputs": example_input,
    "p": 0.2,
    "inplace": False
}

@dataclass
class InputSpace:
    # No parameters in call_func (except "inputs") affect the output shape
    pass