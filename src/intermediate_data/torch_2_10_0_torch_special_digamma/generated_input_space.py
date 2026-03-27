import torch
from dataclasses import dataclass, field
from typing import Optional

def call_func(inputs, out=None):
    return torch.special.digamma(inputs, out=out)

# 1. Valid test case
valid_test_case = {
    "inputs": torch.rand(4, 3) * 2 + 0.5,
    "out": None
}

# 3. & 4. InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output shape with discretized value ranges"""
    # Only 'inputs' affects output shape, 'out' parameter doesn't change output shape
    # Since 'inputs' is excluded per instructions, this class has no fields
    pass