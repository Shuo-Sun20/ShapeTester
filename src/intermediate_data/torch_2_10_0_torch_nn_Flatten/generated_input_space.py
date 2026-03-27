import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

def call_func(start_dim=1, end_dim=-1, inputs=None):
    flatten_layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    output = flatten_layer(inputs)
    return output

example_input = torch.randn(32, 1, 5, 5)
valid_test_case = {"start_dim": 0, "end_dim": 2, "inputs": example_input}

@dataclass
class InputSpace:
    start_dim: List[int] = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3])
    end_dim: List[int] = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3])