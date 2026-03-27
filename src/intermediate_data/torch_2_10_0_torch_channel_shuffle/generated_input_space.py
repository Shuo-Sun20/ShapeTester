import torch
from dataclasses import dataclass, field

def call_func(inputs, groups):
    return torch.channel_shuffle(inputs, groups)

valid_test_case = {
    "inputs": torch.randn(1, 4, 2, 2),
    "groups": 2
}

@dataclass
class InputSpace:
    groups: list = field(default_factory=lambda: [1, 2, 4, 8, 16])