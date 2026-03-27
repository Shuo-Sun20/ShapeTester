import torch
from dataclasses import dataclass

valid_test_case = {
    "inputs": [torch.randn(4), torch.randn(4)],
    "p": 2
}

@dataclass
class InputSpace:
    pass