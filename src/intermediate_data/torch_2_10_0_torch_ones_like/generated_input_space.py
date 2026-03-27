import torch
from dataclasses import dataclass
from typing import List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4, 5),
    "dtype": torch.float32,
    "layout": torch.strided,
    "device": torch.device("cpu"),
    "requires_grad": False,
    "memory_format": torch.preserve_format
}

# Tasks 2, 3, 4: Define InputSpace
@dataclass
class InputSpace:
    # None of the parameters except 'inputs' affect the shape of the output tensor
    pass