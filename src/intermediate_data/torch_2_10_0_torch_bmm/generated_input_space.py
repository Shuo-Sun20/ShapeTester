import torch
from dataclasses import dataclass
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(10, 3, 4), torch.randn(10, 4, 5)],
    "out_dtype": None,
    "out": None
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the shape of the output tensor.
    Since only the 'inputs' parameter affects shape, and it's excluded per instructions,
    this class has no fields.
    """
    pass