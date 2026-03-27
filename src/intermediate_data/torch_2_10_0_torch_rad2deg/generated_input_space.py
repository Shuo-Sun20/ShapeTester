import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List

# Task 1: Define a valid test case
valid_test_case = {
    "inputs": torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]]),
    "out": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output shape (besides inputs) is 'out'
    out: List[Optional[Union[str, torch.Tensor]]] = field(
        default_factory=lambda: [None, "zeros", "ones", "random", "same"]
    )