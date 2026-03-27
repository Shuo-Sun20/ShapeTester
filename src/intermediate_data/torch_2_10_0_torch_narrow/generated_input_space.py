import torch
from typing import List, Union
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4, 5)],
    "dim": 1,
    "start": 2,
    "length": 2
}

# Task 3 & 4: Define InputSpace with parameter value spaces
@dataclass
class InputSpace:
    # dim: int, discrete parameter affecting output shape
    # Value space: dimension indices for a 3D tensor (example input shape)
    dim: List[int] = field(default_factory=lambda: [0, 1, 2])
    # start: int or Tensor, discrete parameter affecting output shape
    # We choose 5 values that are valid for the example input in dimension 1 (size 4)
    start: List[Union[int, torch.Tensor]] = field(default_factory=lambda: [-4, -2, 0, 2, 3])
    # length: int, discrete parameter affecting output shape
    # We choose all possible lengths for the example input in dimension 1 (size 4)
    length: List[int] = field(default_factory=lambda: [1, 2, 3, 4])