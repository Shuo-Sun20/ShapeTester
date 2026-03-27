import torch
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
torch.manual_seed(42)
input_tensor = torch.randn(2, 3, 4)
src_tensor = torch.randn(2, 4)
dim = 1
index = 1

valid_test_case = {
    "inputs": [input_tensor, src_tensor],
    "dim": dim,
    "index": index
}

# 2 & 3 & 4. Parameters affecting output shape: dim and index
#   dim: int, constrained by input tensor rank (0 to input.dim()-1).
#        For a 3D input, values are 0, 1, 2.
#   index: int, constrained by input size along dim.
#        For example input shape (2,3,4) and dim=1, index can be 0, 1, 2.

@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [0, 1, 2])
    index: List[int] = field(default_factory=lambda: [0, 1, 2, 3])