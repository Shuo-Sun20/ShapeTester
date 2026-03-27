import torch
from dataclasses import dataclass, field

# 1. Valid test case
valid_test_case = {
    "inputs": [torch.randn(2, 3)],
    "dim0": 0,
    "dim1": 1
}

# 2. Parameters affecting output shape (except "inputs"): dim0, dim1
# 3. Value space analysis:
#    - dim0/dim1: int values representing tensor dimensions
#    - For a tensor with rank n, valid range is [-n, n-1]
#    - Discretization for typical 2D/3D/4D tensors (covering common cases)

@dataclass
class InputSpace:
    dim0: list = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3, 4])
    dim1: list = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3, 4])