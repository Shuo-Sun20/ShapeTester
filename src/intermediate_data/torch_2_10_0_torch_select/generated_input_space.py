import torch
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case with all call_func parameters
valid_test_case = {
    "inputs": [torch.randn(3, 4, 5)],
    "dim": 1,
    "index": 2
}

# 2. Parameters affecting output shape (except "inputs"): dim and index

# 3-4. Define InputSpace dataclass with all shape-affecting parameters
@dataclass
class InputSpace:
    # dim parameter space: can be any integer from -3 to 2 (for 3D tensor)
    # Including positive, negative, and boundary values
    dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])
    
    # index parameter space: for a dimension of size 4 (as in valid_test_case)
    # Including positive, negative, and boundary values
    index: List[int] = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3])