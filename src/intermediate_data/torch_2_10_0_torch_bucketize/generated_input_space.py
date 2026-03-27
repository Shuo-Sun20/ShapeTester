from dataclasses import dataclass, field
from typing import Optional
import torch

def call_func(inputs, out_int32=False, right=False, out=None):
    input_tensor, boundaries = inputs
    return torch.bucketize(input_tensor, boundaries, out_int32=out_int32, right=right, out=out)

# Task 1: Define a valid test case
valid_test_case = {
    'inputs': (torch.tensor([[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]]), torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])),
    'out_int32': False,
    'right': False,
    'out': None
}

# Task 2: Identify parameters that affect output shape (excluding "inputs")
# Parameters: out_int32, right, out. However, only "out" can directly affect shape if provided.

# Task 3 & 4: Define InputSpace with discretized value spaces
@dataclass
class InputSpace:
    # out: can be None or a tensor matching input shape. Since it must match input shape, we can't predefine a fixed list.
    # We'll use None or a placeholder (e.g., "same") to represent output tensor creation.
    out: Optional[str] = field(default_factory=lambda: [None, "same"])
    
    # out_int32: boolean parameter, discrete
    out_int32: list[bool] = field(default_factory=lambda: [False, True])
    
    # right: boolean parameter, discrete  
    right: list[bool] = field(default_factory=lambda: [False, True])