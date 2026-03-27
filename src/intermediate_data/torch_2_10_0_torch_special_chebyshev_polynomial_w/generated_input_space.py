import torch
from dataclasses import dataclass
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'n': torch.tensor(2, dtype=torch.int32)
}

# 2. Identify parameters affecting output shape (excluding "inputs")
#    Only "n" affects the output shape through broadcasting rules

# 3-4. Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output tensor shape"""
    
    # n can be scalar or tensor with broadcastable dimensions
    n: List[Union[torch.Tensor, int]] = None
    
    def __post_init__(self):
        if self.n is None:
            # Discretized value space for n parameter
            # Including boundary values and 5+ typical values covering all legal scenarios
            self.n = [
                # Scalar integer tensors (boundary and typical values)
                torch.tensor(0, dtype=torch.int32),   # Min valid degree
                torch.tensor(1, dtype=torch.int32),   # Typical small value
                torch.tensor(2, dtype=torch.int32),   # From valid_test_case
                torch.tensor(3, dtype=torch.int32),   # Typical medium value
                torch.tensor(5, dtype=torch.int32),   # Typical medium value
                torch.tensor(10, dtype=torch.int32),  # Typical larger value
                
                # 1D tensors (various lengths for broadcasting)
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                
                # 2D tensors (for 2D broadcasting)
                torch.tensor([[1], [2], [3]], dtype=torch.int32),
                torch.tensor([[1, 2, 3]], dtype=torch.int32),
                
                # Special cases
                torch.tensor([], dtype=torch.int32),  # Empty tensor
                torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32),  # 3D tensor
            ]

# The InputSpace can be successfully instantiated
var = InputSpace()