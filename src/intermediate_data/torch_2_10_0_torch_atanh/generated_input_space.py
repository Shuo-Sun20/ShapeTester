import torch
from dataclasses import dataclass
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.rand(4) * 2 - 1,  # Values in range (-1, 1)
    "out": None  # Optional parameter
}

# 4. Define InputSpace dataclass with all parameters affecting output shape
@dataclass
class InputSpace:
    # Only parameter affecting output shape (other than inputs) is "out"
    # Value space for out parameter
    out: List[Optional[Union[torch.Tensor, str]]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Discretized value space for "out" parameter
            # - None: No output tensor provided
            # - torch.Tensor: Valid tensor with matching shape (represented by "same")
            # - torch.Tensor: Invalid tensor with different shape (represented by "different")
            self.out = [None, "same", "different", torch.zeros(4), torch.zeros(3)]