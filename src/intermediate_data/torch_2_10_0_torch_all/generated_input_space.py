import torch
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

# Generate a random tensor as input
torch.manual_seed(0)
random_tensor = torch.randint(0, 2, (3, 4), dtype=torch.bool)

# Construct input as a list
inputs = [random_tensor]

valid_test_case = {
    "inputs": inputs,
    "dim": 1,
    "keepdim": False,
    "out": None
}

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape, with discretized value spaces"""
    
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = None
    keepdim: List[bool] = None
    
    def __post_init__(self):
        if self.dim is None:
            self.dim = [None, 0, 1, -1, (0, 1)]
        if self.keepdim is None:
            self.keepdim = [True, False]