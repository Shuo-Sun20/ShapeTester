import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

# Define the variable valid_test_case
valid_test_case = {
    "inputs": torch.tensor([[torch.nan, 1.0, 2.0], [1.0, 2.0, 3.0]]),
    "dim": 0,
    "keepdim": False,
    "dtype": None,
    "out": None
}

# Define the InputSpace dataclass
@dataclass
class InputSpace:
    """
    Class containing all parameters that affect the output shape of torch.nanmean,
    with discretized value ranges for parameter testing.
    """
    # Parameters that affect output shape:
    # 1. dim: Can be None, single int, or tuple of ints
    #    For a 2D tensor (2,3) example, possible dim values:
    dim: list = field(default_factory=lambda: [
        None,        # Reduce all dimensions
        0,           # Reduce along first dimension
        1,           # Reduce along second dimension
        (0, 1),      # Reduce along both dimensions
        -1           # Reduce along last dimension
    ])
    
    # 2. keepdim: Boolean parameter
    keepdim: list = field(default_factory=lambda: [
        True,
        False
    ])