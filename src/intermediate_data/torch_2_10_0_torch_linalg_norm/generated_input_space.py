import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4)],
    "ord": 'fro',
    "dim": None,
    "keepdim": False,
    "out": None,
    "dtype": None
}

# 2 & 3 & 4. Define InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of torch.linalg.norm,
    with their discretized value spaces.
    """
    # dim parameter space: discrete values covering all legal scenarios
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(default_factory=lambda: [
        None,  # flatten and compute norm
        0,     # vector norm along first dimension
        1,     # vector norm along second dimension
        -1,    # vector norm along last dimension
        (0, 1),  # matrix norm over two dimensions
        (0, 2),  # matrix norm over non-consecutive dimensions
        (0, -1), # matrix norm with negative index
    ])
    
    # keepdim parameter space: boolean values
    keepdim: List[bool] = field(default_factory=lambda: [
        False,  # reduce dimensions
        True,   # keep dimensions with size 1
    ])