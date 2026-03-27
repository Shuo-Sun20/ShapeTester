import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(dim, unflattened_size, inputs):
    unflatten_instance = nn.Unflatten(dim, unflattened_size)
    output = unflatten_instance(inputs)
    return output

# Valid test case
valid_test_case = {
    'dim': 1,
    'unflattened_size': (2, 5, 5),
    'inputs': torch.randn(2, 50)
}

# Parameters affecting output shape (excluding inputs)
shape_affecting_params = ['dim', 'unflattened_size']

@dataclass
class InputSpace:
    # Discrete parameter: dimension to unflatten
    dim: List[Union[int, str]] = field(default_factory=lambda: [
        # Integer dimensions (positive, negative, zero)
        0, 1, -1, -2,
        # String dimensions (for NamedTensor)
        "N", "C", "H", "W", "features"
    ])
    
    # Discrete parameter: unflattened shape specification
    unflattened_size: List[Union[Tuple[int, ...], 
                                  List[int], 
                                  torch.Size,
                                  Tuple[Tuple[str, int], ...]]] = field(default_factory=lambda: [
        # Tuple of ints
        (2, 5, 5),
        (5, 10),
        (1, 50),
        (50, 1),
        (2, 25),
        (25, 2),
        (10, 5),
        # List of ints
        [2, 5, 5],
        [5, 10],
        # torch.Size
        torch.Size([2, 5, 5]),
        torch.Size([5, 10]),
        # NamedShape (tuple of (name, size) tuples)
        (("C", 2), ("H", 5), ("W", 5)),
        (("H", 5), ("W", 10)),
        (("features", 50),),
    ])