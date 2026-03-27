import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional
import itertools

# 1. Define valid_test_case
valid_test_case = {
    'kernel_size': 3,
    'output_size': (13, 12, 11),
    'output_ratio': None,
    'return_indices': False,
    'inputs': torch.randn(20, 16, 50, 32, 16)
}

# 2. Parameters affecting output shape (excluding "inputs"): kernel_size, output_size, output_ratio
# 3-4. Define InputSpace dataclass with discretized value ranges

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape with discretized value ranges"""
    
    # kernel_size: int or tuple of 3 ints, must be > 0
    # Discrete values: boundary and typical values
    kernel_size: List[Union[int, Tuple[int, int, int]]] = None
    
    # output_size: None, int, or tuple of 3 ints
    # Must be positive and <= input size
    output_size: List[Optional[Union[int, Tuple[int, int, int]]]] = None
    
    # output_ratio: None, float, or tuple of 3 floats
    # Must be in range (0, 1)
    output_ratio: List[Optional[Union[float, Tuple[float, float, float]]]] = None
    
    def __post_init__(self):
        if self.kernel_size is None:
            # Discrete values for kernel_size
            # Single int values and tuple combinations
            single_vals = [1, 2, 3, 4, 5]  # boundary and typical values
            tuple_combs = list(itertools.product([1, 2, 3], repeat=3))  # typical 3D combinations
            self.kernel_size = single_vals + tuple_combs[:5]  # Limit to 5 combinations
            
        if self.output_size is None:
            # Discrete values for output_size
            # Include None (when using output_ratio), single int, and tuples
            # Typical values based on common use cases
            self.output_size = [
                None,  # for output_ratio cases
                10,    # single int
                (13, 12, 11),  # from valid_test_case
                (5, 5, 5),     # small square
                (25, 16, 8),   # different dimensions
                (50, 32, 16)   # same as input (with kernel_size=1)
            ]
            
        if self.output_ratio is None:
            # Discrete values for output_ratio
            # Include None (when using output_size), single float, and tuples
            # Boundary values (avoiding 0 and 1) and typical values
            self.output_ratio = [
                None,  # for output_size cases
                0.1,   # small ratio near 0
                0.25,  # quarter size
                0.5,   # half size (common)
                0.75,  # three-quarters
                0.9,   # near 1
                (0.5, 0.5, 0.5),    # uniform scaling
                (0.25, 0.5, 0.75),  # different scaling per dimension
                (0.8, 0.6, 0.4)     # another 3D combination
            ]

# Example instantiation
if __name__ == "__main__":
    input_space = InputSpace()
    print("Kernel size samples:", input_space.kernel_size[:10])
    print("Output size samples:", input_space.output_size)
    print("Output ratio samples:", input_space.output_ratio)