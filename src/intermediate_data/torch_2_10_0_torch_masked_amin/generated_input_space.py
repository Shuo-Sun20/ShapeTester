import torch
from dataclasses import dataclass
from typing import Union, Tuple, List

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'dim': 1,
    'keepdim': False,
    'dtype': None,
    'mask': torch.tensor([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False]
    ])
}

# 2. Parameters affecting output shape (excluding "inputs"):
# - dim: affects which dimensions are reduced
# - keepdim: affects whether reduced dimensions are kept or squeezed

# 3. Parameter value space analysis:

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape and their value ranges."""
    
    # dim: can be None, int, or tuple of ints
    # For 2D input (3x4), valid dim values are: None, 0, 1, -1, -2, (0,1), (0,), (1,)
    # Also include negative tuples and mixed positive/negative tuples
    dim: List[Union[int, Tuple[int, ...], None]] = None
    
    # keepdim: boolean parameter
    keepdim: List[bool] = None
    
    def __post_init__(self):
        if self.dim is None:
            # Discretized value space for dim parameter
            self.dim = [
                None,           # reduce all dimensions
                0,              # reduce along rows
                1,              # reduce along columns (example case)
                -1,             # negative indexing
                -2,             # negative indexing for 2D
                (0, 1),         # reduce both dimensions
                (1, 0),         # same dimensions, different order
                (0,),           # single element tuple
                (1,),           # single element tuple
                (-1, -2),       # negative tuple
                (0, -1),        # mixed positive/negative
                (-2, 1)         # mixed negative/positive
            ]
        
        if self.keepdim is None:
            # Only two possible values for boolean parameter
            self.keepdim = [True, False]

# Create instance to verify it can be instantiated
var = InputSpace()