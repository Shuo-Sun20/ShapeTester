import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(2, 3)],  # Input tensor
    'dim': 1,  # Dimension to reduce
    'keepdim': False,  # Whether to keep reduced dimension
    'dtype': None,  # Data type (None means use input's dtype)
    'mask': torch.tensor([[True, False, True], [False, False, False]])  # Mask tensor
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape (except 'inputs')
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,  # Reduce all dimensions
            0, 1, -1, -2,  # Single dimensions (positive and negative)
            (0, 1), (1, 0),  # Multiple dimensions in tuple
            (0,), (-1,),  # Single dimension in tuple
            ()  # Empty tuple (no reduction)
        ]
    )
    keepdim: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    # Note: dtype and mask don't affect output shape, so not included

# Example instantiation
var = InputSpace()