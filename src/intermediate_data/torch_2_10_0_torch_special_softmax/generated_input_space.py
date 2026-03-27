import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(2, 3, 4),
    'dim': 1,
    'dtype': None
}

# 2. Parameters affecting output shape: Only 'inputs' affects shape.
#    'dim' only changes computation dimension but not output shape.
#    'dtype' only changes data type but not shape.
#    However, we need to include all call_func parameters except 'inputs' as per requirements.
#    So we include 'dim' and 'dtype' even though they don't affect shape.

@dataclass
class InputSpace:
    """Class containing discretized parameter spaces for torch.special.softmax"""
    
    # Parameter: dim
    # Type: int (discrete)
    # Value space: Includes boundary and typical values for a 3D tensor example
    dim: List[int] = field(default_factory=lambda: [
        -3, -2, -1, 0, 1, 2  # All possible dims for 3D tensor
    ])
    
    # Parameter: dtype
    # Type: torch.dtype or None (discrete)
    # Value space: Common float types and None
    dtype: List[Optional[torch.dtype]] = field(default_factory=lambda: [
        None,  # Default behavior
        torch.float16,  # Half precision
        torch.bfloat16,  # Brain floating point
        torch.float32,  # Single precision
        torch.float64,  # Double precision
        torch.complex64,  # Complex single precision
        torch.complex128  # Complex double precision
    ])

# Ensure InputSpace can be instantiated successfully
var = InputSpace()