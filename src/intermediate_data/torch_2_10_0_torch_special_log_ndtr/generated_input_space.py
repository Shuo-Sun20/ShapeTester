import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np

# 1. Valid test case
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'out': None
}

# 2. Parameters that affect output shape: Only 'out' parameter (if provided, must match input shape)

# 3. Value space analysis for parameters affecting shape:

@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output tensor shape for log_ndtr"""
    
    # Only 'out' parameter affects output shape (besides 'inputs')
    out: List[Optional[Union[torch.Tensor, None]]] = field(default_factory=lambda: [
        # Discrete values for 'out' parameter:
        None,  # Default value - output shape determined by input
        torch.tensor([]),  # Empty tensor (edge case)
        torch.tensor([0.0]),  # Scalar as 1D tensor
        torch.tensor([[-1.0, 2.0], [0.5, -0.5]]),  # 2D tensor (2, 2)
        torch.randn(3, 4),  # Same shape as valid_test_case
        torch.randn(1, 1, 1),  # 3D singleton tensor
        torch.randn(2, 3, 4),  # 3D tensor
        torch.randn(5, 6, 7, 8),  # 4D tensor (higher dimension)
        torch.full((2, 3), float('nan')),  # Tensor with NaN values
        torch.full((2, 3), float('inf')),  # Tensor with inf values
        torch.full((2, 3), float('-inf')),  # Tensor with -inf values
    ])

# Example instantiation
var = InputSpace()