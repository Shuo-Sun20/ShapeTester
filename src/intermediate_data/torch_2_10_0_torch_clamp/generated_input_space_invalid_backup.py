import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional

# 1. Valid test case
valid_test_case = {
    "inputs": torch.randn(4),
    "min": -0.5,
    "max": 0.5,
    "out": None
}

# 2. & 3. Parameters affecting output shape: min and max (only when they are tensors)
@dataclass
class InputSpace:
    # For min parameter
    min: List[Optional[Union[float, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,  # No lower bound
            -1.0,  # Scalar lower bound
            0.0,   # Scalar lower bound (zero)
            1.0,   # Scalar lower bound
            torch.tensor([0.0]),  # 1D tensor with 1 element
        ]
    )
    
    # For max parameter  
    max: List[Optional[Union[float, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,  # No upper bound
            -1.0,  # Scalar upper bound
            0.0,   # Scalar upper bound (zero)
            1.0,   # Scalar upper bound
            torch.tensor([0.0]),  # 1D tensor with 1 element
        ]
    )