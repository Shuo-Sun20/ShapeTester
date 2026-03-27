import torch
from dataclasses import dataclass, field
from typing import Union, List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4),  # Input tensor of shape (3, 4)
    "repeats": torch.tensor([1, 2, 3]),  # Fixed repeats tensor matching the sum for output_size
    "dim": 0,  # Dimension along which to repeat
    "output_size": 6  # Output size hint to avoid meta tensor issue
}

# Task 2 & 3 & 4: Identify parameters affecting output shape and their value spaces
@dataclass
class InputSpace:
    """
    Dataclass containing parameters that affect output tensor shape in call_func(),
    with discretized value ranges for each parameter.
    """
    # repeats: can be integer or 1D tensor; determines repetition counts
    repeats: List[Union[int, torch.Tensor]] = field(
        default_factory=lambda: [
            1,  # Single integer for uniform repetition
            2,  # Another integer
            torch.tensor([1]),  # 1D tensor with single element
            torch.tensor([1, 2]),  # 1D tensor with 2 elements
            torch.tensor([1, 1, 1]),  # 1D tensor with 3 elements
        ]
    )
    
    # dim: dimension along which to repeat values
    dim: List[Optional[int]] = field(
        default_factory=lambda: [
            None,  # Flatten input array
            0,  # First dimension
            1,  # Second dimension
            -1,  # Last dimension
            -2,  # Second last dimension
        ]
    )