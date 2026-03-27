import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(3, 3, dtype=torch.float64),
    'out': None
}

# 2. & 3. Identify parameters affecting output shape (except "inputs") and construct value space
# Only parameter affecting output shape (besides inputs) is "out"

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the shape of the output tensor
    for torch.special.bessel_y0, except for 'inputs'.
    """
    # The 'out' parameter can be None or a Tensor
    # Value space includes None and various tensor types/shapes
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # Default case - output shape matches input
        torch.empty(3, 3, dtype=torch.float64),  # Same shape/dtype as example
        torch.empty(3, 3, dtype=torch.float32),  # Same shape, different dtype
        torch.empty(1, 3, dtype=torch.float64),  # Different shape (broadcastable)
        torch.empty(3, 1, dtype=torch.float64),  # Different shape (broadcastable)
        torch.empty(2, 3, 3, dtype=torch.float64),  # Larger shape
        torch.empty(1, 1, dtype=torch.float64),  # Smaller shape (broadcastable)
    ])

# Note: The above value space for 'out' includes:
# 1. None (default case)
# 2. Tensor with same shape and dtype as example input
# 3. Tensor with same shape but different dtype
# 4-7. Tensors with various broadcastable/non-broadcastable shapes
# All cases ensure legal scenarios are covered