import torch
from dataclasses import dataclass, field
from typing import Union, List

# 1. valid_test_case
valid_test_case = {
    'inputs': torch.tensor([-3., -2, -1, 1, 2, 3]),
    'other': torch.tensor([2.0]),
}

# 3. and 4. InputSpace with all parameters affecting output shape (except inputs)
@dataclass
class InputSpace:
    # other: shape-affecting parameter
    # Discretized value space covering:
    # - Boundary values: scalar 0, very small positive/negative, large positive/negative
    # - Tensor shapes: scalar, 1D, 2D with various dimensions that broadcast
    # - Mixed types: float, int, negative, positive
    other: List[Union[torch.Tensor, float, int]] = field(default_factory=lambda: [
        # Scalars
        0.0,            # Boundary: zero
        0.0001,         # Very small positive
        -0.0001,        # Very small negative
        1.0,            # Unit positive
        -1.0,           # Unit negative
        1000.0,         # Large positive
        -1000.0,        # Large negative
        2,              # Integer
        -3,             # Negative integer
        
        # 1D tensors
        torch.tensor([0.5]),                          # Single element
        torch.tensor([1.0, 2.0, 3.0]),                # Multiple elements
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),    # Mixed signs
        
        # 2D tensors (broadcastable with typical inputs)
        torch.tensor([[2.0]]),                        # 1x1 tensor
        torch.tensor([[1.0, 2.0, 3.0]]),              # 1x3 row
        torch.tensor([[1.0], [2.0], [3.0]]),          # 3x1 column
        
        # Value from valid_test_case
        torch.tensor([2.0]),
    ])