import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4)],  # Input tensor must be a list containing one tensor
    "out": None  # Can be None or a tensor of the same shape as input
}

# Task 3-4: Define InputSpace with discretized value ranges
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect output tensor shape (excluding 'inputs').
    Only 'out' parameter affects output shape in this case.
    """
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # Case 1: No output tensor provided
        torch.tensor([]),  # Case 2: Empty tensor (scalar output)
        torch.tensor([1.0]),  # Case 3: 1D tensor with 1 element
        torch.tensor([1.0, 2.0, 3.0]),  # Case 4: 1D tensor with 3 elements
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Case 5: 2x2 tensor
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),  # Case 6: 1x2x2 tensor
        torch.zeros(2, 3, 4),  # Case 7: 2x3x4 tensor (zeros)
        torch.ones(2, 3, 4),  # Case 8: 2x3x4 tensor (ones)
        torch.full((2, 3, 4), 5.0),  # Case 9: 2x3x4 tensor (filled with 5.0)
        torch.randn(2, 3, 4),  # Case 10: 2x3x4 tensor (random normal)
        torch.tensor(3.14),  # Case 11: Scalar tensor
    ])