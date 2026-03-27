import torch
from dataclasses import dataclass, field
from typing import Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 3), torch.randn(3)],
    "left": True,
    "out": None
}

# 2. Parameters affecting output shape (except "inputs"): left, out
#    Note: 'out' must match expected shape but doesn't change it

# 3. & 4. Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # left: boolean parameter with only two discrete values
    left: list = field(default_factory=lambda: [True, False])
    
    # out: can be None or a tensor with matching shape
    # For discretization: None, and tensor values at shape boundaries
    # Using typical values: None, zero tensor, matching shaped tensor
    # Note: We'll represent tensors as strings to avoid storage issues
    out: list = field(default_factory=lambda: [
        None,
        "zero_tensor",  # Placeholder for zero tensor
        "matching_tensor"  # Placeholder for correctly shaped tensor
    ])

    def get_out_tensor(self, choice: str, A_shape: tuple, B_shape: tuple, left: bool) -> Optional[torch.Tensor]:
        """Helper to convert string choices to actual tensors"""
        import torch
        
        if choice is None:
            return None
        elif choice == "zero_tensor":
            # Create zero tensor with expected output shape
            n = A_shape[-2] if left else A_shape[-1]
            if len(B_shape) == 1 or (len(B_shape) == 2 and B_shape[-1] == 1):
                return torch.zeros(A_shape[:-2] + (n,))
            else:
                return torch.zeros(A_shape[:-2] + (n, B_shape[-1]))
        elif choice == "matching_tensor":
            # Create random tensor with expected output shape
            n = A_shape[-2] if left else A_shape[-1]
            if len(B_shape) == 1 or (len(B_shape) == 2 and B_shape[-1] == 1):
                return torch.randn(A_shape[:-2] + (n,))
            else:
                return torch.randn(A_shape[:-2] + (n, B_shape[-1]))
        return None