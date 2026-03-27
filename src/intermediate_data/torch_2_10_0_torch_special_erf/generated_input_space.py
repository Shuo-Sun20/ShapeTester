import torch
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4),
    "out": None
}

# Task 2 & 3: Identify parameters affecting output shape (besides "inputs")
# Only "out" parameter can affect output shape (must match input shape if provided)
# out can be None or a tensor with matching shape to input

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        # Discrete parameter values:
        None,  # Default value (output shape determined by input)
        
        # Tensor values with various typical patterns:
        torch.tensor(0.0),  # Scalar (broadcasts to any shape)
        torch.tensor(1.0),  # Scalar
        torch.tensor(-1.0),  # Scalar
        torch.tensor(float('inf')),  # Scalar boundary
        torch.tensor(float('-inf')),  # Scalar boundary
        
        # For shape compatibility, these would need to match input shape in practice
        # Representing typical tensor values that could be used:
        torch.zeros(1),  # 1D tensor
        torch.ones(1),  # 1D tensor
        torch.full((2, 2), 0.5),  # 2D tensor
        torch.full((2, 2), -0.5),  # 2D tensor
        torch.randn(2, 2)  # 2D tensor with random values
    ])