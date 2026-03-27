import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.randn(2, 3),
    "out": None
}

# 2. Parameters affecting output shape except "inputs": only "out"
# The "out" parameter directly determines the output tensor's shape if provided.

# 3. Discretized value spaces for parameters affecting output shape

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape of torch.special.bessel_j1."""
    
    # Parameter: out
    # Type: Optional[torch.Tensor]
    # Value space: None + tensors with various shapes matching inputs
    out: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Define base input tensor shapes to match
            base_shapes = [
                torch.randn(1),          # scalar
                torch.randn(3),          # 1D vector
                torch.randn(2, 3),       # 2D matrix
                torch.randn(1, 4, 5),    # 3D tensor
                torch.randn(2, 0, 3),    # empty dimension
                torch.randn(2, 3, 4, 5), # 4D tensor
            ]
            
            # Construct value space for 'out' parameter:
            # 1. None
            # 2. Various tensor shapes (must match input shape in actual calls)
            # 3. Boundary cases (empty tensors, scalar, high-dim)
            self.out = [
                None,  # Case: no output tensor provided
                torch.empty(1),          # scalar output
                torch.empty(3),          # 1D output
                torch.empty(2, 3),       # 2D output  
                torch.empty(1, 4, 5),    # 3D output
                torch.empty(2, 0, 3),    # output with empty dimension
                torch.empty(2, 3, 4, 5), # 4D output
                torch.empty(1, 1),       # 1x1 matrix
                torch.empty(100, 100),   # large square matrix
                torch.empty(0),          # empty tensor
            ]