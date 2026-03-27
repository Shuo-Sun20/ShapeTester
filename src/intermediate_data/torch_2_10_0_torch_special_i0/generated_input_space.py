import torch
from dataclasses import dataclass
from typing import Optional

valid_test_case = {
    "inputs": [torch.randn(3, 3, dtype=torch.float32)],
    "out": None
}

@dataclass
class InputSpace:
    out: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        # Define shape variations for the out tensor
        # Shapes include: scalar, 1D, 2D, 3D, 4D with varying dimensions
        # For simplicity, we'll use a fixed dtype and device for demonstration
        shape_candidates = [
            None,  # No out tensor
            torch.empty(()),  # Scalar
            torch.empty(5),  # 1D
            torch.empty(3, 3),  # 2D square
            torch.empty(2, 5),  # 2D non-square
            torch.empty(2, 3, 4),  # 3D
            torch.empty(1, 2, 3, 4),  # 4D
        ]
        
        # Convert to list for the value space
        self.out = shape_candidates