import torch
from dataclasses import dataclass
from typing import Optional, List

valid_test_case = {
    "inputs": torch.arange(5, dtype=torch.float32),
    "out": None
}

@dataclass
class InputSpace:
    # Parameters that can affect the shape of the output tensor
    out: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Discrete value space for 'out' parameter
            # Includes None and various tensor shapes that match typical input shapes
            self.out = [
                None,
                # Small tensors
                torch.empty(1, dtype=torch.float32),
                torch.empty(3, dtype=torch.float32),
                torch.empty(5, dtype=torch.float32),
                # 2D tensors
                torch.empty(2, 3, dtype=torch.float32),
                torch.empty(4, 5, dtype=torch.float32),
                # 3D tensors
                torch.empty(2, 3, 4, dtype=torch.float32),
                torch.empty(5, 6, 7, dtype=torch.float32),
                # Special cases: empty and scalar tensors
                torch.empty(0, dtype=torch.float32),
                torch.tensor(3.14, dtype=torch.float32),
                # Tensor with different dtype (should still work if compatible)
                torch.empty(5, dtype=torch.float64),
            ]