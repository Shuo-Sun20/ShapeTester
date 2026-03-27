import torch
from dataclasses import dataclass, field
from typing import Optional, Any

# 1. Define valid_test_case variable
valid_test_case = {
    'inputs': [
        torch.randn(10, 3, 5),  # input_tensor
        torch.randn(10, 3, 4),  # batch1
        torch.randn(10, 4, 5)   # batch2
    ],
    'out_dtype': None,
    'beta': 1,
    'alpha': 1,
    'out': None
}

# 2. Parameters affecting output shape: None (except inputs)

# 3. Value space analysis:
# - out_dtype: None, torch.float16, torch.float32, torch.bfloat16, torch.float64
# - beta: int/float values
# - alpha: int/float values
# - out: None, or tensor with correct shape

@dataclass
class InputSpace:
    """Parameters that can affect the shape of the output tensor"""
    
    out_dtype: Optional[Any] = field(
        default_factory=lambda: [
            None,
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.float64
        ]
    )
    
    beta: float = field(
        default_factory=lambda: [
            0.0,  # special case (ignores input)
            0.5,
            1.0,  # default
            2.0,
            -1.0
        ]
    )
    
    alpha: float = field(
        default_factory=lambda: [
            0.0,
            0.5,
            1.0,  # default
            2.0,
            -1.0
        ]
    )
    
    out: Optional[Any] = field(
        default_factory=lambda: [
            None,
            'same_shape_zeros',
            'same_shape_ones',
            'same_shape_randn',
            'broadcastable_shape'
        ]
    )