import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': torch.rand(5, 5),  # Random 5x5 tensor
    'out': None  # No output tensor provided
}

# Task 2 & 3: Analyze parameters affecting output shape
# Only 'out' parameter affects output shape besides 'inputs'
# 'out' can be: None (default), or a torch.Tensor

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the shape of the output tensor
    for torch.special.modified_bessel_k0.
    
    The 'out' parameter controls whether an output tensor is provided.
    When 'out' is None, output shape matches input shape.
    When 'out' is a tensor, output shape matches that tensor's shape.
    """
    out: List[Optional[Union[str, torch.Tensor]]] = field(
        default_factory=lambda: [
            None,  # No output tensor provided
            'same_shape_same_dtype',  # Tensor with same shape/dtype as input
            'same_shape_different_dtype',  # Tensor with same shape, different dtype
            'preallocated',  # Pre-allocated tensor with compatible shape
            'broadcastable',  # Tensor with broadcastable shape
            torch.tensor(0.0),  # Scalar tensor (edge case)
            torch.randn(5, 5),  # Random tensor with same shape as typical input
        ]
    )
    
    @staticmethod
    def generate_out_value(option: Optional[Union[str, torch.Tensor]], 
                          input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert InputSpace options to actual tensor values for testing."""
        if option is None:
            return None
        elif isinstance(option, torch.Tensor):
            return option
        elif option == 'same_shape_same_dtype':
            return torch.empty_like(input_tensor)
        elif option == 'same_shape_different_dtype':
            return torch.empty(input_tensor.shape, 
                             dtype=torch.float64 if input_tensor.dtype == torch.float32 
                             else torch.float32)
        elif option == 'preallocated':
            return torch.empty(input_tensor.shape, device=input_tensor.device)
        elif option == 'broadcastable':
            # For a 5x5 input, this creates a tensor that can broadcast
            return torch.randn(5, 1) if input_tensor.dim() == 2 else torch.randn(5)
        else:
            raise ValueError(f"Unknown out option: {option}")

# The InputSpace class can be instantiated successfully:
# var = InputSpace()