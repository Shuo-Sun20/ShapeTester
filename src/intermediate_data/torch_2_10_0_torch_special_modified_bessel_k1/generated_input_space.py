import torch
from dataclasses import dataclass

# 1. Define valid_test_case with all call_func parameters
valid_test_case = {
    'inputs': torch.randn(5, 3, dtype=torch.float32),  # Single tensor input
    'out': None  # Optional output tensor parameter
}

# 2 & 3 & 4. Define InputSpace dataclass for parameters affecting output shape
# Note: modified_bessel_k1 is an element-wise function, so output shape = input shape
# The only parameter affecting output shape is 'out' (when provided)

@dataclass
class InputSpace:
    # 'out' parameter: can be None or a tensor with same shape as input
    # Discretized value space for 'out':
    # - None (default)
    # - torch.Tensor with various dtypes and devices
    out: list = None  # Will be initialized in __post_init__
    
    def __post_init__(self):
        if self.out is None:
            # Create input tensor (shape 5x3) for reference
            input_tensor = torch.randn(5, 3)
            
            # Define discretized value space for 'out':
            # 1. None (default)
            # 2-6. Tensors with same shape but different dtypes
            # 7. Tensor on CPU (default)
            # 8. Tensor on GPU if available
            self.out = [
                None,  # Default
                torch.empty_like(input_tensor, dtype=torch.float32),  # Same dtype
                torch.empty_like(input_tensor, dtype=torch.float64),  # Double precision
                torch.empty_like(input_tensor, dtype=torch.float16),  # Half precision
                torch.empty_like(input_tensor, dtype=torch.bfloat16),  # Brain floating point
                torch.empty_like(input_tensor, dtype=torch.complex64),  # Complex
                torch.empty_like(input_tensor).cpu(),  # Explicit CPU
            ]
            
            # Add GPU tensor if CUDA is available
            if torch.cuda.is_available():
                self.out.append(
                    torch.empty_like(input_tensor, device='cuda')
                )