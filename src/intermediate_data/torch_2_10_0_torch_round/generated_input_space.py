import torch
from dataclasses import dataclass
from typing import Optional, List, Union

# Valid test case definition
valid_test_case = {
    'inputs': torch.randn(3, 4, dtype=torch.float32),
    'decimals': 2,
    'out': None
}

# Analysis of parameters affecting output shape:
# The torch.round API returns a tensor with the same shape as the input tensor.
# The 'decimals' parameter affects the numerical values but not the shape.
# The 'out' parameter, when provided, must have the same shape as the input tensor
# to store the result, but it doesn't change the shape of the computation result.
# Therefore, only 'inputs' directly determines the output shape.
# However, since we're asked to identify parameters OTHER than 'inputs' that
# could affect shape, and there are none, InputSpace will be empty.

@dataclass
class InputSpace:
    # Note: After analysis, no parameters besides 'inputs' affect the output shape.
    # The shape is solely determined by the input tensor's shape.
    # Therefore, this dataclass has no fields related to shape-affecting parameters
    # (excluding 'inputs' as specified in the requirements).
    pass

# For completeness, here's a separate dataclass showing the value spaces
# for ALL call_func parameters, even though only 'inputs' affects shape:

@dataclass
class CompleteParameterSpace:
    """Value spaces for all call_func parameters."""
    # Input tensor specifications (affects shape)
    input_shapes: List[tuple] = None  # Example: [(2,3), (4,5,6), (10,)]
    
    # decimals parameter value space
    decimals_values: List[int] = None
    
    # out parameter value space (tensor or None)
    out_values: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.input_shapes is None:
            self.input_shapes = [
                (2, 3),           # 2D tensor
                (4, 5, 6),        # 3D tensor  
                (10,),            # 1D tensor
                (1, 1, 1, 1),     # 4D tensor
                (0, 2),           # Edge case: empty dimension
                (3, 0, 4)         # Edge case: zero elements
            ]
        
        if self.decimals_values is None:
            self.decimals_values = [
                -5,               # Large negative
                -3,               # Negative boundary
                -1,               # Negative typical
                0,                # Default/zero
                1,                # Positive typical
                3,                # Positive boundary
                5                 # Large positive
            ]
        
        if self.out_values is None:
            # Create various output tensor options
            self.out_values = [
                None,  # No output tensor provided
                # Output tensors with different shapes (must match input shape in actual use)
                torch.empty(2, 3),
                torch.empty(4, 5, 6),
                torch.empty(10,),
                torch.empty(1, 1, 1, 1),
                torch.empty(0, 2)
            ]

# Example usage demonstrating the classes can be instantiated
if __name__ == "__main__":
    # The required InputSpace class (empty as no parameters affect shape besides 'inputs')
    shape_params = InputSpace()
    
    # Complete parameter space for reference
    param_space = CompleteParameterSpace()
    
    # Test that valid_test_case works
    def call_func(inputs, decimals=0, out=None):
        return torch.round(inputs, decimals=decimals, out=out)
    
    result = call_func(**valid_test_case)
    print("Test call successful, result shape:", result.shape)