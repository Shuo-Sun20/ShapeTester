import torch
from dataclasses import dataclass

# Task 1: Define a valid test case
valid_test_case = {
    "approximate": "none",
    "inputs": torch.randn(2, 3)
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the shape of the output tensor
    (excluding "inputs") and their discretized value ranges.
    
    For torch.nn.GELU, only the "approximate" parameter is relevant,
    but it does NOT affect the output shape. However, as per the task
    requirements, we include parameters that could theoretically influence
    shape in other layers (though not in GELU specifically).
    
    Since torch.nn.GELU's output shape always matches input shape,
    no parameters besides "inputs" affect shape. We still define the
    class with relevant parameters to match the API.
    """
    # Only parameter that could be relevant for activation functions
    approximate: list = None
    
    def __post_init__(self):
        if self.approximate is None:
            # All possible discrete values for approximate parameter
            self.approximate = ['none', 'tanh']

# Example usage
var = InputSpace()