import torch
from dataclasses import dataclass
from typing import Optional, List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn((3, 4, 6, 2))],  # prod(3,4)=12 equals prod(6,2)=12
    "ind": 2,
    "out": None
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the output tensor shape (except 'inputs') 
    with their discretized value ranges.
    
    Parameters:
    - ind: Integer specifying the split point for tensor dimensions. 
            Must be between 1 and tensor_ndim-1 (inclusive) for meaningful cases.
            Value 0 is technically allowed for 0D/1D edge cases but rarely used.
    - out: Optional output tensor parameter (doesn't affect output shape, included for completeness)
    """
    
    # ind value space: discrete parameter covering legal scenarios
    ind: List[int] = None
    
    # Note: 'out' doesn't affect output shape but is included in call_func signature
    # We include its type space for completeness
    out: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        # Define ind value space if not provided
        if self.ind is None:
            # All possible legal values for ind (discrete parameter)
            # Covering: boundary values (1, tensor_ndim), typical values (2,3,4), 
            # and edge case 0 (for completeness)
            self.ind = [0, 1, 2, 3, 4, 5, 6]
            # Note: 0 corresponds to special cases (empty product=1)
            # The upper bound 6 covers most practical tensor dimensionalities
        
        if self.out is None:
            # out parameter type space: None or tensor with matching shape
            self.out = [None]
            # Note: Actual tensor values would require generating matching shapes,
            # but for type space representation we just indicate the possibilities

# Example instantiation
if __name__ == "__main__":
    input_space = InputSpace()
    print(f"ind value space: {input_space.ind}")
    print(f"out type space: {input_space.out}")
    
    # Demonstrate that valid_test_case parameters are in value spaces
    assert valid_test_case["ind"] in input_space.ind
    assert valid_test_case["out"] in input_space.out