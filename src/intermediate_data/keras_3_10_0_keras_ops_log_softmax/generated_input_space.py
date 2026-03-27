import numpy as np
from dataclasses import dataclass
from typing import List

valid_test_case = {
    'inputs': np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]),
    'axis': -1
}

@dataclass
class InputSpace:
    # Parameters that affect the shape of the output tensor (excluding "inputs")
    # In log_softmax, only "axis" parameter affects the shape since it determines
    # along which dimension the normalization is applied (though output shape remains same as input)
    
    # axis: integer axis parameter for log_softmax operation
    # Discrete parameter - list all possible meaningful values
    axis: List[int] = None
    
    def __post_init__(self):
        if self.axis is None:
            # Create value space for axis parameter
            # For a typical 2D input tensor with shape (batch, features)
            # Axis can be: -2 (batch dimension), -1 (feature dimension), 
            # 0 (batch dimension), 1 (feature dimension)
            
            # Typical 2D tensor shape: (batch_size, features)
            # We'll generate values that cover:
            # 1. Normalized indices: -2, -1
            # 2. Absolute indices: 0, 1
            # 3. Edge cases for different tensor dimensions
            
            self.axis = [-2, -1, 0, 1]
            # For completeness, we also consider typical values for 3D tensors
            # which might be common in sequence models
            # Add some values that would be valid for 3D tensors too
            self.axis.extend([-3, 2])
            
            # Ensure we have at least 5 values as requested
            # Our current list has 6 values: [-2, -1, 0, 1, -3, 2]
            # These cover all legal scenarios for tensors up to 3 dimensions
            
            # Sort for better organization
            self.axis.sort()