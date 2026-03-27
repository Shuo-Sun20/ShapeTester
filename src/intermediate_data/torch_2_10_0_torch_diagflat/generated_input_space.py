import torch
from dataclasses import dataclass
from typing import List

# 1. Define a valid test case
valid_test_case = {
    'inputs': [torch.randn(3)],
    'offset': 0
}

# 2. Identify parameters affecting output shape (excluding "inputs")
# The only parameter is "offset"

# 3. Analyze parameter types and construct value spaces
# offset: integer, can be positive, negative, or zero. 
# Discretization includes boundary values and 5 typical values:
# -10 (large negative), -5, -2, -1, 0 (main diagonal), 1, 2, 5, 10 (large positive)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    offset: List[int] = None
    
    def __post_init__(self):
        if self.offset is None:
            # Discretized value space for offset parameter
            self.offset = [-10, -5, -2, -1, 0, 1, 2, 5, 10]

# This allows instantiation with: var = InputSpace()