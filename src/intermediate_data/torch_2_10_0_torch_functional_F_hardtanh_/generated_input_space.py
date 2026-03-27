import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4),
    "min_val": -1.0,
    "max_val": 1.0
}

# 2. Parameters affecting output shape (except "inputs"): None
# hardtanh_ is an in-place operation that doesn't change tensor shape

# 3. Value space analysis for all call_func parameters:
# - inputs: Tensor type, affects shape but excluded per requirements
# - min_val: float type, doesn't affect shape
# - max_val: float type, doesn't affect shape

@dataclass
class InputSpace:
    """
    Note: hardtanh_ is an in-place operation that doesn't change tensor shape.
    The shape of the output is always identical to the input shape.
    No parameters besides 'inputs' affect the output shape.
    """
    pass

# Example instantiation
var = InputSpace()