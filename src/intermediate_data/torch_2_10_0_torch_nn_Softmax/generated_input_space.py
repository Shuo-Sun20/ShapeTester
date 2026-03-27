import torch
from dataclasses import dataclass
from typing import List, Union

valid_test_case = {
    'dim': 1,
    'inputs': torch.randn(2, 3)
}

@dataclass
class InputSpace:
    # Only the 'dim' parameter affects the computation but not the output shape.
    # According to the documentation, the output shape is always the same as the input shape.
    # Therefore, no parameters other than 'inputs' affect the output shape.
    pass

# Example instantiation
var = InputSpace()