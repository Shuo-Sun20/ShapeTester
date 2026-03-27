import torch
from dataclasses import dataclass
from typing import List

# 1. Valid test case
valid_test_case = {"inputs": torch.randn(3, 3, dtype=torch.complex64)}

# 4. InputSpace definition
@dataclass
class InputSpace:
    # Only parameter affecting output shape (except 'inputs' is the only parameter)
    pass