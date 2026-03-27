import torch
import torch.nn as nn
from dataclasses import dataclass, field

# Task 1: Define a valid test case
valid_test_case = {
    "p": 0.2,
    "inplace": False,
    "inputs": torch.randn(20, 16)
}

# Task 2, 3, and 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Based on analysis:
    # 1. p (float): Probability of element being zeroed, must be 0 <= p < 1
    # 2. inplace (bool): Whether to perform operation in-place
    
    # For p: continuous parameter between 0 and 1 (excluding 1)
    # Discretized with boundary values and typical values
    p: list = field(default_factory=lambda: [
        0.0,       # boundary: no dropout
        0.1,       # typical low dropout
        0.2,       # typical moderate dropout (included from valid_test_case)
        0.3,       # typical moderate-high dropout
        0.5,       # typical: default value from documentation
        0.7,       # typical high dropout
        0.9,       # boundary: very high dropout
        0.999      # boundary: almost all dropped (just below 1)
    ])
    
    # For inplace: discrete boolean parameter
    inplace: list = field(default_factory=lambda: [
        False,     # default from documentation
        True       # in-place operation
    ])