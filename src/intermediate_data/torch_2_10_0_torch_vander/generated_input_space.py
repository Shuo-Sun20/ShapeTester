import torch
from dataclasses import dataclass, field

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': [torch.tensor([1.0, 2.0, 3.0, 5.0])],
    'N': 3,
    'increasing': True
}

# 2. Parameters affecting output shape (excluding 'inputs'): 'N'
# 3. Parameter value spaces:
#    - N: Optional[int]. Discrete values: None (default), and non-negative integers
#    We include boundary values (0, 1), typical values including the valid test case value (3),
#    and values representing different scenarios relative to input length (here input length is 4)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    N: list = field(default_factory=lambda: [None, 0, 1, 2, 3, 4, 5, 10])