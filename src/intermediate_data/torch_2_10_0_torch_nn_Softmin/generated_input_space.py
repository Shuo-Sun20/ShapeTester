import torch
from dataclasses import dataclass
from typing import List

# 1. valid_test_case definition
valid_test_case = {
    "dim": 1,
    "inputs": torch.randn(2, 3)
}

# 3. Parameter analysis and value space construction
# For torch.nn.Softmin, the only parameter besides inputs is 'dim'
# 'dim' must be an integer within the valid dimension range of the input tensor
# Since input shape varies, we consider typical cases for 0D to 5D tensors
# For discrete parameter 'dim', we list possible values from -5 to 4 to cover negative indexing

# 4. InputSpace definition
@dataclass
class InputSpace:
    # 'dim' is the only parameter that affects the computation (though not the shape)
    # We provide a comprehensive list of typical values including valid and invalid cases
    dim: List[int] = None
    
    def __post_init__(self):
        if self.dim is None:
            # Discrete value space for dim parameter
            # Including typical values: negative indices, valid indices, boundary cases
            self.dim = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]

# Instantiation test (as required by the problem)
var = InputSpace()