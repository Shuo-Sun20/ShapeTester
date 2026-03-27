import torch
from dataclasses import dataclass

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.rand(4, 4) * 5,
    "generator": None
}

# 2. Identify parameters affecting output shape (except "inputs")
# torch.poisson output shape is solely determined by the input tensor shape.
# No other parameters (like generator) affect the output shape.
# Therefore, there are no such parameters besides "inputs".

# 3. Since there are no such parameters, no value spaces need to be constructed.
# 4. Define InputSpace dataclass with all parameters affecting shape
@dataclass
class InputSpace:
    # No fields since no parameters (except "inputs") affect output shape
    pass