from dataclasses import dataclass
import torch

def call_func(inputs):
    return torch.isnan(inputs)

# 1. Valid test case
valid_test_case = {"inputs": torch.tensor([1.0, float('nan'), 2.0, float('nan'), 3.0])}

# 2. Parameters affecting output shape (excluding "inputs")
# torch.isnan only has one parameter "input" which directly determines the output shape.
# In call_func, the parameter is named "inputs". There are no other parameters affecting shape.

# 3. No other parameters exist besides "inputs"

# 4. InputSpace definition
@dataclass
class InputSpace:
    pass  # No parameters except "inputs" to affect output shape