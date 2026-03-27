import torch
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, out=None):
    return torch.bitwise_left_shift(inputs[0], inputs[1], out=out)

# 1. Valid test case
example_input1 = torch.randint(-128, 127, (3,), dtype=torch.int8)
example_input2 = torch.tensor([1, 0, 3], dtype=torch.int8)
valid_test_case = {
    "inputs": [example_input1, example_input2],
    "out": None
}

# 2. Parameters affecting output shape (except "inputs"): only "out"

# 3. Discretized value space for "out" parameter
# Type: Union[None, torch.Tensor]
# Discrete values: None and tensor shapes broadcastable to (3,)
out_values = [
    None,
    torch.empty(3, dtype=torch.int8),
    torch.empty(1, 3, dtype=torch.int8),
    torch.empty(3, 1, dtype=torch.int8),
    torch.empty(1, 1, 3, dtype=torch.int8)
]

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Union[None, torch.Tensor]] = field(default_factory=lambda: out_values)