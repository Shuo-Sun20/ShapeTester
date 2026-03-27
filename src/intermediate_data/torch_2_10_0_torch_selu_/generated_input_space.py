import torch
from dataclasses import dataclass
from typing import Union, List

example_input = torch.randn(3, 4)
valid_test_case = {'inputs': example_input}

@dataclass
class InputSpace:
    # There are no parameters in call_func that affect output tensor shape
    # besides 'inputs', which is explicitly excluded per the instructions
    # Therefore, this dataclass has no fields
    pass