import torch
from dataclasses import dataclass, field
from typing import List, Union

valid_test_case = {
    "inputs": torch.randn(4, dtype=torch.cfloat)
}

@dataclass
class InputSpace:
    """
    Note: torch.imag only accepts a single parameter 'input' (tensor).
    The output shape is identical to the input shape, which is determined
    solely by the input tensor's shape attribute. There are no other
    explicit parameters in the API call that affect output shape.
    
    Since the shape is entirely determined by the input tensor itself,
    and not by any additional parameters of torch.imag, the InputSpace
    class is defined without additional fields.
    """
    pass