from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, List
import torch

def call_func(inputs, dim, keepdim=False, dtype=None, mask=None):
    input_tensor = inputs[0] if isinstance(inputs, list) else inputs
    mask_tensor = mask
    return torch.masked.amax(input=input_tensor, dim=dim, keepdim=keepdim, dtype=dtype, mask=mask_tensor)

torch.manual_seed(42)
input_tensor = torch.randn(3, 4, 5)
mask_tensor = torch.bernoulli(torch.full((3, 4, 5), 0.7)).bool()
example_output = call_func(inputs=[input_tensor], dim=1, keepdim=True, mask=mask_tensor)

valid_test_case = {
    "inputs": [input_tensor],
    "dim": 1,
    "keepdim": True,
    "dtype": None,
    "mask": mask_tensor
}

@dataclass
class InputSpace:
    dim: List[Union[int, Tuple[int, ...], None]] = field(
        default_factory=lambda: [
            None,  # default: reduce over all dimensions
            0, 1, 2, -1, -2, -3,  # single dimension indices
            (0, 1), (0, 2), (1, 2), (0, 1, 2),  # multiple dimensions
            (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3),  # negative indices
            (0, -1), (1, -2)  # mixed positive/negative indices
        ]
    )
    keepdim: List[bool] = field(default_factory=lambda: [True, False])