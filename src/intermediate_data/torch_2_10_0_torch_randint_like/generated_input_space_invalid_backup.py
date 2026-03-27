import torch
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, low=0, high=None, generator=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.randint_like(inputs, low=low, high=high, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

valid_test_case = {
    'inputs': torch.randn(3, 4, 5),
    'low': 0,
    'high': 100,
    'generator': None,
    'dtype': None,
    'layout': None,
    'device': None,
    'requires_grad': False,
    'memory_format': torch.preserve_format
}

@dataclass
class InputSpace:
    inputs: List[torch.Size] = field(default_factory=lambda: [
        torch.Size([2, 3]),
        torch.Size([4, 5, 6]),
        torch.Size([1]),
        torch.Size([2, 2, 2, 2]),
        torch.Size([3, 1, 4])
    ])