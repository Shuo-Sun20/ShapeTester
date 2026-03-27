import torch
from dataclasses import dataclass
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
    low: List[int] = None
    high: List[int] = None
    dtype: List[Optional[torch.dtype]] = None
    layout: List[Optional[torch.layout]] = None
    device: List[Optional[torch.device]] = None
    
    def __post_init__(self):
        if self.low is None:
            self.low = [-10, -5, 0, 5, 10, 50, 100]
        if self.high is None:
            self.high = [1, 2, 5, 10, 50, 100, 1000]
        if self.dtype is None:
            self.dtype = [None, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
        if self.layout is None:
            self.layout = [None, torch.strided, torch.sparse_coo, torch.sparse_csr]
        if self.device is None:
            self.device = [None, torch.device('cpu'), torch.device('cuda')]