import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

valid_test_case = {
    "inputs": [torch.randn(3, 4)],
    "dtype": None,
    "layout": None,
    "device": None,
    "requires_grad": False,
    "memory_format": torch.preserve_format
}

@dataclass
class InputSpace:
    dtype: List[Optional[torch.dtype]] = field(default_factory=lambda: [None, torch.float32, torch.int64, torch.bool, torch.float64])
    layout: List[Optional[torch.layout]] = field(default_factory=lambda: [None, torch.strided])
    device: List[Optional[torch.device]] = field(default_factory=lambda: [None, torch.device('cpu'), torch.device('cuda')])
    requires_grad: List[bool] = field(default_factory=lambda: [True, False])
    memory_format: List[torch.memory_format] = field(default_factory=lambda: [torch.preserve_format, torch.contiguous_format])