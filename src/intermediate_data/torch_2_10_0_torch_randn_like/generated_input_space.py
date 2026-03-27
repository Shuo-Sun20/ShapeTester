import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. valid_test_case definition
example_input = torch.randn(3, 4)
valid_test_case = {
    "inputs": example_input,
    "generator": None,
    "dtype": torch.float64,
    "layout": torch.strided,
    "device": torch.device('cpu'),
    "requires_grad": True,
    "memory_format": torch.preserve_format
}

@dataclass
class InputSpace:
    """
    Parameters affecting output tensor shape for torch.randn_like.
    Contains discretized value ranges for each parameter.
    """
    # Based on torch.randn_like documentation and PyTorch 2.10.0:
    # Only 'inputs' parameter affects the output shape directly.
    # Other parameters (dtype, layout, device, requires_grad, memory_format) 
    # affect tensor properties but not shape.
    # Since 'inputs' is excluded per requirement, this class remains empty.
    pass