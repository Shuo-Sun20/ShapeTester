import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    'in_channels': 16,
    'out_channels': 33,
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'bias': True,
    'padding_mode': 'zeros',
    'inputs': torch.randn(20, 16, 50, 100)
}

# 2. Parameters affecting output shape (excluding inputs):
# in_channels, out_channels, kernel_size, stride, padding, dilation, groups
# Note: in_channels and out_channels affect the channel dimensions, while the others affect spatial dimensions

@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output tensor shape with discretized value ranges."""
    
    # 1. in_channels: Must match input tensor channels, typical values for testing
    in_channels: List[int] = field(default_factory=lambda: [1, 3, 16, 32, 64, 128, 256])
    
    # 2. out_channels: Must be divisible by groups
    out_channels: List[int] = field(default_factory=lambda: [1, 16, 33, 64, 128, 256, 512])
    
    # 3. kernel_size: Can be int or tuple
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 3, 5, 7, (3, 3), (5, 5), (7, 7)]
    )
    
    # 4. stride: Can be int or tuple
    stride: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 1), (2, 2), (1, 2), (2, 1)]
    )
    
    # 5. padding: Can be int, tuple, or str
    padding: List[Union[int, Tuple[int, int], str]] = field(
        default_factory=lambda: [0, 1, 2, 3, (1, 1), (2, 2), (3, 3), 'valid', 'same']
    )
    
    # 6. dilation: Can be int or tuple
    dilation: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 1), (2, 2), (1, 2)]
    )
    
    # 7. groups: Must divide both in_channels and out_channels
    groups: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])