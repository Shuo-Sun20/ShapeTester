import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

def call_func(size=None, scale_factor=None, mode='nearest', align_corners=False, recompute_scale_factor=None, inputs=None):
    upsample_layer = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    return upsample_layer(inputs[0])

# 1. Valid test case
valid_test_case = {
    'size': None,
    'scale_factor': 2.0,
    'mode': 'bilinear',
    'align_corners': False,
    'recompute_scale_factor': None,
    'inputs': [torch.randn(1, 3, 24, 32)]
}

# 2 & 3. Parameters affecting output shape and their value spaces
# size: Can be int or tuple, None means not specified
# scale_factor: Can be float or tuple, None means not specified
# recompute_scale_factor: bool or None, doesn't affect shape

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # size value space: covers None, integer values, and tuple values
    size: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,  # Not specified
            48,    # Single int for 1D
            64,    # Another single int
            (48, 64),    # Tuple for 2D
            (100, 150),  # Different tuple values
            (24, 32),    # Same as input size
            (10, 10),    # Smaller than input
            (200, 200),  # Larger than input
            (48, 64, 96),  # Tuple for 3D (if input were 5D)
        ]
    )
    
    # scale_factor value space: covers None, float values, and tuple values
    scale_factor: List[Optional[Union[float, Tuple[float, ...]]]] = field(
        default_factory=lambda: [
            None,      # Not specified
            0.5,       # Downsampling
            1.0,       # No change
            2.0,       # Upsampling (from valid_test_case)
            3.5,       # Larger upsampling
            0.25,      # More aggressive downsampling
            4.0,       # Larger upsampling
            (0.5, 2.0),   # Tuple for different scaling per dimension
            (2.0, 3.0),   # Another tuple
            (1.5, 1.5),   # Uniform tuple scaling
        ]
    )
    
    # mode: discrete parameter (doesn't affect shape but included for completeness)
    mode: List[str] = field(
        default_factory=lambda: [
            'nearest',
            'linear',
            'bilinear',
            'bicubic',
            'trilinear'
        ]
    )
    
    # align_corners: discrete parameter (doesn't affect shape)
    align_corners: List[Optional[bool]] = field(
        default_factory=lambda: [False, True, None]
    )
    
    # recompute_scale_factor: discrete parameter (doesn't affect shape)
    recompute_scale_factor: List[Optional[bool]] = field(
        default_factory=lambda: [None, False, True]
    )

# Test instantiation
var = InputSpace()
print("InputSpace instantiated successfully")
print(f"Size values: {var.size[:3]}...")
print(f"Scale factor values: {var.scale_factor[:3]}...")