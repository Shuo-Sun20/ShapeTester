import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

def call_func(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False, inputs=None):
    instance_norm = nn.InstanceNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    return instance_norm(inputs)

# 1. Define valid_test_case
input_tensor = torch.randn(4, 3, 32, 32, 32)
valid_test_case = {
    "num_features": 3,
    "eps": 1e-5,
    "momentum": 0.1,
    "affine": False,
    "track_running_stats": False,
    "inputs": input_tensor
}

# 2. Identify parameters affecting output shape (except "inputs")
# Based on the documentation, the only parameter that can affect output shape 
# is "num_features" as it must match the channel dimension of the input.
# Other parameters (eps, momentum, affine, track_running_stats) only affect 
# the computation values but not the shape.

@dataclass
class InputSpace:
    # num_features must match input's channel dimension (C) in (N, C, D, H, W)
    # Typical values for channel dimensions in 3D data
    num_features: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 8, 16, 32, 64, 128])
    # Other parameters do not affect output shape, but included for completeness
    eps: List[float] = field(default_factory=lambda: [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    momentum: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 0.9, 0.99, 0.999])
    affine: List[bool] = field(default_factory=lambda: [True, False])
    track_running_stats: List[bool] = field(default_factory=lambda: [True, False])