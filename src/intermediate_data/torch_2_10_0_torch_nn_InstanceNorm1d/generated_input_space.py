import torch
from dataclasses import dataclass, field
from typing import List, Union

valid_test_case = {
    "num_features": 100,
    "inputs": torch.randn(20, 100, 40),
    "eps": 1e-5,
    "momentum": 0.1,
    "affine": False,
    "track_running_stats": False
}

@dataclass
class InputSpace:
    # Parameters that affect output shape (directly or indirectly):
    # Only num_features affects shape (must match input channel dimension)
    num_features: List[int] = field(default_factory=lambda: [1, 2, 16, 64, 100, 256, 512])
    
    # Other parameters that don't affect shape but affect computation:
    eps: List[float] = field(default_factory=lambda: [0.0, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 1.0])
    momentum: List[Union[float, None]] = field(default_factory=lambda: [None, 0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
    affine: List[bool] = field(default_factory=lambda: [True, False])
    track_running_stats: List[bool] = field(default_factory=lambda: [True, False])