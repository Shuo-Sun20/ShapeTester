import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List

valid_test_case = {
    "eps": 1e-05,
    "momentum": 0.1,
    "affine": False,
    "track_running_stats": False,
    "device": None,
    "dtype": None,
    "inputs": torch.randn(2, 3, 4, 5)
}

@dataclass
class InputSpace:
    pass