import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

valid_test_case = {
    'inputs': [4],
    'd': 2.0,
    'out': None,
    'dtype': torch.float32,
    'layout': torch.strided,
    'device': torch.device('cpu'),
    'requires_grad': False
}

@dataclass
class InputSpace:
    # n is inside inputs[0], but inputs is excluded from analysis
    # Therefore InputSpace is empty since no other parameters affect output shape
    pass