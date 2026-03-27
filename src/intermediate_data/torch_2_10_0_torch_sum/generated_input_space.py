import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    'inputs': [torch.randn(4, 4)], 
    'dim': 1, 
    'keepdim': False, 
    'dtype': None
}

@dataclass
class InputSpace:
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(default_factory=lambda: [
        None,      # Reduce all dimensions
        0,         # Reduce along first dimension
        1,         # Reduce along second dimension  
        -1,        # Reduce along last dimension
        (0, 1),    # Reduce along multiple dimensions
        (0, -1),   # Reduce along first and last dimensions
        (-2, -1),  # Reduce along last two dimensions
    ])
    
    keepdim: List[bool] = field(default_factory=lambda: [
        False,      # Output squeezed (default)
        True,       # Output dimensions retained
    ])