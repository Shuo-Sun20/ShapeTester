import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# First, let's define the valid test case as required
torch.manual_seed(42)
input_tensor = torch.randn(3, 4)
mask_tensor = torch.randint(0, 2, (3, 4), dtype=torch.bool)

valid_test_case = {
    "inputs": input_tensor,
    "dim": 1,
    "keepdim": False,
    "dtype": None,
    "mask": mask_tensor
}

# Now define the InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the shape of output tensor
    from torch.masked.logsumexp API call
    """
    # dim parameter affects shape - can be int, tuple of ints, or None
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,          # reduces all dimensions
            0,             # reduce along first dimension
            1,             # reduce along second dimension  
            -1,            # reduce along last dimension (negative indexing)
            -2,            # reduce along second-to-last dimension
            (0, 1),        # reduce along first two dimensions
            (0, 2),        # reduce along first and third dimensions (for 3D+ tensors)
            (1, -1),       # mixed positive and negative indices
            (0,),          # single-element tuple (equivalent to int 0)
        ]
    )
    
    # keepdim parameter affects shape - boolean flag
    keepdim: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    # Note: 'dtype' and 'mask' do not affect output tensor shape, 
    # so they are not included in this class
    # 'inputs' parameter is explicitly excluded as per requirements