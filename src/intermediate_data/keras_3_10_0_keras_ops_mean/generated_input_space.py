import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)),
    "axis": 1,
    "keepdims": True
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of keras.ops.mean
    """
    
    # axis parameter value space
    axis: List[Union[int, Tuple[int, ...], List[int], None]] = field(
        default_factory=lambda: [
            None,                    # mean of flattened tensor
            0,                       # reduce along first axis
            1,                       # reduce along second axis
            -1,                      # reduce along last axis
            -2,                      # reduce along second-last axis
            (0, 1),                  # reduce along first two axes
            (0, -1),                 # reduce along first and last axes
            (1, 2),                  # reduce along last two axes (for 3D+)
            (0, 1, 2),               # reduce along all three axes (for 3D)
            [-1, -2],                # reduce along last two axes (as list)
            [0],                     # reduce along first axis (as list)
        ]
    )
    
    # keepdims parameter value space
    keepdims: List[bool] = field(
        default_factory=lambda: [
            True,   # keep reduced dimensions with size 1
            False,  # remove reduced dimensions
        ]
    )