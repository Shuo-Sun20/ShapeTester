import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    "factor": 0.5,
    "data_format": "channels_last",
    "seed": 42,
    "inputs": np.random.rand(4, 32, 32, 3).astype(np.float32)
}

# 2 & 3 & 4. Define InputSpace dataclass with shape-affecting parameters
@dataclass
class InputSpace:
    """
    Data class containing all parameters that affect the output tensor shape 
    for RandomGrayscale layer, with their discretized value spaces.
    
    data_format: Affects the shape interpretation/ordering of dimensions
                 by determining the channel dimension position.
    """
    data_format: List[str] = field(
        default_factory=lambda: ["channels_last", "channels_first"]
    )

# Example instantiation
var = InputSpace()