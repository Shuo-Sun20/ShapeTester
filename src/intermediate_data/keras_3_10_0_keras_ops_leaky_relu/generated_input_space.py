import numpy as np
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': np.random.uniform(-1, 1, size=(5, 3)),
    'negative_slope': 0.2
}

@dataclass
class InputSpace:
    # Parameter that can affect the shape of the output tensor
    # The shape is determined by the input tensor itself, not by any other parameters
    
    # Since no parameters besides 'inputs' affect the output shape,
    # this dataclass has no fields that affect shape
    pass