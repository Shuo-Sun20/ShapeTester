import keras
import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "factor": 0.5,
    "value_range": (0, 255),
    "seed": 42,
    "inputs": np.random.uniform(low=0.0, high=255.0, size=(4, 32, 32, 3)).astype('float32'),
    "training": True
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the shape of the output tensor.
    Based on the documentation, the output shape is determined by the input shape.
    However, the parameters that affect shape are: 'training' (affects batch size behavior)
    but the actual shape parameters (batch, height, width, channels) only come from 'inputs'.
    Since we exclude 'inputs', there are no shape-affecting parameters.
    """
    pass