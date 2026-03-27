import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Define valid test case
valid_test_case = {
    'inputs': np.random.randn(4, 8).astype(np.float32),
    'axis': 1
}

@dataclass
class InputSpace:
    """
    Parameter space for the call_func parameters affecting output shape.
    """
    axis: List[int] = field(default_factory=lambda: [
        # Negative boundary values
        -3, -2, -1,
        # Zero
        0,
        # Positive boundary values
        1, 2, 3,
        # Other typical values
        -10, -5, 5, 10
    ])