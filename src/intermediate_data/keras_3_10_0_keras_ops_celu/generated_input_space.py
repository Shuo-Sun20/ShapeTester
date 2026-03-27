import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define a valid test case
valid_test_case = {
    'inputs': np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]),
    'alpha': 1.0
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters affecting output shape (alpha doesn't affect shape)
    pass