import keras
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    'num_bins': 5,
    'output_mode': 'int',
    'sparse': False,
    'name': None,
    'dtype': None,
    'inputs': [np.array(['A', 'B', 'C']), np.array([101, 102, 103])]
}

# Task 2-4: InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding inputs):
    # 1. num_bins: continuous integer parameter
    num_bins: list = field(default_factory=lambda: [1, 10, 100, 1000, 10000])
    # 2. output_mode: discrete parameter
    output_mode: list = field(default_factory=lambda: ["int", "one_hot"])