import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 4), dtype=tf.float32),
    "decreasing": True,
    "axis": 1
}

# 2. Parameters affecting output shape (besides "inputs"): 
# Only "axis" affects which dimension the operation is performed on
# "decreasing" is boolean and doesn't affect shape

# 3. Value space construction:
# - decreasing: Boolean, all possible values [True, False]
# - axis: Integer, discrete but can be continuous range; we'll discretize

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # decreasing parameter: boolean, all possible values
    decreasing: List[bool] = field(default_factory=lambda: [True, False])
    
    # axis parameter: discretized integer values
    # For a typical 2D tensor of shape (3,4), axis values can be:
    # -2, -1 (negative indexing)
    # 0, 1 (positive indexing)
    axis: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])
    
    # Note: "inputs" also affects shape but is not included here as per instructions