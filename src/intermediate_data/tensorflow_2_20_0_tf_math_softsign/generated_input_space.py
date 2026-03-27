import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 4)),
    "name": None
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing discretized value ranges for parameters 
       that affect the output tensor shape.
    """
    # Parameter 'name' doesn't affect output shape, so no field needed
    # Only 'inputs' affects shape, but we don't include it per instruction
    pass