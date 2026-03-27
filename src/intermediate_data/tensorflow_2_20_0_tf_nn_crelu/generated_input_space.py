import tensorflow as tf
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.constant([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
    "axis": -1,
    "name": None
}

# Task 2, 3 & 4: Define InputSpace
@dataclass
class InputSpace:
    # Parameter 'axis' is the only shape-affecting parameter besides 'inputs'
    # Value space: discrete integers within valid axis range [-rank, rank-1]
    # Selected 5 typical values covering negative, zero, and positive axes
    axis: list = field(default_factory=lambda: [-3, -1, 0, 2, 4])