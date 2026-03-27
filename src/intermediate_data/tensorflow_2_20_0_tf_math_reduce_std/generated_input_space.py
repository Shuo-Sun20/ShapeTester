import tensorflow as tf
from dataclasses import dataclass, field
from typing import Any, List, Union, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    'axis': 0,
    'keepdims': False,
    'name': None
}

# Task 2 & 3 & 4: Parameters affecting output shape and their value spaces
@dataclass
class InputSpace:
    """
    Dataclass containing parameters that affect output shape of tf.math.reduce_std.
    Value ranges are discretized according to specification.
    """
    axis: List[Optional[Union[int, List[int]]]] = field(
        default_factory=lambda: [None, 0, 1, [0, 1], -1]
    )
    keepdims: List[bool] = field(default_factory=lambda: [True, False])