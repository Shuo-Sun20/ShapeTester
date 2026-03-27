import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, axes, shift=None, keepdims=False, name=None):
    """
    Calls tf.nn.moments with given parameters.
    
    Args:
        inputs: List containing a single input tensor [x] for tf.nn.moments
        axes: Axes along which to compute mean and variance
        shift: Not used in current implementation (kept for compatibility)
        keepdims: Produce moments with same dimensionality as input
        name: Name used to scope operations
    Returns:
        List containing two tensors: [mean, variance]
    """
    x = inputs[0]
    mean, variance = tf.nn.moments(x, axes=axes, shift=shift, keepdims=keepdims, name=name)
    return [mean, variance]

# 1. Define valid_test_case dictionary
valid_test_case = {
    "inputs": [tf.constant(np.random.randn(2, 4, 4, 3), dtype=tf.float32)],
    "axes": [0, 1, 2],
    "keepdims": False,
    "shift": None,
    "name": None
}

# 2. Parameters affecting output shape: axes, keepdims

# 3. Value space analysis:
# axes: List[int] - discrete parameter, possible values include:
#   - Single axis: [0], [1], [2], [3]
#   - Multiple axes: [0,1], [1,2], [0,1,2], [0,3]
#   Boundary cases: empty list [], full list [0,1,2,3]
# keepdims: bool - discrete parameter with two values

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    axes: List[List[int]] = field(
        default_factory=lambda: [
            [0],                # batch dimension only
            [0, 1],             # batch and height
            [0, 1, 2],          # batch, height, width
            [0, 3],             # batch and channels
            [0, 1, 2, 3],       # all dimensions
        ]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [False, True]
    )