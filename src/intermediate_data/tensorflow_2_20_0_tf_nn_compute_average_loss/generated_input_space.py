import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case with all parameters for call_func
valid_test_case = {
    'inputs': [tf.random.normal(shape=[4, 1])],
    'sample_weight': tf.random.uniform(shape=[4, 1]),
    'global_batch_size': 4
}

# 2. Parameters affecting output shape (excluding "inputs"):
# - sample_weight: Can be None or a tensor, affects value but not shape (output is scalar)
# - global_batch_size: Integer or None, affects value but not shape (output is scalar)
# Output is always scalar (shape []) regardless of parameters

# 3. Value spaces for parameters that affect output value:
# sample_weight: None or tensor with broadcastable shape
# global_batch_size: None, 0, or positive integers

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    sample_weight: List[Optional[tf.Tensor]] = field(default_factory=lambda: [
        None,
        tf.ones(shape=[4, 1]),
        tf.zeros(shape=[4, 1]),
        tf.constant([[0.5], [1.5], [2.5], [3.5]]),
        tf.constant(2.0)  # scalar that broadcasts
    ])
    
    global_batch_size: List[Optional[int]] = field(default_factory=lambda: [
        None,
        0,
        1,
        4,
        8
    ])