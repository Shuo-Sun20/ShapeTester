import tensorflow as tf
from dataclasses import dataclass

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 3), dtype=tf.float32),
    "name": None
}

# 2. Only parameter affecting output shape (except "inputs") is "name"
#    But "name" does NOT affect output shape - it's only for operation naming
#    The output shape is solely determined by the batch dimensions of "inputs"
#    Therefore, no parameters in call_func (except "inputs") affect output shape

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # Since no parameters (except "inputs") affect output shape,
    # we only include the mandatory fields
    pass