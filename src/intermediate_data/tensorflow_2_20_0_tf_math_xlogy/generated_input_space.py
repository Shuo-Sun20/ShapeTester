import tensorflow as tf
from dataclasses import dataclass, field

# 1. Define valid_test_case
x = tf.random.uniform(shape=(3, 3), dtype=tf.float32)
y = tf.random.uniform(shape=(3, 3), minval=0.1, dtype=tf.float32)
valid_test_case = {
    "inputs": [x, y],
    "name": None
}

# 2. Parameters affecting output tensor shape: only "inputs" (explicitly excluded per instructions).
#    No other parameters in call_func() affect shape.

# 3. & 4. Define InputSpace dataclass for parameters that affect shape.
#    Since no such parameters exist (excluding "inputs"), define empty class.
@dataclass
class InputSpace:
    # No fields required as no parameters affect shape beyond "inputs"
    pass

# This class can be instantiated as var=InputSpace()