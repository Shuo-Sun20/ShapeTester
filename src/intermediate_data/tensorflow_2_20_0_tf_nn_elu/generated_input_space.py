import tensorflow as tf
from dataclasses import dataclass

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': tf.random.normal(shape=(3, 4), dtype=tf.float32),
    'name': None
}

# Task 2, 3, and 4: Define InputSpace
@dataclass
class InputSpace:
    # The only parameter in call_func that can affect the shape of the output tensor 
    # (besides 'inputs') is 'name'. However, the 'name' parameter does not affect 
    # the shape of the tensor; it only provides a name for the operation in the graph.
    # Therefore, no parameters besides 'inputs' affect the output shape.
    # Since we must include all parameters that affect shape (excluding 'inputs'), 
    # and there are none, we create an empty dataclass.
    pass