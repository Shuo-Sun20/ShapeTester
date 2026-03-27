import tensorflow as tf
from dataclasses import dataclass, field

# 1. Define valid_test_case
def call_func(inputs, name="is_strictly_increasing"):
    return tf.math.is_strictly_increasing(x=inputs, name=name)

tf.random.set_seed(42)
random_tensor = tf.random.uniform(shape=(5,), minval=0, maxval=10, dtype=tf.float32)

valid_test_case = {
    "inputs": random_tensor,
    "name": "is_strictly_increasing"
}

# 2. Identify parameters affecting output shape (excluding inputs)
# Only 'inputs' affects the output tensor's shape. Other parameters like 'name' 
# do not affect the output tensor's shape (output is always scalar).
shape_affecting_params_excluding_inputs = []  # Empty list

# 3. Value spaces for shape-affecting parameters (excluding inputs)
# Since there are no shape-affecting parameters besides 'inputs', 
# we define an empty value space.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # No fields are defined because there are no parameters 
    # (excluding 'inputs') that affect the output tensor's shape.
    # The class is instantiable without any arguments.
    pass

# Example instantiation
var = InputSpace()