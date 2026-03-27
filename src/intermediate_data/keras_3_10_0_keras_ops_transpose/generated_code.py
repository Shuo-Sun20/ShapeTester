import keras
import numpy as np

def call_func(inputs, axes=None):
    return keras.ops.transpose(x=inputs, axes=axes)

# Create a random 3D tensor with shape (2, 3, 4) as input
example_input = keras.random.normal(shape=(2, 3, 4))
# Call the function without specifying axes (default reversal)
example_output = call_func(inputs=example_input)