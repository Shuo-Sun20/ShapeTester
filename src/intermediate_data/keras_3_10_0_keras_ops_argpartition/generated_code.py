import keras
import numpy as np

def call_func(inputs, kth, axis=-1):
    return keras.ops.argpartition(inputs, kth, axis)

# Generate random input tensor
np.random.seed(42)
input_tensor = keras.random.normal(shape=(4, 4, 4))

# Call the function with valid parameters
example_output = call_func(
    inputs=input_tensor,
    kth=2,
    axis=-1
)