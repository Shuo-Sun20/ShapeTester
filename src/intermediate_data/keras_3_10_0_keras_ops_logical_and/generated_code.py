import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs
    return keras.ops.logical_and(x1, x2)

# Generate random tensors for testing
random_tensor_1 = keras.random.uniform(shape=(3, 4), minval=0, maxval=2, seed=42)
random_tensor_2 = keras.random.uniform(shape=(3, 4), minval=0, maxval=2, seed=24)
example_output = call_func([random_tensor_1, random_tensor_2])