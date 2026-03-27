import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.floor_divide(x1, x2)

# Generate random tensors
x1 = keras.random.normal(shape=(2, 3))
x2 = keras.random.normal(shape=(2, 3)) + 0.5  # Avoid division by zero

example_output = call_func([x1, x2])