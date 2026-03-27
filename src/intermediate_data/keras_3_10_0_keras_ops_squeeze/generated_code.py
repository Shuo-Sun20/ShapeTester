import keras
import numpy as np

def call_func(inputs, axis=None):
    x = inputs[0]
    return keras.ops.squeeze(x, axis)

# Construct a random tensor with shape (1, 3, 1, 5)
x = keras.random.normal(shape=(1, 3, 1, 5))
example_output = call_func(inputs=[x])