import keras
import numpy as np

def call_func(inputs, repeats, axis=None):
    return keras.ops.repeat(x=inputs, repeats=repeats, axis=axis)

# Generate random input tensor
input_tensor = keras.random.normal(shape=(3, 4))
example_output = call_func(inputs=input_tensor, repeats=2, axis=1)