import keras
import numpy as np

def call_func(inputs, newshape):
    return keras.ops.reshape(inputs, newshape)

# Create random input tensor
random_tensor = keras.random.normal(shape=(6, 8))
example_output = call_func(random_tensor, newshape=(12, 4))