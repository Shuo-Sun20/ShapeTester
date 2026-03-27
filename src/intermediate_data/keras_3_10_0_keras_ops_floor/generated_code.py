import keras
import numpy as np

def call_func(inputs):
    return keras.ops.floor(inputs)

# Generate random input tensor
random_input = keras.random.uniform(shape=(3, 4), minval=-5, maxval=5)
example_output = call_func(random_input)