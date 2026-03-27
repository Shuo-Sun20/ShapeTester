import keras

def call_func(inputs):
    return keras.ops.absolute(inputs)

# Generate a random tensor as input
import numpy as np
random_input = keras.random.normal(shape=(3, 4))
example_output = call_func(random_input)