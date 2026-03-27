import keras
import numpy as np

def call_func(inputs):
    return keras.ops.log2(inputs)

# Generate a random tensor for testing
test_input = keras.random.normal(shape=(3, 4))
example_output = call_func(test_input)