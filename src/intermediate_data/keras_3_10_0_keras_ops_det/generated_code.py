import keras
import numpy as np

def call_func(inputs):
    return keras.ops.det(inputs)

# Create a random 3x3 matrix for testing
random_matrix = keras.random.normal(shape=(3, 3))
example_output = call_func(random_matrix)