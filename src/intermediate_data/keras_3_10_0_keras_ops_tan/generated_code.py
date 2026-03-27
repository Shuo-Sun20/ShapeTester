import keras
import numpy as np

def call_func(inputs):
    return keras.ops.tan(inputs)

# Create random input tensor
random_tensor = keras.random.normal(shape=(3, 4, 5))
example_output = call_func(random_tensor)