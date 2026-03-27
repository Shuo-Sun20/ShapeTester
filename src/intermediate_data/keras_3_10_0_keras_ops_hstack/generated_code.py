import keras
import numpy as np

def call_func(inputs):
    return keras.ops.hstack(inputs)

# Generate random tensors for testing
tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(3, 2))
example_output = call_func([tensor1, tensor2])