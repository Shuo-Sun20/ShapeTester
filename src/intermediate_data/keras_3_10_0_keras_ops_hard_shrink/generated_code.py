import keras
import numpy as np

def call_func(inputs, threshold=0.5):
    return keras.ops.hard_shrink(inputs, threshold)

# Generate random input tensor
np.random.seed(42)
x = np.random.randn(3, 4)
example_output = call_func(x, threshold=0.5)