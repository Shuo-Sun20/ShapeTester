import keras
import numpy as np

def call_func(inputs, alpha=1.0):
    return keras.ops.celu(inputs, alpha)

example_output = call_func(np.random.randn(3, 4))