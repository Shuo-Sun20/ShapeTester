import keras
import numpy as np

def call_func(inputs, threshold, default_value):
    return keras.ops.threshold(inputs, threshold, default_value)

x = np.random.randn(4, 3).astype(np.float32)
example_output = call_func(x, 0.5, 0.0)