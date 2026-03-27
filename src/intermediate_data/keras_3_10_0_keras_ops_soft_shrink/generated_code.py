import keras
import numpy as np

def call_func(inputs, threshold=0.5):
    return keras.ops.soft_shrink(inputs, threshold)

x = np.random.randn(3, 4).astype(np.float32)
example_output = call_func(x, threshold=0.5)