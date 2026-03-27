import keras
import numpy as np

def call_func(inputs, axis=-1):
    return keras.ops.sparsemax(inputs, axis=axis)

np.random.seed(42)
example_tensor = np.random.randn(3, 5).astype(np.float32)
example_output = call_func(example_tensor, axis=-1)