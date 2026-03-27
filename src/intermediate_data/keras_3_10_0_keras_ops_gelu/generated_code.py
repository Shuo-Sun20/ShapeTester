import numpy as np
import keras

def call_func(inputs, approximate=True):
    return keras.ops.gelu(inputs, approximate=approximate)

x = np.random.randn(3, 4).astype(np.float32)
example_output = call_func(x, approximate=True)