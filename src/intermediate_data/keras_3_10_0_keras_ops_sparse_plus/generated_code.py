import keras
import numpy as np

def call_func(inputs):
    return keras.ops.sparse_plus(inputs[0])

example_input = np.random.randn(5).astype(np.float32)
example_output = call_func([example_input])