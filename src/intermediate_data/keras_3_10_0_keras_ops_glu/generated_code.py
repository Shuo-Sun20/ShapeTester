import keras
import numpy as np

def call_func(inputs, axis=-1):
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    return keras.ops.glu(inputs, axis)

example_input = np.random.randn(4, 8).astype(np.float32)
example_output = call_func(example_input, axis=1)