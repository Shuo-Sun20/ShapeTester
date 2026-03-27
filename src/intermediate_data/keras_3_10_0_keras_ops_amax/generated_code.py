import keras.ops
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.amax(x=inputs, axis=axis, keepdims=keepdims)

example_output = call_func(
    inputs=keras.ops.convert_to_tensor(np.random.randn(3, 4)),
    axis=1,
    keepdims=True
)