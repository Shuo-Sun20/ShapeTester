import keras
import numpy as np

def call_func(inputs, axis=None, dtype=None):
    return keras.ops.cumprod(x=inputs, axis=axis, dtype=dtype)

np.random.seed(42)
random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func(random_tensor, axis=0)