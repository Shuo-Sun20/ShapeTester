import keras
import numpy as np

def call_func(inputs, nan=0.0, posinf=None, neginf=None):
    return keras.ops.nan_to_num(inputs, nan=nan, posinf=posinf, neginf=neginf)

np.random.seed(42)
data = np.random.randn(3, 4).astype(np.float32)
data[0, 0] = np.nan
data[1, 1] = np.inf
data[2, 2] = -np.inf
x = keras.ops.convert_to_tensor(data)
example_output = call_func(x)