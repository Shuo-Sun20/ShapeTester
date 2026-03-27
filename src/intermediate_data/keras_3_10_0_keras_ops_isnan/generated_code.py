import keras
import numpy as np

def call_func(inputs):
    return keras.ops.isnan(x=inputs)

# Generate random tensor with some NaN values
np.random.seed(42)
data = np.random.randn(3, 4).astype(np.float32)
data[0, 1] = np.nan
data[2, 3] = np.nan
x = keras.ops.convert_to_tensor(data)

example_output = call_func(inputs=x)