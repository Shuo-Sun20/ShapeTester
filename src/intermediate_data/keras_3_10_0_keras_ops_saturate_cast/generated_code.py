import keras.ops as ops
import numpy as np

def call_func(inputs, dtype):
    return ops.saturate_cast(inputs, dtype)

random_tensor = np.random.uniform(-10, 300, size=(3, 4, 5)).astype("float32")
example_output = call_func(random_tensor, "uint8")