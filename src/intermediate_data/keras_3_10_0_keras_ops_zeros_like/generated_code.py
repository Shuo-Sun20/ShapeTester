import keras
import numpy as np

def call_func(inputs, dtype=None):
    return keras.ops.zeros_like(x=inputs, dtype=dtype)

random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
example_output = call_func(inputs=random_tensor, dtype="float32")