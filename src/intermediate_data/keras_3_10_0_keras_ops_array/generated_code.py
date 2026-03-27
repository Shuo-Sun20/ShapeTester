import keras
import numpy as np

def call_func(inputs, dtype=None):
    return keras.ops.array(x=inputs, dtype=dtype)

random_tensor = np.random.rand(3, 4)
example_output = call_func(inputs=random_tensor, dtype="float32")