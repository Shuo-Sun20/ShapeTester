import keras
import numpy as np

def call_func(shape, dtype=None, inputs=None):
    return keras.ops.zeros(shape, dtype)

# Generate random shape for tensor
random_shape = np.random.randint(1, 10, size=np.random.randint(1, 4)).tolist()
example_output = call_func(shape=random_shape, dtype="float32")