import keras
import numpy as np

def call_func(inputs, dtype=None, sparse=None, ragged=None):
    return keras.ops.convert_to_tensor(x=inputs, dtype=dtype, sparse=sparse, ragged=ragged)

x = np.random.rand(3, 4, 5)
example_output = call_func(x, dtype='float32')