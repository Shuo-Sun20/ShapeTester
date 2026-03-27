import keras
import numpy as np

def call_func(inputs):
    return keras.ops.softsign(inputs)

example_output = call_func(keras.ops.convert_to_tensor(np.random.randn(3, 4).astype('float32')))