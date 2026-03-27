import keras
import numpy as np

def call_func(inputs):
    return keras.ops.tanh(inputs)

random_tensor = keras.ops.convert_to_tensor(np.random.randn(2, 3), dtype='float32')
example_output = call_func(random_tensor)