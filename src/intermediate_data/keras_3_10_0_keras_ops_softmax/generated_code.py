import keras
import numpy as np

def call_func(inputs, axis=-1):
    return keras.ops.softmax(inputs, axis=axis)

random_tensor = np.random.randn(3, 4)
example_output = call_func(random_tensor, axis=1)