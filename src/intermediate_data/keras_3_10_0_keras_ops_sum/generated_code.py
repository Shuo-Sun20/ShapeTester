import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.sum(x=inputs, axis=axis, keepdims=keepdims)

# Construct valid input
random_tensor = keras.ops.convert_to_tensor(np.random.rand(3, 4, 5))
example_output = call_func(random_tensor, axis=1, keepdims=True)