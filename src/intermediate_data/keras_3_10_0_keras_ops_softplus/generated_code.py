import keras
import numpy as np

def call_func(inputs):
    return keras.ops.softplus(inputs)

random_input = keras.ops.convert_to_tensor(np.random.randn(3))
example_output = call_func(random_input)