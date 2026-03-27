import keras
import numpy as np

def call_func(inputs):
    return keras.ops.hard_tanh(inputs)

x = np.random.uniform(-2, 2, size=(5,))
example_output = call_func(x)