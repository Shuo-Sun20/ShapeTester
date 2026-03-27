import keras
import numpy as np

def call_func(inputs, axis=-1):
    return keras.ops.argsort(x=inputs, axis=axis)

example_input = keras.ops.array(np.random.randint(0, 10, size=(3, 4)))
example_output = call_func(example_input, axis=1)