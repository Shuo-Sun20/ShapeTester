import keras
import keras.ops as ops

def call_func(inputs):
    return ops.arcsin(inputs)

input_tensor = keras.random.uniform(shape=(3,), minval=-1.0, maxval=1.0)
example_output = call_func(input_tensor)