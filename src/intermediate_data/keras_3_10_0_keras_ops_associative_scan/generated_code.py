import keras
import numpy as np

def call_func(f, inputs, reverse=False, axis=0):
    return keras.ops.associative_scan(f, inputs, reverse=reverse, axis=axis)

# Generate random input tensor
xs = keras.ops.convert_to_tensor(np.random.rand(5))
sum_fn = lambda x, y: x + y
example_output = call_func(sum_fn, xs, axis=0)