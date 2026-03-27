import keras
import numpy as np

def call_func(cond, body, inputs, maximum_iterations=None):
    return keras.ops.while_loop(cond, body, inputs, maximum_iterations)

# Construct a valid input example
start_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
cond = lambda x: keras.ops.sum(x) < 10
body = lambda x: x + 0.5
example_output = call_func(cond, body, start_tensor, maximum_iterations=20)