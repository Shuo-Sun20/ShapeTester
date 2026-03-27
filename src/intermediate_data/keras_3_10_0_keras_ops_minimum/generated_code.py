import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs
    return keras.ops.minimum(x1, x2)

np.random.seed(42)
example_inputs = [keras.ops.convert_to_tensor(np.random.randn(3, 4)), keras.ops.convert_to_tensor(np.random.randn(3, 4))]
example_output = call_func(example_inputs)