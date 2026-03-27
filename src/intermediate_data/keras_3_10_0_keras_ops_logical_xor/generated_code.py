import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.logical_xor(x1, x2)

tensor_a = keras.ops.convert_to_tensor(np.random.choice([True, False], size=(3, 4)))
tensor_b = keras.ops.convert_to_tensor(np.random.choice([True, False], size=(3, 4)))
example_output = call_func([tensor_a, tensor_b])